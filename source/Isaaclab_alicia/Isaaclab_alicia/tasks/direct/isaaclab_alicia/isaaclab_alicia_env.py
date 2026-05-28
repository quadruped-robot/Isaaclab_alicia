from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply

from .isaaclab_alicia_env_cfg import IsaaclabAliciaEnvCfg

_TARGET_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/TargetSphere",
    markers={
        "target": sim_utils.SphereCfg(
            radius=0.015,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
    },
)


class IsaaclabAliciaEnv(DirectRLEnv):
    cfg: IsaaclabAliciaEnvCfg

    def __init__(self, cfg: IsaaclabAliciaEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._arm_dof_idx, _ = self.robot.find_joints(self.cfg.arm_joint_names, preserve_order=True)
        self._gripper_dof_idx, _ = self.robot.find_joints(self.cfg.gripper_joint_names, preserve_order=True)
        self._all_dof_idx = self._arm_dof_idx + self._gripper_dof_idx
        self._ee_body_idx = self.robot.find_bodies(self.cfg.ee_body_name)[0][0]

        self._ee_offset_local = torch.tensor(self.cfg.ee_offset_local, device=self.device).repeat(self.num_envs, 1)

        n = self.num_envs
        self._row_ids = torch.arange(n, device=self.device, dtype=torch.long)
        self._obs_dim = int(self.cfg.observation_space)

        self.target_pos = torch.zeros(n, 3, device=self.device)
        self.target_pos_local = torch.zeros(n, 3, device=self.device)
        self.prev_actions = torch.zeros(n, 8, device=self.device)
        self.actions = torch.zeros(n, 8, device=self.device)
        self.applied_torques = torch.zeros(n, 8, device=self.device)
        self.prev_joint_vel = torch.zeros(n, 8, device=self.device)
        self.dt = self.cfg.sim.dt * self.cfg.decimation

        self._ee_pos_w = torch.zeros(n, 3, device=self.device)
        self._ee_vel_w = torch.zeros(n, 3, device=self.device)
        self._distance = torch.full((n,), float("inf"), device=self.device)

        self._close_thresholds = torch.tensor(self.cfg.rew_close_phase_thresholds, device=self.device)
        self._close_rewards = torch.tensor(self.cfg.rew_close_phase_rewards, device=self.device)
        self.close_phase_given = torch.zeros(n, len(self.cfg.rew_close_phase_thresholds), dtype=torch.bool, device=self.device)

        self._gripper_forward_local = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(n, 1)

        self._max_obs_latency = self._validate_delay_range(
            "obs_latency_steps", self.cfg.obs_latency_steps_min, self.cfg.obs_latency_steps_max
        )
        self._max_act_latency = self._validate_delay_range(
            "act_latency_steps", self.cfg.act_latency_steps_min, self.cfg.act_latency_steps_max
        )
        self._obs_latency_steps = torch.zeros(n, dtype=torch.long, device=self.device)
        self._act_latency_steps = torch.zeros(n, dtype=torch.long, device=self.device)
        self._obs_hist = torch.zeros(n, self._max_obs_latency + 1, self._obs_dim, device=self.device)
        self._act_hist = torch.zeros(n, self._max_act_latency + 1, 8, device=self.device)

        self._actuator_arm_gain = torch.ones(n, 1, device=self.device)
        self._actuator_gripper_gain = torch.ones(n, 1, device=self.device)
        self._obs_joint_pos_bias = torch.zeros(n, 8, device=self.device)
        self._obs_joint_vel_bias = torch.zeros(n, 8, device=self.device)
        self._gripper_close_latched = torch.zeros(n, dtype=torch.bool, device=self.device)

    @staticmethod
    def _validate_delay_range(name: str, min_steps: int, max_steps: int) -> int:
        if min_steps < 0 or max_steps < 0:
            raise ValueError(f"{name} must be >= 0, got ({min_steps}, {max_steps})")
        if min_steps > max_steps:
            raise ValueError(f"{name} min must be <= max, got ({min_steps}, {max_steps})")
        return int(max_steps)

    def _sample_uniform(self, count: int, value_range: tuple[float, float]) -> torch.Tensor:
        low, high = float(value_range[0]), float(value_range[1])
        if high < low:
            raise ValueError(f"Invalid range: ({low}, {high})")
        values = torch.empty(count, device=self.device)
        if low == high:
            values.fill_(low)
        else:
            values.uniform_(low, high)
        return values

    def _sample_int(self, count: int, min_val: int, max_val: int) -> torch.Tensor:
        if min_val < 0 or max_val < 0:
            raise ValueError(f"Latency must be >= 0, got ({min_val}, {max_val})")
        if min_val > max_val:
            raise ValueError(f"Latency min must be <= max, got ({min_val}, {max_val})")
        if min_val == max_val:
            return torch.full((count,), min_val, device=self.device, dtype=torch.long)
        return torch.randint(min_val, max_val + 1, (count,), device=self.device, dtype=torch.long)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        self.target_marker = VisualizationMarkers(_TARGET_MARKER_CFG)

    def _apply_sim2real_randomization(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return

        if not self.cfg.enable_sim2real_randomization:
            self._obs_latency_steps[env_ids] = 0
            self._act_latency_steps[env_ids] = 0
            self._actuator_arm_gain[env_ids] = 1.0
            self._actuator_gripper_gain[env_ids] = 1.0
            self._obs_joint_pos_bias[env_ids] = 0.0
            self._obs_joint_vel_bias[env_ids] = 0.0
            return

        count = env_ids.numel()
        self._obs_latency_steps[env_ids] = self._sample_int(
            count, self.cfg.obs_latency_steps_min, self.cfg.obs_latency_steps_max
        )
        self._act_latency_steps[env_ids] = self._sample_int(
            count, self.cfg.act_latency_steps_min, self.cfg.act_latency_steps_max
        )
        self._actuator_arm_gain[env_ids, 0] = self._sample_uniform(count, self.cfg.actuator_arm_gain_range)
        self._actuator_gripper_gain[env_ids, 0] = self._sample_uniform(count, self.cfg.actuator_gripper_gain_range)

        pos_low, pos_high = self.cfg.obs_joint_pos_bias_range
        vel_low, vel_high = self.cfg.obs_joint_vel_bias_range
        self._obs_joint_pos_bias[env_ids] = torch.empty(count, 8, device=self.device).uniform_(pos_low, pos_high)
        self._obs_joint_vel_bias[env_ids] = torch.empty(count, 8, device=self.device).uniform_(vel_low, vel_high)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        clamped_actions = actions.clone().clamp(-1.0, 1.0)

        self._act_hist = torch.roll(self._act_hist, shifts=1, dims=1)
        self._act_hist[:, 0] = clamped_actions

        if self.cfg.enable_sim2real_randomization:
            delayed_actions = self._act_hist[self._row_ids, self._act_latency_steps]
        else:
            delayed_actions = clamped_actions

        dist_for_control = torch.where(
            torch.isfinite(self._distance), self._distance, torch.full_like(self._distance, float("inf"))
        )

        if self.cfg.enable_sim2real_randomization and self.cfg.action_dropout_prob > 0.0:
            dropout_mask = torch.rand((self.num_envs, 1), device=self.device) < self.cfg.action_dropout_prob
            if self.cfg.randomization_near_disable_dist > 0.0:
                far_mask = dist_for_control > self.cfg.randomization_near_disable_dist
                dropout_mask = dropout_mask & far_mask.unsqueeze(-1)
            delayed_actions = torch.where(dropout_mask, self.prev_actions, delayed_actions)

        if self.cfg.enable_sim2real_randomization and self.cfg.action_noise_std > 0.0:
            action_noise = torch.randn_like(delayed_actions) * self.cfg.action_noise_std
            if self.cfg.randomization_near_disable_dist > 0.0:
                near_noise_scale_value = max(float(self.cfg.randomization_near_noise_scale), 0.0)
                near_noise_scale = torch.where(
                    dist_for_control < self.cfg.randomization_near_disable_dist,
                    torch.full_like(dist_for_control, near_noise_scale_value),
                    torch.ones_like(dist_for_control),
                )
                action_noise = action_noise * near_noise_scale.unsqueeze(-1)
            delayed_actions = delayed_actions + action_noise

        arm_actions = delayed_actions[:, :6]
        if self.cfg.near_target_dist > 0.0:
            near_ratio = torch.clamp(dist_for_control / self.cfg.near_target_dist, min=0.0, max=1.0)
            near_target_scale = min(max(float(self.cfg.near_target_action_scale), 0.0), 1.0)
            arm_scale = near_target_scale + (1.0 - near_target_scale) * near_ratio
            arm_actions = arm_actions * arm_scale.unsqueeze(-1)
        near_target_deadband = max(float(self.cfg.near_target_action_deadband), 0.0)
        if near_target_deadband > 0.0:
            arm_actions = torch.where(
                arm_actions.abs() < near_target_deadband,
                torch.zeros_like(arm_actions),
                arm_actions,
            )
        delayed_actions = delayed_actions.clone()
        delayed_actions[:, :6] = arm_actions

        self.actions = delayed_actions.clamp(-1.0, 1.0)

        torques = torch.empty_like(self.actions)
        torques[:, :6] = self.actions[:, :6] * self.cfg.action_scale_arm * self._actuator_arm_gain
        gripper_q = self.robot.data.joint_pos[:, self._gripper_dof_idx]
        gripper_dq = self.robot.data.joint_vel[:, self._gripper_dof_idx]
        gripper_hold = -self.cfg.gripper_hold_kp * gripper_q - self.cfg.gripper_hold_kd * gripper_dq
        gripper_torque_limit = max(float(self.cfg.gripper_hold_torque_limit), 0.0)
        if gripper_torque_limit > 0.0:
            gripper_hold = gripper_hold.clamp(-gripper_torque_limit, gripper_torque_limit)
        else:
            gripper_hold = torch.zeros_like(gripper_hold)

        if self.cfg.enable_auto_gripper_close:
            close_trigger_dist = max(float(self.cfg.gripper_close_trigger_dist), 0.0)
            reopen_trigger_dist = max(float(self.cfg.gripper_reopen_trigger_dist), close_trigger_dist)
            should_close = dist_for_control < close_trigger_dist
            should_reopen = dist_for_control > reopen_trigger_dist
            self._gripper_close_latched = torch.where(
                should_reopen,
                torch.zeros_like(self._gripper_close_latched),
                self._gripper_close_latched | should_close,
            )

            close_torque = max(float(self.cfg.gripper_close_torque), 0.0)
            if close_torque > 0.0:
                close_vec = torch.tensor([-close_torque, close_torque], device=self.device).unsqueeze(0)
                close_mask = self._gripper_close_latched.unsqueeze(-1)
                gripper_hold = torch.where(close_mask, close_vec, gripper_hold)

        torques[:, 6:] = gripper_hold * self._actuator_gripper_gain
        self.applied_torques = torques

    def _apply_action(self) -> None:
        self.robot.set_joint_effort_target(self.applied_torques, joint_ids=self._all_dof_idx)

    def _update_ee_state(self) -> None:
        link6_pos_w = self.robot.data.body_pos_w[:, self._ee_body_idx]
        link6_quat_w = self.robot.data.body_quat_w[:, self._ee_body_idx]

        offset_world = quat_apply(link6_quat_w, self._ee_offset_local)
        self._ee_pos_w = link6_pos_w + offset_world

        link6_lin_vel = self.robot.data.body_lin_vel_w[:, self._ee_body_idx]
        link6_ang_vel = self.robot.data.body_ang_vel_w[:, self._ee_body_idx]
        self._ee_vel_w = link6_lin_vel + torch.linalg.cross(link6_ang_vel, offset_world, dim=-1)

        self._distance = torch.linalg.norm(self._ee_pos_w - self.target_pos, dim=-1)

    def _compose_observation(self) -> torch.Tensor:
        joint_pos = self.robot.data.joint_pos[:, self._all_dof_idx]
        joint_vel = self.robot.data.joint_vel[:, self._all_dof_idx]
        relative_pos = self.target_pos - self._ee_pos_w
        ee_vel = self._ee_vel_w

        if self.cfg.enable_sim2real_randomization:
            joint_pos = joint_pos + self._obs_joint_pos_bias
            joint_vel = joint_vel + self._obs_joint_vel_bias

            if self.cfg.obs_joint_pos_noise_std > 0.0:
                joint_pos = joint_pos + torch.randn_like(joint_pos) * self.cfg.obs_joint_pos_noise_std
            if self.cfg.obs_joint_vel_noise_std > 0.0:
                joint_vel = joint_vel + torch.randn_like(joint_vel) * self.cfg.obs_joint_vel_noise_std
            if self.cfg.obs_target_pos_noise_std > 0.0:
                relative_pos = relative_pos + torch.randn_like(relative_pos) * self.cfg.obs_target_pos_noise_std
            if self.cfg.obs_ee_vel_noise_std > 0.0:
                ee_vel = ee_vel + torch.randn_like(ee_vel) * self.cfg.obs_ee_vel_noise_std

        return torch.cat((relative_pos, joint_pos, joint_vel, self.prev_actions, ee_vel), dim=-1)

    def _get_observations(self) -> dict:
        self._update_ee_state()
        raw_obs = self._compose_observation()

        self._obs_hist = torch.roll(self._obs_hist, shifts=1, dims=1)
        self._obs_hist[:, 0] = raw_obs

        if self.cfg.enable_sim2real_randomization:
            obs = self._obs_hist[self._row_ids, self._obs_latency_steps]
        else:
            obs = raw_obs
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        c = self.cfg
        dist = self._distance
        rew = torch.full_like(dist, -c.rew_step_penalty)

        inner = 1.0 / (1.0 + (c.rew_dist_beta * dist) ** 2)
        dist_reward = c.rew_dist_gain * inner * inner
        close_mask = dist < c.close_threshold
        succ_mask = dist < c.success_threshold
        dist_reward = torch.where(close_mask, dist_reward * c.rew_close_multiplier, dist_reward)
        dist_reward = torch.where(succ_mask, dist_reward * c.rew_success_multiplier, dist_reward)
        rew = rew + dist_reward

        to_target = self.target_pos - self._ee_pos_w
        to_target_norm = torch.linalg.norm(to_target, dim=-1, keepdim=True).clamp_min(1e-6)
        to_target_unit = to_target / to_target_norm
        vel_norm = torch.linalg.norm(self._ee_vel_w, dim=-1, keepdim=True).clamp_min(1e-6)
        vel_cos = (to_target_unit * (self._ee_vel_w / vel_norm)).sum(dim=-1)
        direction_gate = (dist > c.success_threshold).float()
        rew = rew + direction_gate * c.rew_direction_gain * torch.clamp(vel_cos, min=0.0) ** 2

        link6_quat_w = self.robot.data.body_quat_w[:, self._ee_body_idx]
        gripper_forward_w = quat_apply(link6_quat_w, self._gripper_forward_local)
        align_cos = (gripper_forward_w * to_target_unit).sum(dim=-1)
        rew = rew + c.rew_gripper_align_gain * align_cos * align_cos.abs()

        ee_speed = torch.linalg.norm(self._ee_vel_w, dim=-1)
        rew = rew - torch.where(
            ee_speed > c.rew_speed_threshold,
            torch.full_like(ee_speed, c.rew_speed_penalty),
            torch.zeros_like(ee_speed),
        )
        speed_excess = torch.clamp(ee_speed - c.rew_speed_threshold, min=0.0)
        rew = rew - c.rew_speed_excess * speed_excess

        cur_joint_vel = self.robot.data.joint_vel[:, self._all_dof_idx]
        joint_vel_change = (cur_joint_vel - self.prev_joint_vel).abs().sum(dim=-1)
        rew = rew - c.rew_joint_vel_change * joint_vel_change
        self.prev_joint_vel = cur_joint_vel.clone()

        action_rate = (self.actions - self.prev_actions).abs().sum(dim=-1)
        rew = rew - c.rew_action_rate * action_rate

        wrist_action_abs = self.actions[:, 3:6].abs().sum(dim=-1)
        rew = rew - c.rew_wrist_action * wrist_action_abs

        wrist_q = self.robot.data.joint_pos[:, self._arm_dof_idx[3:6]]
        wrist_qvel = self.robot.data.joint_vel[:, self._arm_dof_idx[3:6]]
        wrist_pos_pen = wrist_q.abs().sum(dim=-1)
        wrist_vel_pen = wrist_qvel.abs().sum(dim=-1)
        rew = rew - c.rew_wrist_pos * wrist_pos_pen - c.rew_wrist_vel * wrist_vel_pen

        gripper_q = self.robot.data.joint_pos[:, self._gripper_dof_idx]
        gripper_asym = (gripper_q[:, 0] + gripper_q[:, 1]).abs()
        rew = rew - c.rew_gripper_asymmetry * gripper_asym

        crossed = (dist.unsqueeze(-1) < self._close_thresholds) & (~self.close_phase_given)
        rew = rew + (crossed.float() * self._close_rewards).sum(dim=-1)
        self.close_phase_given = self.close_phase_given | crossed

        settled = succ_mask & (ee_speed < c.rew_settle_speed_threshold)
        rew = rew + c.rew_settle_bonus * settled.float()

        self.extras["log"] = {
            "rew/distance_mean": dist.mean(),
            "rew/success_rate": succ_mask.float().mean(),
            "rew/settled_rate": settled.float().mean(),
            "rew/align_cos_mean": align_cos.mean(),
            "diag/wrist_pos_l1": wrist_pos_pen.mean(),
            "diag/wrist_vel_l1": wrist_vel_pen.mean(),
            "diag/gripper_asymmetry": gripper_asym.mean(),
            "diag/action_rate_l1": action_rate.mean(),
            "diag/gripper_auto_close_rate": self._gripper_close_latched.float().mean(),
            "diag/actuator_arm_gain": self._actuator_arm_gain.mean(),
            "diag/obs_latency_steps": self._obs_latency_steps.float().mean(),
            "diag/act_latency_steps": self._act_latency_steps.float().mean(),
            "rew/total_mean": rew.mean(),
        }
        self.prev_actions = self.actions.clone()
        return rew

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = torch.zeros_like(time_out)
        return terminated, time_out

    def _sample_targets(self, env_ids: torch.Tensor) -> torch.Tensor:
        n = env_ids.numel()
        c = self.cfg

        r = torch.empty(n, device=self.device).uniform_(c.target_r_min, c.target_r_max)
        elev = torch.empty(n, device=self.device).uniform_(c.target_elev_min, c.target_elev_max)
        az = torch.empty(n, device=self.device).uniform_(-math.pi, math.pi)

        cos_e, sin_e = torch.cos(elev), torch.sin(elev)
        cos_a, sin_a = torch.cos(az), torch.sin(az)
        local = torch.stack(
            [
                r * cos_e * cos_a,
                r * cos_e * sin_a,
                r * sin_e + c.target_center_z,
            ],
            dim=-1,
        )
        local[:, 2] = local[:, 2].clamp_min(c.target_z_min)
        return local + self.scene.env_origins[env_ids]

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        targets_world = self._sample_targets(env_ids)
        self.target_pos[env_ids] = targets_world
        self.target_pos_local[env_ids] = targets_world - self.scene.env_origins[env_ids]

        self.prev_actions[env_ids] = 0.0
        self.actions[env_ids] = 0.0
        self.prev_joint_vel[env_ids] = 0.0
        self.close_phase_given[env_ids] = False
        self._gripper_close_latched[env_ids] = False

        self._apply_sim2real_randomization(env_ids)

        self._act_hist[env_ids] = 0.0
        self._obs_hist[env_ids] = 0.0

        self._update_ee_state()
        reset_obs = self._compose_observation()
        self._obs_hist[env_ids] = reset_obs[env_ids].unsqueeze(1).repeat(1, self._max_obs_latency + 1, 1)

        self.target_marker.visualize(translations=self.target_pos)
