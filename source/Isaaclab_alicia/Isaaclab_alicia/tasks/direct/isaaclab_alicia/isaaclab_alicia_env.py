# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Vectorised re-implementation of the original MuJoCo + SB3-TD3 RobotArmEnv
# for IsaacLab. First phase: reach-only.
#
# Mapping to the original `robot_arm_env.py` (run2 / run5 baseline):
#   - target sampling -> workspace-shell sampling around the base
#     (true FK probing requires a physics step; see _sample_targets)
#   - 30-dim obs (rel_pos | q | qdot | prev_action | ee_vel)
#   - reward terms identical except gripper-related ones are skipped
#   - termination: success (dist <= 2cm) | out-of-bounds | timeout

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

    # ------------------------------------------------------------------ #
    # Scene + buffers                                                    #
    # ------------------------------------------------------------------ #

    def __init__(self, cfg: IsaaclabAliciaEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Resolve joint / body indices once.
        self._arm_dof_idx, _ = self.robot.find_joints(self.cfg.arm_joint_names, preserve_order=True)
        self._gripper_dof_idx, _ = self.robot.find_joints(self.cfg.gripper_joint_names, preserve_order=True)
        self._all_dof_idx = self._arm_dof_idx + self._gripper_dof_idx  # 8 entries
        self._ee_body_idx = self.robot.find_bodies(self.cfg.ee_body_name)[0][0]

        # End-effector offset in Link6 frame (since tool0 was merged).
        self._ee_offset_local = torch.tensor(self.cfg.ee_offset_local, device=self.device).repeat(self.num_envs, 1)

        # Per-step buffers (allocate once, never reallocate).
        n = self.num_envs
        self.target_pos = torch.zeros(n, 3, device=self.device)             # in world frame
        self.target_pos_local = torch.zeros(n, 3, device=self.device)       # relative to env origin
        self.prev_actions = torch.zeros(n, 8, device=self.device)
        self.actions = torch.zeros(n, 8, device=self.device)                # raw policy output (~[-1,1])
        self.applied_torques = torch.zeros(n, 8, device=self.device)        # after scale + hardmask (future)
        self.prev_joint_vel = torch.zeros(n, 8, device=self.device)
        # control dt (sim_dt * decimation) for downstream maths
        self.dt = self.cfg.sim.dt * self.cfg.decimation
        # Cached every step in _get_observations so _get_rewards/_get_dones can reuse.
        self._ee_pos_w = torch.zeros(n, 3, device=self.device)
        self._ee_vel_w = torch.zeros(n, 3, device=self.device)
        self._distance = torch.full((n,), float("inf"), device=self.device)

        # Per-cm one-shot bonus state. Each env tracks which close tier it
        # has already collected so backing away from the target doesn't let
        # the policy re-claim the same reward.
        self._close_thresholds = torch.tensor(self.cfg.rew_close_phase_thresholds, device=self.device)
        self._close_rewards = torch.tensor(self.cfg.rew_close_phase_rewards, device=self.device)
        self.close_phase_given = torch.zeros(n, len(self.cfg.rew_close_phase_thresholds),
                                             dtype=torch.bool, device=self.device)

        # Local +z axis of Link6 = direction the gripper points (toward
        # tool0). Used in the gripper-alignment reward.
        self._gripper_forward_local = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(n, 1)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        # green sphere drawn at each target_pos (one marker per env)
        self.target_marker = VisualizationMarkers(_TARGET_MARKER_CFG)

    # ------------------------------------------------------------------ #
    # Action pipeline                                                    #
    # ------------------------------------------------------------------ #

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Keep the *previous* policy output for the next obs and for the
        # `joint_velocity_change` reward; only update after _get_observations
        # has consumed it (matches the original MuJoCo step ordering).
        self.actions = actions.clone().clamp(-1.0, 1.0)
        torques = torch.empty_like(self.actions)
        torques[:, :6] = self.actions[:, :6] * self.cfg.action_scale_arm
        # Phase 1 (reach-only): gripper effort is hard-zeroed regardless of
        # policy output. The URDF->USD pipeline doesn't preserve MuJoCo's
        # <equality> coupling between left/right fingers, so unmasked
        # gripper torques produce asymmetric flailing that the reach reward
        # has no incentive to suppress.
        torques[:, 6:] = 0.0
        self.applied_torques = torques

    def _apply_action(self) -> None:
        # joint_ids must be passed positionally as a tuple/list in IsaacLab.
        self.robot.set_joint_effort_target(self.applied_torques, joint_ids=self._all_dof_idx)

    # ------------------------------------------------------------------ #
    # Observation                                                        #
    # ------------------------------------------------------------------ #

    def _update_ee_state(self) -> None:
        """Recompute end-effector world position + linear velocity (tool0 frame)."""
        link6_pos_w = self.robot.data.body_pos_w[:, self._ee_body_idx]
        link6_quat_w = self.robot.data.body_quat_w[:, self._ee_body_idx]
        offset_world = quat_apply(link6_quat_w, self._ee_offset_local)
        self._ee_pos_w = link6_pos_w + offset_world

        link6_lin_vel = self.robot.data.body_lin_vel_w[:, self._ee_body_idx]
        link6_ang_vel = self.robot.data.body_ang_vel_w[:, self._ee_body_idx]
        # v_ee = v_link6 + omega x (R * offset_local)
        self._ee_vel_w = link6_lin_vel + torch.linalg.cross(link6_ang_vel, offset_world, dim=-1)

        self._distance = torch.linalg.norm(self._ee_pos_w - self.target_pos, dim=-1)

    def _get_observations(self) -> dict:
        self._update_ee_state()

        joint_pos = self.robot.data.joint_pos[:, self._all_dof_idx]
        joint_vel = self.robot.data.joint_vel[:, self._all_dof_idx]
        relative_pos = self.target_pos - self._ee_pos_w  # 3

        obs = torch.cat(
            (
                relative_pos,
                joint_pos,
                joint_vel,
                self.prev_actions,
                self._ee_vel_w,
            ),
            dim=-1,
        )
        return {"policy": obs}

    # ------------------------------------------------------------------ #
    # Reward (Route B: dense distance + tier amplifiers)                 #
    # ------------------------------------------------------------------ #

    def _get_rewards(self) -> torch.Tensor:
        c = self.cfg
        dist = self._distance

        # 1. step penalty (mild push for speed)
        rew = torch.full_like(dist, -c.rew_step_penalty)

        # 2. dense distance reward: ( 1 / (1 + (beta*d)^2) )^2  in (0, 1]
        #    multiplied x2 when within `close_threshold` and an extra x2
        #    when within `success_threshold` (so a held reach is ~4.0/step).
        inner = 1.0 / (1.0 + (c.rew_dist_beta * dist) ** 2)
        dist_reward = c.rew_dist_gain * inner * inner
        close_mask = dist < c.close_threshold
        succ_mask = dist < c.success_threshold
        dist_reward = torch.where(close_mask, dist_reward * c.rew_close_multiplier, dist_reward)
        dist_reward = torch.where(succ_mask, dist_reward * c.rew_success_multiplier, dist_reward)
        rew = rew + dist_reward

        # 3. direction reward: cos^2 of (ee_vel, to_target)
        to_target = self.target_pos - self._ee_pos_w
        to_target_norm = torch.linalg.norm(to_target, dim=-1, keepdim=True).clamp_min(1e-6)
        to_target_unit = to_target / to_target_norm
        vel_norm = torch.linalg.norm(self._ee_vel_w, dim=-1, keepdim=True).clamp_min(1e-6)
        vel_cos = (to_target_unit * (self._ee_vel_w / vel_norm)).sum(dim=-1)
        rew = rew + c.rew_direction_gain * torch.clamp(vel_cos, min=0.0) ** 2

        # 3b. gripper alignment: signed cos^2 between Link6's local +z axis
        # (the direction the gripper points) and the unit vector from the EE
        # to the target. The signed form is crucial -- pointing AWAY from
        # the ball is actively penalised, not just "zero reward". Without
        # this, the policy preferred "keep wrist locked at default (which
        # happens to point away) and pay no reward" over "spend wrist budget
        # to align". Now it must align, and the cheapest way to do that is
        # using just enough wrist movement to flip cos positive.
        link6_quat_w = self.robot.data.body_quat_w[:, self._ee_body_idx]
        gripper_forward_w = quat_apply(link6_quat_w, self._gripper_forward_local)
        align_cos = (gripper_forward_w * to_target_unit).sum(dim=-1)
        rew = rew + c.rew_gripper_align_gain * align_cos * align_cos.abs()

        # 4. ee speed penalty (cap on linear hand speed)
        ee_speed = torch.linalg.norm(self._ee_vel_w, dim=-1)
        rew = rew - torch.where(ee_speed > c.rew_speed_threshold,
                                torch.full_like(ee_speed, c.rew_speed_penalty),
                                torch.zeros_like(ee_speed))

        # 5. joint velocity change penalty (suppress oscillation)
        cur_joint_vel = self.robot.data.joint_vel[:, self._all_dof_idx]
        joint_vel_change = (cur_joint_vel - self.prev_joint_vel).abs().sum(dim=-1)
        rew = rew - c.rew_joint_vel_change * joint_vel_change
        self.prev_joint_vel = cur_joint_vel

        # 6. wrist suppression: action + position + velocity penalties on
        # Joint4/5/6. Reach only needs J1-3, so each of these terms prefers
        # the policy to leave the wrist at neutral. J4-6 remain physically
        # controllable for Phase 2 grasp work.
        wrist_action_abs = self.actions[:, 3:6].abs().sum(dim=-1)
        rew = rew - c.rew_wrist_action * wrist_action_abs

        # arm joint indices: [J1,J2,J3,J4,J5,J6]; slice 3:6 = J4/J5/J6
        wrist_q = self.robot.data.joint_pos[:, self._arm_dof_idx[3:6]]
        wrist_qvel = self.robot.data.joint_vel[:, self._arm_dof_idx[3:6]]
        wrist_pos_pen = wrist_q.abs().sum(dim=-1)
        wrist_vel_pen = wrist_qvel.abs().sum(dim=-1)
        rew = rew - c.rew_wrist_pos * wrist_pos_pen - c.rew_wrist_vel * wrist_vel_pen

        # 6b. gripper symmetry: left + right == 0 means mirrored motion.
        # Replaces the MuJoCo <equality> coupling the URDF -> USD pipeline
        # dropped. For Phase 2 (grasp), this keeps both fingers in lockstep
        # by reward instead of by a hard constraint.
        gripper_q = self.robot.data.joint_pos[:, self._gripper_dof_idx]
        gripper_asym = (gripper_q[:, 0] + gripper_q[:, 1]).abs()
        rew = rew - c.rew_gripper_asymmetry * gripper_asym

        # 7. per-cm one-shot bonuses inside the close region (4/3/2/1 cm).
        # Each tier fires exactly once per episode the first time the EE
        # crosses below the threshold, so backing away doesn't refund the
        # bonus -- only forward progress counts.
        crossed = (dist.unsqueeze(-1) < self._close_thresholds) & (~self.close_phase_given)
        rew = rew + (crossed.float() * self._close_rewards).sum(dim=-1)
        self.close_phase_given = self.close_phase_given | crossed

        # 8. settle bonus: per-step bonus only while inside the success
        # threshold AND moving slowly. This is what teaches the policy to
        # stop, instead of orbiting the target to keep banking dist_reward.
        settled = succ_mask & (ee_speed < c.rew_settle_speed_threshold)
        rew = rew + c.rew_settle_bonus * settled.float()

        # logging (RSL-RL picks this up automatically)
        self.extras["log"] = {
            "rew/distance_mean": dist.mean(),
            "rew/success_rate": succ_mask.float().mean(),
            "rew/settled_rate": settled.float().mean(),
            "rew/align_cos_mean": align_cos.mean(),
            "diag/wrist_pos_l1": wrist_pos_pen.mean(),    # avg |J4|+|J5|+|J6| in rad
            "diag/wrist_vel_l1": wrist_vel_pen.mean(),    # avg |qvel J4-6| in rad/s
            "diag/gripper_asymmetry": gripper_asym.mean(),
            "rew/total_mean": rew.mean(),
        }
        # Save previous action *after* it has been consumed by obs and reward.
        self.prev_actions = self.actions.clone()
        return rew

    # ------------------------------------------------------------------ #
    # Termination                                                        #
    # ------------------------------------------------------------------ #

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # No early success termination: terminating on reach made hovering
        # near the target more rewarding than actually arriving (lost future
        # dist_reward outweighed the one-shot bonus). The full episode runs
        # so the policy is incentivised to settle and hold.
        terminated = torch.zeros_like(time_out)
        return terminated, time_out

    # ------------------------------------------------------------------ #
    # Reset                                                              #
    # ------------------------------------------------------------------ #

    def _sample_targets(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Sample target positions in a spherical shell around each env origin.

        Replaces the original FK-based sampling: IsaacLab does not refresh
        `body_pos_w` after `write_joint_state_to_sim` without a physics step,
        so probing FK inside `_reset_idx` would read stale data and collapse
        every target to the default end-effector position.
        """
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

        # 1. reset robot to default pose
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # 2. sample targets in a workspace shell (does not touch sim state).
        targets_world = self._sample_targets(env_ids)
        self.target_pos[env_ids] = targets_world
        self.target_pos_local[env_ids] = targets_world - self.scene.env_origins[env_ids]

        # 3. reset per-env bookkeeping
        self.prev_actions[env_ids] = 0.0
        self.prev_joint_vel[env_ids] = 0.0
        self.close_phase_given[env_ids] = False

        # 4. move target visualization markers to the new world positions.
        # `visualize` expects all-env translations even on partial reset.
        self.target_marker.visualize(translations=self.target_pos)
