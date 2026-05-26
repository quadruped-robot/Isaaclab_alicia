# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Alicia_duo 6+2 DoF reach environment.
# Migrated from the MuJoCo + SB3-TD3 project under
# `mechanical arm/zero-robotic-arm/5. Deep_LR/`.
# First phase: reach-only (gripper is an action dim but not part of reward).

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

# Absolute path to the converted USD asset.
_ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "assets", "alicia_duo"))
_USD_PATH = os.path.join(_ASSETS_DIR, "usd", "alicia_duo.usd")


ALICIA_DUO_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=_USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=12,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "Joint1": 0.0,
            "Joint2": 0.0,
            "Joint3": 0.0,
            "Joint4": 0.0,
            "Joint5": 0.0,
            "Joint6": 0.0,
            "left_finger": 0.0,
            "right_finger": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["Joint[1-6]"],
            effort_limit_sim=5.0,
            velocity_limit_sim=12.0,
            stiffness=0.0,
            damping=0.5,
        ),
        # Gripper: small return-to-centre spring (stiffness=20) so the
        # fingers don't drift asymmetrically under arm-induced vibrations
        # when Phase 1 leaves torque=0. The max spring force at full travel
        # is 20 * 0.05 = 1 N*m -- easily overpowered by the 5 N*m effort
        # limit, so Phase 2 grasp control is unaffected.
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["left_finger", "right_finger"],
            effort_limit_sim=5.0,
            velocity_limit_sim=12.0,
            stiffness=20.0,
            damping=1.0,
        ),
    },
)


@configclass
class IsaaclabAliciaEnvCfg(DirectRLEnvCfg):
    # ---- core ----
    decimation = 2
    episode_length_s = 4.0  # 4s episode @ 60 Hz control -> 240 steps

    # action: 8 torques (arm 6 + gripper 2), policy outputs in [-1, 1]
    action_space = 8
    # obs = relative_pos(3) + joint_pos(8) + joint_vel(8) + prev_action(8) + ee_vel(3) = 30
    observation_space = 30
    state_space = 0

    # ---- simulation ----
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # ---- robot + scene ----
    robot_cfg: ArticulationCfg = ALICIA_DUO_CFG
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=2.0, replicate_physics=True
    )

    # ---- joint names (resolved at env init via find_joints / find_bodies) ----
    arm_joint_names = ["Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Joint6"]
    gripper_joint_names = ["left_finger", "right_finger"]
    # `tool0` was a fixed leaf in the URDF and got merged into Link6 by --merge-joints.
    # End-effector pose = Link6 pose composed with this offset.
    #
    # The URDF's Grasp2tool joint placed tool0 at z=0.13118 in Link6 frame,
    # i.e. the OUTER tip of the fingers. We aim slightly inward of that
    # (toward the grasp centre between the two fingers) so a closing
    # gripper actually wraps the ball. Empirically 0.10 hits the sweet
    # spot: closer than the finger tip but not so far in that targets at
    # the edge of the sampled shell become unreachable.
    ee_body_name = "Link6"
    ee_offset_local = (-0.0002, -0.0003, 0.10)

    # ---- action scaling ----
    # Policy outputs in [-1, 1]; multiply to get N*m. effort_limit_sim already
    # caps this in PhysX, so we choose 5.0 to match the joint effort limit.
    action_scale_arm = 5.0
    action_scale_gripper = 5.0

    # ---- target sampling (spherical shell around the base) ----
    # IsaacLab cannot refresh body_pos_w without a physics step, so the original
    # MuJoCo FK-based sampling can't be ported verbatim. Instead we sample
    # uniformly inside a known-reachable shell. Tune ranges if the policy
    # plateaus on harder targets.
    target_r_min = 0.18         # min radius from base [m]
    target_r_max = 0.35         # max radius from base [m] (shrunk a bit so
                                # the new inward ee_offset still keeps the
                                # whole shell reachable)
    target_elev_min = -0.17     # min elevation [rad] (~ -10 deg, slightly below horizon)
    target_elev_max = 0.87      # max elevation [rad] (~ 50 deg, above-horizon)
    target_center_z = 0.20      # workspace center height [m] (~ shoulder)
    target_z_min = 0.05         # final z clamp to keep targets off the ground

    # ---- reward / termination (reach-only, dense PPO-friendly variant) ----
    # Route B: continuous distance shaping + close/success tier amplifiers.
    # Replaces the original TD3-tuned discrete 7-phase ladder which made PPO
    # plateau at ~0.2 m because the gradient flattens between phase steps.
    #
    # Per-step magnitudes when policy is reasonable:
    #   dist_reward in [0.04 .. 1.0]  (×2 below close_threshold, ×4 below success_threshold)
    #   step_penalty -0.001
    #   direction <= 0.1
    #   regularization terms ~ -0.005
    # Episode return for a quick reach ~50 + dist_shaping; baseline drifters ~ +20.

    # Reach reward shape (no terminating success: hovering just outside the
    # success threshold to milk the per-step bonus was the previous failure
    # mode -- terminating on reach made the early-exit bonus smaller than the
    # accumulated hover reward). Episodes only end on timeout now, and a
    # large multiplier inside the success threshold pulls the policy through
    # the last few cm.
    success_threshold = 0.02    # 2 cm reach (used by logging + the inner amplifier)
    close_threshold = 0.05      # 5 cm "close" tier

    # Reward magnitudes deliberately kept modest. PPO normalizes advantages
    # per batch, so absolute scale doesn't matter -- but the *ratio* between
    # rewards and penalties does. Previously dist+settle dominated the return
    # at ~3800 vs ~-20 of regularization, making wrist/asymmetry penalties
    # invisible (0.5% of total) and producing slack behaviour around target.
    # The shrunken main rewards bring that ratio to ~10:1, so the soft
    # constraints actually matter once the arm is on target.
    rew_step_penalty = 0.001          # subtracted each step
    rew_dist_gain = 0.5               # peak coefficient of dist_reward
    rew_dist_beta = 2.0               # controls width of 1/(1+(beta*d)^2) curve
    rew_close_multiplier = 1.5        # smaller bonus for "close" -- discourages hovering
    rew_success_multiplier = 8.0      # large bonus inside success threshold

    rew_direction_gain = 0.05         # cos^2 of (ee_vel, to_target)
    rew_speed_threshold = 1.0         # ee_speed beyond this -> penalty
    rew_speed_penalty = 0.01

    # Gripper-orientation reward: signed cos^2 between Link6's local +z
    # (the gripper-forward axis) and the unit vector from the EE to the
    # target. Pointing away from the target is penalised (not just "zero
    # reward"), so the policy is forced to align even at the cost of using
    # wrist DoF -- which is exactly the "use wrist only when necessary"
    # behaviour the user asked for.
    rew_gripper_align_gain = 0.5

    # Per-centimetre one-shot bonuses inside `close_threshold`. The dense
    # distance reward is smooth but its gradient is gentle near zero; these
    # discrete tiers give the policy a sharp incentive to keep closing the
    # last few cm rather than orbit at ~4 cm.
    rew_close_phase_thresholds = (0.04, 0.03, 0.02, 0.01)
    rew_close_phase_rewards = (1.0, 2.0, 3.0, 5.0)

    rew_joint_vel_change = 0.001      # subtract gain * sum|delta qvel|

    # Wrist suppression -- soft, not hard. Reach only needs J1/2/3, but
    # J4/5/6 must stay available for Phase 2 (grasping in cluttered scenes)
    # AND for aiming the gripper at off-axis targets.
    #
    # With the signed alignment reward, the policy is strongly incentivised
    # to point at the target, so wrist movement happens when needed. These
    # penalties shape *how much* wrist is used: any wrist deflection beyond
    # what is needed for alignment is unrewarded and costs ~`rew_wrist_pos`
    # per radian per step. Together with the alignment reward, the policy's
    # optimum is "use the minimum wrist angle that yields cos~=1".
    rew_wrist_action = 0.05           # |a[3]| + |a[4]| + |a[5]|
    rew_wrist_pos = 0.15              # |q[J4]| + |q[J5]| + |q[J6]|
    rew_wrist_vel = 0.02              # |qvel[J4]| + |qvel[J5]| + |qvel[J6]|

    # Gripper symmetry: substitute for the MuJoCo <equality> coupling the
    # URDF -> USD pipeline dropped. left_finger = -right_finger is the
    # symmetric-close condition; their sum (in radians) is the asymmetry.
    rew_gripper_asymmetry = 5.0       # |q[left_finger] + q[right_finger]|

    # Settle bonus: extra reward when both within success threshold and
    # near-stationary. Scaled down with the rest of the positive rewards so
    # regularisation penalties are not buried by it.
    rew_settle_speed_threshold = 0.05  # m/s; ee considered "settled" below this
    rew_settle_bonus = 4.0             # added per step when settled inside success
