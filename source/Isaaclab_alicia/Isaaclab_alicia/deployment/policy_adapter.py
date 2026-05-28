from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .sim2real_config import DeploymentConfig


def _as_1d_array(name: str, value: np.ndarray | list[float], size: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size != size:
        raise ValueError(f"{name} must have size {size}, got {arr.size}")
    return arr


def build_observation(
    robot_state: Mapping[str, np.ndarray | list[float]],
    target_pos: np.ndarray | list[float],
    prev_action: np.ndarray | list[float],
    cfg: DeploymentConfig,
) -> np.ndarray:
    joint_dim = cfg.observation.joint_dim
    action_dim = cfg.observation.action_dim

    joint_pos = _as_1d_array("joint_pos", robot_state["joint_pos"], joint_dim)
    joint_vel = _as_1d_array("joint_vel", robot_state["joint_vel"], joint_dim)
    ee_pos = _as_1d_array("ee_pos", robot_state["ee_pos"], 3)
    ee_vel = _as_1d_array("ee_vel", robot_state["ee_vel"], 3)
    target = _as_1d_array("target_pos", target_pos, 3)
    prev = _as_1d_array("prev_action", prev_action, action_dim)

    relative_pos = target - ee_pos
    obs = np.concatenate([relative_pos, joint_pos, joint_vel, prev, ee_vel], axis=0).astype(np.float32, copy=False)
    if obs.size != cfg.observation.observation_dim:
        raise ValueError(f"Observation size mismatch: expected {cfg.observation.observation_dim}, got {obs.size}")
    return obs


def action_to_position_cmd(
    action: np.ndarray | list[float],
    q_now: np.ndarray | list[float],
    cfg: DeploymentConfig,
) -> np.ndarray:
    q_now_arr = np.asarray(q_now, dtype=np.float32).reshape(-1)
    act = _as_1d_array("action", action, cfg.observation.action_dim)
    act = np.clip(act, -1.0, 1.0)

    q_cmd = q_now_arr.copy()

    arm_delta = np.clip(
        act[:6] * cfg.action.action_gain_arm,
        -cfg.action.dq_limit_arm,
        cfg.action.dq_limit_arm,
    )
    q_cmd[:6] = q_now_arr[:6] + arm_delta

    if q_now_arr.size >= 8:
        if cfg.action.enable_gripper:
            grip_delta = float(np.clip(act[6] * cfg.action.action_gain_gripper, -cfg.action.dq_limit_gripper, cfg.action.dq_limit_gripper))
            q_cmd[6] = q_now_arr[6] + grip_delta
            q_cmd[7] = q_now_arr[7] - grip_delta
        else:
            q_cmd[6:8] = q_now_arr[6:8]

    return q_cmd


def safety_guard(
    q_cmd: np.ndarray | list[float],
    q_now: np.ndarray | list[float],
    cfg: DeploymentConfig,
    joint_limits: np.ndarray | None = None,
    *,
    watchdog_timed_out: bool = False,
    emergency_stop: bool = False,
) -> np.ndarray:
    q_now_arr = np.asarray(q_now, dtype=np.float32).reshape(-1)
    q_cmd_arr = np.asarray(q_cmd, dtype=np.float32).reshape(-1)
    if q_cmd_arr.size != q_now_arr.size:
        raise ValueError("q_cmd and q_now must have the same size")

    if emergency_stop or (watchdog_timed_out and cfg.safety.hold_on_timeout):
        return q_now_arr.copy()

    if cfg.safety.finite_check and not np.isfinite(q_cmd_arr).all():
        return q_now_arr.copy()

    delta = q_cmd_arr - q_now_arr
    arm_dim = min(6, delta.size)
    delta[:arm_dim] = np.clip(delta[:arm_dim], -cfg.action.dq_limit_arm, cfg.action.dq_limit_arm)
    if delta.size > 6:
        delta[6:] = np.clip(delta[6:], -cfg.action.dq_limit_gripper, cfg.action.dq_limit_gripper)
    q_safe = q_now_arr + delta

    if joint_limits is not None:
        limits = np.asarray(joint_limits, dtype=np.float32)
        if limits.shape != (q_safe.size, 2):
            raise ValueError(f"joint_limits must have shape ({q_safe.size}, 2), got {limits.shape}")
        margin = max(float(cfg.safety.joint_limit_margin), 0.0)
        low = limits[:, 0] + margin
        high = limits[:, 1] - margin
        q_safe = np.clip(q_safe, low, high)

    if cfg.safety.finite_check and not np.isfinite(q_safe).all():
        return q_now_arr.copy()
    return q_safe.astype(np.float32, copy=False)
