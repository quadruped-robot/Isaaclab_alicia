from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - depends on runtime env
    yaml = None


@dataclass
class RuntimeConfig:
    control_hz: float = 60.0
    obs_latency_frames: int = 1
    act_latency_frames: int = 1


@dataclass
class ObservationConfig:
    observation_dim: int = 30
    action_dim: int = 8
    joint_dim: int = 8


@dataclass
class ActionMappingConfig:
    action_gain_arm: float = 0.035
    action_gain_gripper: float = 0.015
    dq_limit_arm: float = 0.08
    dq_limit_gripper: float = 0.04
    enable_gripper: bool = False


@dataclass
class SafetyConfig:
    joint_limit_margin: float = 0.03
    watchdog_timeout_s: float = 0.2
    hold_on_timeout: bool = True
    finite_check: bool = True


@dataclass
class DeploymentConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    action: ActionMappingConfig = field(default_factory=ActionMappingConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)


def _update_dataclass(instance: Any, payload: dict[str, Any]) -> None:
    for key, value in payload.items():
        if not hasattr(instance, key):
            continue
        current_value = getattr(instance, key)
        if is_dataclass(current_value) and isinstance(value, dict):
            _update_dataclass(current_value, value)
        else:
            setattr(instance, key, value)


def _load_payload(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    if yaml is None:
        raise RuntimeError("PyYAML is required to load non-JSON deployment config files.")
    data = yaml.safe_load(text)
    return {} if data is None else data


def load_deployment_config(path: str | Path) -> DeploymentConfig:
    config_path = Path(path)
    cfg = DeploymentConfig()
    if not config_path.exists():
        return cfg
    payload = _load_payload(config_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Deployment config must be a dict-like object, got {type(payload)}")
    _update_dataclass(cfg, payload)
    return cfg


def save_deployment_config(cfg: DeploymentConfig, path: str | Path) -> None:
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(cfg)
    if config_path.suffix.lower() == ".json":
        config_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return
    if yaml is None:
        raise RuntimeError("PyYAML is required to save non-JSON deployment config files.")
    config_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
