"""Config types for eval callbacks."""

from __future__ import annotations

import dataclasses
from typing import Any

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class RecordingConfig:
    """Settings for trajectory recording during evaluation."""

    enabled: bool = False
    """Whether to enable trajectory recording."""

    output_path: str = "eval_recording.npz"
    """Path to save NPZ recording."""

    env_id: int = 0
    """Environment ID to record."""


@dataclass(frozen=True)
class RecordingCallbackConfig:
    """Instantiation config for EvalRecordingCallback."""

    _target_: str = "holosoma.agents.callbacks.recording.EvalRecordingCallback"
    """Class to instantiate."""

    config: RecordingConfig = RecordingConfig()
    """Recording settings."""


@dataclass(frozen=True)
class PushConfig:
    """Settings for push perturbation during evaluation."""

    enabled: bool = False
    """Enable push perturbations."""

    force_range: tuple[float, float] = (50.0, 200.0)
    """Min and max force magnitude in Newtons."""

    duration_s: tuple[float, float] = (0.1, 0.3)
    """Min and max push duration in seconds."""

    interval_s: tuple[float, float] = (3.0, 8.0)
    """Min and max interval between pushes in seconds."""

    body_names: str = "torso_link,pelvis"
    """Comma-separated body names to push."""

    env_id: int = 0
    """Environment ID to apply pushes."""


@dataclass(frozen=True)
class PushCallbackConfig:
    """Instantiation config for EvalPushCallback."""

    _target_: str = "holosoma.agents.callbacks.push.EvalPushCallback"
    """Class to instantiate."""

    config: PushConfig = PushConfig()
    """Push perturbation settings."""


@dataclass(frozen=True)
class PayloadConfig:
    """Settings for wrist payload simulation during evaluation.

    Applies a constant downward force on wrist/elbow links to simulate
    the robot holding a payload (force = mass * 9.81).
    """

    enabled: bool = False
    """Enable wrist payload forces."""

    mass_kg: float = 1.0
    """Payload mass in kg. Force is split evenly across resolved bodies."""

    body_names: str = "left_wrist_yaw_link,right_wrist_yaw_link"
    """Comma-separated wrist body names."""

    env_id: int = 0
    """Environment ID to apply payload forces."""


@dataclass(frozen=True)
class PayloadCallbackConfig:
    """Instantiation config for EvalPayloadCallback."""

    _target_: str = "holosoma.agents.callbacks.payload.EvalPayloadCallback"
    """Class to instantiate."""

    config: PayloadConfig = PayloadConfig()
    """Payload simulation settings."""


@dataclass(frozen=True)
class EvalCallbacksConfig:
    """Container for all eval callback configs.

    To add a new callback, add a field here with its config type.
    Each field's value is passed to instantiate() if it has a _target_.
    """

    recording: RecordingCallbackConfig = RecordingCallbackConfig()
    """Trajectory recording callback."""

    push: PushCallbackConfig = PushCallbackConfig()
    """Push perturbation callback."""

    payload: PayloadCallbackConfig = PayloadCallbackConfig()
    """Wrist payload simulation callback."""

    def collect_active_callbacks(self) -> dict[str, Any]:
        """Collect callback configs where config.enabled is True."""
        cb_configs: dict[str, Any] = {}
        for f in dataclasses.fields(self):
            cfg = getattr(self, f.name)
            if not hasattr(cfg, "_target_"):
                raise ValueError(f"Callback config '{f.name}' missing _target_ field")
            if not hasattr(cfg.config, "enabled"):
                raise ValueError(f"Callback config '{f.name}' missing config.enabled field")
            if cfg.config.enabled:
                cb_configs[f.name] = cfg
        return cb_configs
