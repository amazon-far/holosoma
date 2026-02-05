"""Task configuration types for holosoma_inference."""

from __future__ import annotations

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class TaskConfig:
    """Task execution configuration for policy inference."""

    model_path: str | list[str]
    """Path to ONNX model(s). Supports local paths and wandb:// URIs. Required field."""

    rl_rate: float = 50
    """Policy inference rate in Hz."""

    policy_action_scale: float = 0.25
    """Scaling factor applied to policy actions."""

    use_phase: bool = True
    """Whether to use gait phase observations."""

    gait_period: float = 1.0
    """Gait cycle period in seconds."""

    domain_id: int = 0
    """DDS domain ID for communication."""

    interface: str = "lo"
    """Network interface name."""

    use_joystick: bool = False
    """Enable joystick control input."""

    joystick_type: str = "xbox"
    """Joystick type."""

    joystick_device: int = 0
    """Joystick device index."""

    use_sim_time: bool = False
    """Use synchronized simulation time for WBT policies."""

    motion_file_path: str | None = None
    """Path to motion file (npz) for WBT policies with future motion encoder."""

    save_debug: bool = False
    """Save debug log for future_motion_targets (sim-to-sim) to compare with Isaac eval."""

    wandb_download_dir: str = "/tmp"
    """Directory for downloading W&B checkpoints."""

    # Deprecation candidates:
    desired_base_height: float = 0.75
    """Target base height in meters."""

    residual_upper_body_action: bool = False
    """Whether to use residual control for upper body."""

    use_ros: bool = False
    """Use ROS2 for rate limiting."""
