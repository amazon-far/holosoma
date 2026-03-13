"""Default task configurations for holosoma_inference."""

from __future__ import annotations

from pathlib import Path

from holosoma_inference.config.config_types.task import TaskConfig

_MODELS_DIR = Path(__file__).parent.parent.parent / "models"

# Locomotion task
locomotion = TaskConfig(
    model_path="",  # Must be provided by user
    rl_rate=50,
    policy_action_scale=0.25,
    use_phase=True,
    gait_period=1.0,
    desired_base_height=0.75,
    residual_upper_body_action=False,
    domain_id=0,
    interface="lo",
    use_joystick=False,
    joystick_type="xbox",
    joystick_device=0,
    use_ros=False,
    wandb_download_dir="/tmp",
)

# Whole-body tracking task
wbt = TaskConfig(
    model_path="",  # Must be provided by user
    rl_rate=50,
    policy_action_scale=1.0,
    use_phase=False,
    gait_period=1.0,
    desired_base_height=0.75,
    residual_upper_body_action=False,
    domain_id=0,
    interface="lo",
    use_joystick=False,
    joystick_type="xbox",
    joystick_device=0,
    use_ros=False,
    wandb_download_dir="/tmp",
)

# Safety locomotion (FastSAC) — used as default secondary for dual-mode
safety_locomotion_g1 = TaskConfig(
    model_path=str(_MODELS_DIR / "loco" / "g1_29dof" / "fastsac_g1_29dof.onnx"),
    rl_rate=50,
    policy_action_scale=0.25,
    use_phase=True,
    gait_period=1.0,
    desired_base_height=0.75,
    residual_upper_body_action=False,
    domain_id=0,
    interface="lo",
    use_joystick=False,
    joystick_type="xbox",
    joystick_device=0,
    use_ros=False,
    wandb_download_dir="/tmp",
)

DEFAULTS = {
    "locomotion": locomotion,
    "wbt": wbt,
    "safety_locomotion_g1": safety_locomotion_g1,
}
