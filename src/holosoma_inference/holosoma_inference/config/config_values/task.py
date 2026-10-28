"""Default task configurations for holosoma_inference."""

from __future__ import annotations

from pathlib import Path

from holosoma_inference.config.config_types.task import TaskConfig
from holosoma_inference.utils.config_registry import (
    ConfigRegistry,
    deprecated_defaults_alias,
    deprecated_get_defaults,
)

TASK_REGISTRY = ConfigRegistry(TaskConfig, group="holosoma.config.task")

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
    velocity_input="keyboard",
    state_input="keyboard",
    joystick_type="xbox",
    joystick_device=0,
)

# Whole-body tracking task
wbt = TaskConfig(
    model_path="",  # Must be provided by user
    rl_rate=50,
    policy_action_scale=1.0,
    action_scales_by_effort_limit_over_p_gain=True,
    use_phase=False,
    gait_period=1.0,
    desired_base_height=0.75,
    residual_upper_body_action=False,
    domain_id=0,
    interface="lo",
    velocity_input="keyboard",
    state_input="keyboard",
    joystick_type="xbox",
    joystick_device=0,
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
    velocity_input="keyboard",
    state_input="keyboard",
    joystick_type="xbox",
    joystick_device=0,
)

TASK_REGISTRY.add("locomotion", locomotion)
TASK_REGISTRY.add("wbt", wbt)
TASK_REGISTRY.add("safety_locomotion_g1", safety_locomotion_g1)

__getattr__ = deprecated_defaults_alias(__name__, TASK_REGISTRY)
get_defaults = deprecated_get_defaults(__name__, TASK_REGISTRY)
