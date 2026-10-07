"""Default inference configurations for holosoma_inference."""

from dataclasses import replace

import tyro
from typing_extensions import Annotated

from holosoma_inference.config.config_types.inference import InferenceConfig
from holosoma_inference.config.config_values import camera, observation, robot, task
from holosoma_inference.utils.config_registry import (
    ConfigRegistry,
    deprecated_defaults_alias,
    deprecated_get_defaults,
)

INFERENCE_REGISTRY = ConfigRegistry(InferenceConfig, group="holosoma.config.inference")

# Shared safety secondary for all G1 configs — FastSAC locomotion.
# Each config references the same object; users can override any field
# with --secondary.task.model-path etc., or disable with --secondary none.
_g1_safety_secondary = InferenceConfig(
    robot=robot.g1_29dof,
    observation=observation.loco_g1_29dof,
    task=task.safety_locomotion_g1,
)

g1_29dof_loco = InferenceConfig(
    robot=robot.g1_29dof,
    observation=observation.loco_g1_29dof,
    task=replace(task.locomotion, model_path=task.safety_locomotion_g1.model_path),
    secondary=_g1_safety_secondary,
)

t1_29dof_loco = InferenceConfig(
    robot=robot.t1_29dof,
    observation=observation.loco_t1_29dof,
    task=task.locomotion,
)

# fmt: off
_g1_29dof_wbt_robot = replace(
    robot.g1_29dof,
    stiff_startup_pos=(
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # left leg
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # right leg
        0.0, 0.0, 0.0,                          # waist
        0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,      # left arm
        0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,     # right arm
    ),
    stiff_startup_kp=(
        350.0, 200.0, 200.0, 300.0, 300.0, 150.0,
        350.0, 200.0, 200.0, 300.0, 300.0, 150.0,
        200.0, 200.0, 200.0,
        40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0,
        40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0,
    ),
    stiff_startup_kd=(
        5.0, 5.0, 5.0, 10.0, 5.0, 5.0,
        5.0, 5.0, 5.0, 10.0, 5.0, 5.0,
        5.0, 5.0, 5.0,
        3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
        3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
    ),
)

g1_29dof_wbt = InferenceConfig(
    robot=_g1_29dof_wbt_robot,
# fmt: on
    observation=observation.wbt,
    task=task.wbt,
    secondary=_g1_safety_secondary,
)

# Depth distillation: vision-based locomotion over stairs / rough terrain.
# No safety secondary — the depth backbone and student form one composite policy,
# and its two model paths would collide with the multi-policy switching semantics.
g1_wbt_distillation = InferenceConfig(
    robot=robot.g1_29dof_wbt_distillation,
    observation=observation.wbt_distillation_g1,
    task=task.wbt_distillation,
    camera=camera.single_zed2i_depth,
)

# Same policy, but describing the RealSense D435i rig the D435i checkpoints were trained with
# (27deg-down torso mount, 106x60, clip [0.3, 3.0]) instead of the ZED 2i's. The depth tensor is
# 58x87 either way, so the plain preset also runs — just with the wrong extrinsics and far clip.
# Pair with the sim's `sensor.<key>:g1-d435i-front-depth` + `plugin.<key>:depth-shm-d435i`.
g1_wbt_distillation_d435i = replace(g1_wbt_distillation, camera=camera.single_d435i_depth)

# Register core presets. Keys use hyphen-case naming convention for CLI compatibility.
INFERENCE_REGISTRY.add("g1-29dof-loco", g1_29dof_loco)
INFERENCE_REGISTRY.add("t1-29dof-loco", t1_29dof_loco)
INFERENCE_REGISTRY.add("g1-29dof-wbt", g1_29dof_wbt)
INFERENCE_REGISTRY.add("g1-wbt-distillation", g1_wbt_distillation)
INFERENCE_REGISTRY.add("g1-wbt-distillation-d435i", g1_wbt_distillation_d435i)


def get_annotated_inference_config() -> type:
    """Return the ``inference:`` subcommand type."""
    return Annotated[
        InferenceConfig,
        tyro.conf.arg(
            constructor=tyro.extras.subcommand_type_from_defaults(
                {f"inference:{k}": v for k, v in INFERENCE_REGISTRY.items()}
            )
        ),
    ]


__getattr__ = deprecated_defaults_alias(__name__, INFERENCE_REGISTRY)
get_defaults = deprecated_get_defaults(__name__, INFERENCE_REGISTRY)
