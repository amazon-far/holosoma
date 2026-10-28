"""Default inference configurations for holosoma_inference."""

from dataclasses import replace
from typing import Annotated

import tyro

from holosoma_inference.config.config_types.contextual import ContextualConfig
from holosoma_inference.config.config_types.inference import InferenceConfig
from holosoma_inference.config.config_types.observation import ObservationConfig
from holosoma_inference.config.config_values import observation, robot, task
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

# Depth-conditioned locomotion for students distilled on the RealSense D435i head rig (27deg-down
# mount, depth clipped to [0.3, 3.0] m and resized to 58x87). The model is one contextual ONNX
# fusing the depth backbone with the student (holosoma_cpp/scripts/fuse_depth_student_onnx.py).
# Its depth comes from a producer that already applies that preprocessing, e.g. holosoma's
# `depth-shm-d435i` sim plugin: run with `--depth-shm depth_img_shm`. The students have no climb
# head. No safety secondary: the depth source attaches to a single contextual runner.
g1_29dof_depth_d435i = InferenceConfig(
    robot=robot.g1_29dof,
    observation=ObservationConfig({"actor_obs": []}, {}, {}, {}),
    task=replace(task.locomotion, use_phase=False, policy_action_scale=1.0),
    contextual=ContextualConfig(require_climb_head=False),
)

# Register core presets. Keys use hyphen-case naming convention for CLI compatibility.
INFERENCE_REGISTRY.add("g1-29dof-loco", g1_29dof_loco)
INFERENCE_REGISTRY.add("t1-29dof-loco", t1_29dof_loco)
INFERENCE_REGISTRY.add("g1-29dof-wbt", g1_29dof_wbt)
INFERENCE_REGISTRY.add("g1-29dof-depth-d435i", g1_29dof_depth_d435i)


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
