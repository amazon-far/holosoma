"""Default inference configurations for holosoma_inference."""

from __future__ import annotations

from dataclasses import replace

import tyro
from typing_extensions import Annotated

from holosoma_inference.config.config_types.inference import InferenceConfig
from holosoma_inference.config.config_values import observation, robot, task

# G1 Locomotion
g1_29dof_loco = InferenceConfig(
    robot=robot.g1_29dof,
    observation=observation.loco_g1_29dof,
    task=task.locomotion,
)

# T1 Locomotion
t1_29dof_loco = InferenceConfig(
    robot=robot.t1_29dof,
    observation=observation.loco_t1_29dof,
    task=task.locomotion,
)

# G1 Whole-Body Tracking
g1_29dof_wbt = InferenceConfig(
    robot=replace(
        robot.g1_29dof,
        stiff_startup_pos=(
            -0.312,
            0.0,
            0.0,
            0.669,
            -0.363,
            0.0,  # left leg
            -0.312,
            0.0,
            0.0,
            0.669,
            -0.363,
            0.0,  # right leg
            0.0,
            0.0,
            0.0,  # waist
            0.2,
            0.2,
            0.0,
            0.6,
            0.0,
            0.0,
            0.0,  # left arm
            0.2,
            -0.2,
            0.0,
            0.6,
            0.0,
            0.0,
            0.0,  # right arm
        ),
        stiff_startup_kp=(
            350.0,
            200.0,
            200.0,
            300.0,
            300.0,
            150.0,  # left leg
            350.0,
            200.0,
            200.0,
            300.0,
            300.0,
            150.0,  # right leg
            200.0,
            200.0,
            200.0,  # waist
            40.0,
            40.0,
            40.0,
            40.0,
            40.0,
            40.0,
            40.0,  # left arm
            40.0,
            40.0,
            40.0,
            40.0,
            40.0,
            40.0,
            40.0,  # right arm
        ),
        stiff_startup_kd=(
            5.0,
            5.0,
            5.0,
            10.0,
            5.0,
            5.0,  # left leg
            5.0,
            5.0,
            5.0,
            10.0,
            5.0,
            5.0,  # right leg
            5.0,
            5.0,
            5.0,  # waist
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,  # left arm
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,  # right arm
        ),
    ),
    observation=observation.wbt,
    task=task.wbt,
)

# T1 Whole-Body Tracking (23DOF)
t1_23dof_wbt = InferenceConfig(
    robot=replace(
        robot.t1_23dof,
        stiff_startup_pos=(
            0.0,
            0.0,  # head (yaw, pitch)
            0.2,
            -1.35,
            0.0,
            -0.5,  # left arm
            0.2,
            1.35,
            0.0,
            0.5,  # right arm
            0.0,  # waist
            -0.2,
            0.0,
            0.0,
            0.4,
            -0.25,
            0.0,  # left leg
            -0.2,
            0.0,
            0.0,
            0.4,
            -0.25,
            0.0,  # right leg
        ),
        stiff_startup_kp=(
            20,
            20,  # head
            20,
            20,
            20,
            20,  # left arm
            20,
            20,
            20,
            20,  # right arm
            200,  # waist
            200,
            200,
            200,
            200,
            50,
            50,  # left leg
            200,
            200,
            200,
            200,
            50,
            50,  # right leg
        ),
        stiff_startup_kd=(
            0.2,
            0.2,  # head
            0.5,
            0.5,
            0.5,
            0.5,  # left arm
            0.5,
            0.5,
            0.5,
            0.5,  # right arm
            5,  # waist
            5,
            5,
            5,
            5,
            3,
            3,  # left leg
            5,
            5,
            5,
            5,
            3,
            3,  # right leg
        ),
    ),
    observation=observation.wbt_23dof,
    task=task.wbt,
)

DEFAULTS = {
    "g1-29dof-loco": g1_29dof_loco,
    "t1-29dof-loco": t1_29dof_loco,
    "g1-29dof-wbt": g1_29dof_wbt,
    "t1-23dof-wbt": t1_23dof_wbt,
}

# Annotated version for Tyro CLI with subcommands
AnnotatedInferenceConfig = Annotated[
    InferenceConfig,
    tyro.conf.arg(
        constructor=tyro.extras.subcommand_type_from_defaults({f"inference:{k}": v for k, v in DEFAULTS.items()})
    ),
]
