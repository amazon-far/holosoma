"""Default inference configurations for holosoma_inference."""

from dataclasses import replace

import tyro
from typing_extensions import Annotated

from holosoma_inference.compat import entry_points
from holosoma_inference.config.config_types.inference import InferenceConfig
from holosoma_inference.config.config_values import observation, robot, task

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

h1_19dof_loco = InferenceConfig(
    robot=robot.h1_19dof,
    observation=observation.loco_h1_19dof,
    task=replace(
        task.locomotion,
        model_path="/home/pjm/Desktop/holosoma/logs/walk_h1.onnx",
        interface="auto",
        velocity_input="interface",
        state_input="interface",
        auto_walk_on_vel_cmd=True,
    ),
)

_h1_safety_secondary = InferenceConfig(
    robot=robot.h1_19dof,
    observation=observation.loco_h1_19dof,
    task=replace(
        task.locomotion,
        model_path="/home/pjm/Desktop/holosoma/logs/walk_h1.onnx",
        interface="auto",
        velocity_input="interface",
        state_input="interface",
        auto_walk_on_vel_cmd=True,
    ),
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

_h1_19dof_wbt_robot = replace(
    robot.h1_19dof,
    # H1 stable PD-hold stance from dance_h1's final static frame.
    # Joint order: left leg, right leg, torso, left arm, right arm.
    stiff_startup_pos=(
        0.0, 0.0, -0.4, 0.8, -0.4,
        0.0, 0.0, -0.4, 0.8, -0.4,
        0.0,
        0.2, 0.0, 0.0, 0.6,
        0.2, 0.0, 0.0, 0.6,
    ),
    stiff_startup_kp=(
        200.0, 200.0, 200.0, 300.0, 40.0,
        200.0, 200.0, 200.0, 300.0, 40.0,
        300.0,
        100.0, 100.0, 100.0, 100.0,
        100.0, 100.0, 100.0, 100.0,
    ),
    stiff_startup_kd=(
        5.0, 5.0, 5.0, 6.0, 2.0,
        5.0, 5.0, 5.0, 6.0, 2.0,
        6.0,
        2.0, 2.0, 2.0, 2.0,
        2.0, 2.0, 2.0, 2.0,
    ),
)

h1_19dof_wbt = InferenceConfig(
    robot=_h1_19dof_wbt_robot,
    observation=observation.wbt_h1_19dof,
    task=replace(
        task.wbt,
        model_path="/home/pjm/Desktop/holosoma/logs/dance_h1.onnx",
        interface="auto",
        velocity_input="interface",
        state_input="interface",
    ),
    secondary=_h1_safety_secondary,
)

h1_19dof_loco_wbt = InferenceConfig(
    robot=robot.h1_19dof,
    observation=observation.loco_h1_19dof,
    task=replace(
        task.locomotion,
        model_path="/home/pjm/Desktop/holosoma/logs/walk_h1.onnx",
        interface="auto",
        velocity_input="interface",
        state_input="interface",
        auto_walk_on_vel_cmd=True,
    ),
    secondary=InferenceConfig(
        robot=_h1_19dof_wbt_robot,
        observation=observation.wbt_h1_19dof,
        task=replace(
            task.wbt,
            model_path="/home/pjm/Desktop/holosoma/logs/dance_h1.onnx",
            interface="auto",
            velocity_input="interface",
            state_input="interface",
            motion_start_timestep=150,
        ),
    ),
)

_h1_19dof_real_wbt_robot = replace(
    _h1_19dof_wbt_robot,
    message_type=robot.h1_19dof_real_go.message_type,
    num_motors=robot.h1_19dof_real_go.num_motors,
    # WBT policy-space zero point and residual-action base must match
    # dance_h1.onnx training.  FixStand remains separate below as the startup
    # hold pose only.
    default_dof_angles=robot.h1_19dof.default_dof_angles,
    default_motor_angles=robot.h1_19dof.default_motor_angles,
    motor2joint=robot.h1_19dof_real_go.motor2joint,
    joint2motor=robot.h1_19dof_real_go.joint2motor,
    # Runtime PD gains match the values embedded in dance_h1.onnx.
    motor_kp=robot.h1_19dof.motor_kp,
    motor_kd=robot.h1_19dof.motor_kd,
    stiff_startup_pos=robot.h1_19dof_real_go.default_dof_angles,
    stiff_startup_kp=(
        300.0, 300.0, 300.0, 300.0, 60.0,  # left leg
        300.0, 300.0, 300.0, 300.0, 60.0,  # right leg
        300.0,                               # torso
        60.0, 60.0, 60.0, 60.0,             # left arm
        60.0, 60.0, 60.0, 60.0,             # right arm
    ),
    stiff_startup_kd=(
        5.0, 5.0, 5.0, 6.0, 1.5,   # left leg
        5.0, 5.0, 5.0, 6.0, 1.5,   # right leg
        6.0,                         # torso
        1.5, 1.5, 1.5, 1.5,          # left arm
        1.5, 1.5, 1.5, 1.5,          # right arm
    ),
    weak_motor_joint_index=robot.h1_19dof_real_go.weak_motor_joint_index,
)

# Keep the real GO2 transport and motor mapping, but use the exact policy-space
# pose and PD gains from locomotion training.  The angles define both the
# dof_pos observation zero point and the residual-action base.
_h1_19dof_real_loco_robot = replace(
    robot.h1_19dof_real_go,
    default_dof_angles=robot.h1_19dof.default_dof_angles,
    default_motor_angles=robot.h1_19dof.default_motor_angles,
    motor_kp=robot.h1_19dof.motor_kp,
    motor_kd=robot.h1_19dof.motor_kd,
)

h1_19dof_loco_real = InferenceConfig(
    robot=_h1_19dof_real_loco_robot,
    observation=observation.loco_h1_19dof,
    task=replace(
        task.locomotion,
        model_path="/home/pjm/Desktop/holosoma/logs/walk_h1.onnx",
        interface="auto",
        velocity_input="interface",
        state_input="interface",
        auto_walk_on_vel_cmd=True,
    ),
)

h1_19dof_loco_wbt_real = InferenceConfig(
    robot=_h1_19dof_real_loco_robot,
    observation=observation.loco_h1_19dof,
    task=replace(
        task.locomotion,
        model_path="/home/pjm/Desktop/holosoma/logs/walk_h1.onnx",
        interface="auto",
        velocity_input="interface",
        state_input="interface",
        auto_walk_on_vel_cmd=True,
    ),
    secondary=InferenceConfig(
        robot=_h1_19dof_real_wbt_robot,
        observation=observation.wbt_h1_19dof,
        task=replace(
            task.wbt,
            model_path="/home/pjm/Desktop/holosoma/logs/dance_h1.onnx",
            interface="auto",
            velocity_input="interface",
            state_input="interface",
            motion_start_timestep=150,
        ),
    ),
)

# Core defaults - no extension imports at module load time
DEFAULTS = {
    "g1-29dof-loco": g1_29dof_loco,
    "t1-29dof-loco": t1_29dof_loco,
    "h1-19dof-loco": h1_19dof_loco,
    "h1-19dof-loco-wbt": h1_19dof_loco_wbt,
    "h1-19dof-loco-real": h1_19dof_loco_real,
    "h1-19dof-loco-wbt-real": h1_19dof_loco_wbt_real,
    "g1-29dof-wbt": g1_29dof_wbt,
    "h1-19dof-wbt": h1_19dof_wbt,
}

# Track whether extensions have been loaded
_extensions_loaded = False


def _load_extensions() -> None:
    """Lazily load extension configs from entry points.

    This is deferred to avoid circular imports when extensions import
    from holosoma_inference.config at module load time.
    """
    global _extensions_loaded  # noqa: PLW0603
    if _extensions_loaded:
        return
    _extensions_loaded = True
    for ep in entry_points(group="holosoma.config.inference"):
        DEFAULTS[ep.name] = ep.load()


def get_annotated_inference_config() -> type:
    """Build the annotated InferenceConfig type with all discovered configs.

    This function loads extension configs lazily and returns a tyro-compatible
    annotated type for CLI subcommand generation.

    Returns:
        Annotated type suitable for use with tyro.cli()
    """
    _load_extensions()
    return Annotated[
        InferenceConfig,
        tyro.conf.arg(
            constructor=tyro.extras.subcommand_type_from_defaults(
                {f"inference:{k}": v for k, v in DEFAULTS.items()}
            )
        ),
    ]


def get_defaults() -> dict:
    """Get all inference config defaults, including extensions.

    Returns:
        Dictionary mapping config names to InferenceConfig instances.
    """
    _load_extensions()
    return DEFAULTS
