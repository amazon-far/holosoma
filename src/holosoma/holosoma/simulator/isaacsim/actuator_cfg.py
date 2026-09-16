from __future__ import annotations

from typing import Any

from holosoma.config_types.robot import RobotConfig
from holosoma.managers.utils import resolve_callable


def joint_drive_target_type(robot_config: RobotConfig) -> str:
    """Select the URDF drive target required by the configured actuator path."""
    if robot_config.control.control_mode == "implicit_position_target":
        return "position"
    return "none"


def _joint_pd_gains(robot_config: RobotConfig, joint_name: str) -> tuple[float, float]:
    joint_key = joint_name.replace("_joint", "")
    matched_keys = [key for key in robot_config.control.stiffness if key in joint_key]
    if len(matched_keys) != 1:
        raise ValueError(f"Expected exactly one PD gain match for joint '{joint_name}', got {len(matched_keys)}")

    gain_key = matched_keys[0]
    if gain_key not in robot_config.control.damping:
        raise ValueError(f"Missing damping for PD gain key '{gain_key}'")
    return robot_config.control.stiffness[gain_key], robot_config.control.damping[gain_key]


def build_actuator_configs(
    robot_config: RobotConfig,
    *,
    ideal_pd_actuator_cls: type[Any] | None = None,
    implicit_actuator_cls: type[Any] | None = None,
) -> dict[str, Any]:
    """Build the selected IsaacLab actuator backend from one robot contract."""
    if ideal_pd_actuator_cls is None or implicit_actuator_cls is None:
        from isaaclab.actuators import IdealPDActuatorCfg as DefaultIdealPDActuatorCfg
        from isaaclab.actuators import ImplicitActuatorCfg as DefaultImplicitActuatorCfg

        if ideal_pd_actuator_cls is None:
            ideal_pd_actuator_cls = DefaultIdealPDActuatorCfg
        if implicit_actuator_cls is None:
            implicit_actuator_cls = DefaultImplicitActuatorCfg

    control = robot_config.control
    if control.control_mode == "implicit_position_target" and control.implicit_actuator_builder is not None:
        return resolve_callable(control.implicit_actuator_builder, context="implicit actuator builder")()

    actuators: dict[str, Any] = {}
    for index, joint_name in enumerate(robot_config.dof_names):
        common = {
            "joint_names_expr": [joint_name],
            "armature": robot_config.dof_armature_list[index],
            "friction": robot_config.dof_joint_friction_list[index],
        }
        if control.control_mode == "implicit_position_target":
            stiffness, damping = _joint_pd_gains(robot_config, joint_name)
            actuators[joint_name] = implicit_actuator_cls(
                effort_limit_sim=robot_config.dof_effort_limit_list[index],
                velocity_limit_sim=robot_config.dof_vel_limit_list[index],
                stiffness=stiffness,
                damping=damping,
                **common,
            )
        else:
            actuators[joint_name] = ideal_pd_actuator_cls(
                effort_limit=robot_config.dof_effort_limit_list[index],
                velocity_limit=robot_config.dof_vel_limit_list[index],
                stiffness=0,
                damping=0,
                **common,
            )
    return actuators
