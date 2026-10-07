"""Headless checks for grouped Isaac Lab actuator and contact-sensor configuration."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import sys
import traceback
from pathlib import Path

if sys.path and sys.path[0].endswith("tests/simulators"):
    sys.path.pop(0)


def _synthetic_robot_config():
    from holosoma.config_values.robot import g1_29dof

    joint_names = [f"test_joint_{index:02d}" for index in range(33)]
    return dataclasses.replace(
        g1_29dof,
        dof_names=joint_names,
        dof_pos_lower_limit_list=[float(-2 * (index + 1)) for index in range(33)],
        dof_pos_upper_limit_list=[float(2 * (index + 2)) for index in range(33)],
        dof_effort_limit_list=[float(10 + index) for index in range(33)],
        dof_vel_limit_list=[float(20 + index) for index in range(33)],
        dof_armature_list=[0.01 * index for index in range(33)],
        dof_joint_friction_list=[0.1 * index for index in range(33)],
        soft_dof_pos_limit=0.5,
        init_state=dataclasses.replace(
            g1_29dof.init_state,
            default_joint_angles=dict.fromkeys(joint_names, 0.0),
        ),
    )


def _legacy_actuator_cfgs(robot):
    from isaaclab.actuators import IdealPDActuatorCfg

    return [
        IdealPDActuatorCfg(
            joint_names_expr=[joint_name],
            effort_limit=robot.dof_effort_limit_list[index],
            velocity_limit=robot.dof_vel_limit_list[index],
            stiffness=0,
            damping=0,
            armature=robot.dof_armature_list[index],
            friction=robot.dof_joint_friction_list[index],
        )
        for index, joint_name in enumerate(robot.dof_names)
    ]


def _instantiate_actuator(cfg, joint_names):
    import torch

    count = len(joint_names)
    defaults = torch.zeros((2, count), dtype=torch.float32)
    return cfg.class_type(
        cfg=copy.deepcopy(cfg),
        joint_names=joint_names,
        joint_ids=slice(None),
        num_envs=2,
        device="cpu",
        stiffness=defaults,
        damping=defaults,
        armature=defaults,
        friction=defaults,
        dynamic_friction=defaults,
        viscous_friction=defaults,
        effort_limit=torch.full_like(defaults, 1.0e6),
        velocity_limit=torch.full_like(defaults, 1.0e6),
    )


def _check_grouped_actuator_config() -> None:
    import torch
    from isaaclab.actuators import IdealPDActuator
    from isaaclab.utils.types import ArticulationActions

    from holosoma.simulator.isaacsim.isaacsim_articulation_cfg import build_robot_articulation_cfg

    robot = _synthetic_robot_config()
    articulation_cfg = build_robot_articulation_cfg(robot)
    assert list(articulation_cfg.actuators) == ["all_dofs"]
    grouped_cfg = articulation_cfg.actuators["all_dofs"]
    assert grouped_cfg.class_type is IdealPDActuator
    assert grouped_cfg.joint_names_expr == robot.dof_names

    expected_properties = {
        "effort_limit": robot.dof_effort_limit_list,
        "velocity_limit": robot.dof_vel_limit_list,
        "armature": robot.dof_armature_list,
        "friction": robot.dof_joint_friction_list,
        "stiffness": [0.0] * len(robot.dof_names),
        "damping": [0.0] * len(robot.dof_names),
    }
    for property_name, values in expected_properties.items():
        expected = dict(zip(robot.dof_names, values))
        assert getattr(grouped_cfg, property_name) == expected, f"{property_name} mapping changed"

    # Isaac Lab supplies articulation joints in simulator order. Reverse that order to prove
    # every property follows its joint name rather than the source-list position.
    simulator_names = list(reversed(robot.dof_names))
    grouped = _instantiate_actuator(grouped_cfg, simulator_names)
    for property_name, values in expected_properties.items():
        source_mapping = dict(zip(robot.dof_names, values))
        expected = torch.tensor([[source_mapping[name] for name in simulator_names]] * 2)
        assert torch.equal(getattr(grouped, property_name), expected), f"{property_name} resolved by source order"

    # Compare the grouped model against the former one-model-per-joint implementation. Include
    # unsaturated commands and both signs of all-joint saturation.
    effort_limits = torch.tensor([grouped_cfg.effort_limit[name] for name in simulator_names])
    commanded = torch.stack((effort_limits * 0.5, -effort_limits * 1.25))
    grouped_action = ArticulationActions(
        joint_positions=torch.zeros_like(commanded),
        joint_velocities=torch.zeros_like(commanded),
        joint_efforts=commanded.clone(),
    )
    grouped_output = grouped.compute(grouped_action, torch.zeros_like(commanded), torch.zeros_like(commanded))

    legacy_applied = torch.empty_like(commanded)
    legacy_computed = torch.empty_like(commanded)
    legacy_cfg_by_name = dict(zip(robot.dof_names, _legacy_actuator_cfgs(robot)))
    for simulator_index, joint_name in enumerate(simulator_names):
        legacy = _instantiate_actuator(legacy_cfg_by_name[joint_name], [joint_name])
        joint_command = commanded[:, simulator_index : simulator_index + 1]
        legacy_action = ArticulationActions(
            joint_positions=torch.zeros_like(joint_command),
            joint_velocities=torch.zeros_like(joint_command),
            joint_efforts=joint_command.clone(),
        )
        legacy.compute(legacy_action, torch.zeros_like(joint_command), torch.zeros_like(joint_command))
        legacy_applied[:, simulator_index] = legacy.applied_effort[:, 0]
        legacy_computed[:, simulator_index] = legacy.computed_effort[:, 0]

    assert torch.equal(grouped.computed_effort, legacy_computed)
    assert torch.equal(grouped.applied_effort, legacy_applied)
    assert torch.equal(grouped_output.joint_efforts, legacy_applied)
    assert torch.equal(grouped.applied_effort[0], commanded[0])
    assert torch.equal(grouped.applied_effort[1], -effort_limits)
    assert grouped_output.joint_positions is None
    assert grouped_output.joint_velocities is None


def _check_dof_position_limits() -> None:
    import torch

    from holosoma.simulator.shared.dof_limits import build_dof_limits_from_config

    robot = _synthetic_robot_config()
    hard_limits, soft_limits, _, _ = build_dof_limits_from_config(robot, len(robot.dof_names), "cpu")

    expected_hard_limits = torch.tensor(
        list(zip(robot.dof_pos_lower_limit_list, robot.dof_pos_upper_limit_list)),
        dtype=torch.float32,
    )
    expected_soft_limits = torch.tensor(
        [[-float(index) - 0.5, float(index) + 2.5] for index in range(len(robot.dof_names))],
        dtype=torch.float32,
    )
    assert torch.equal(hard_limits, expected_hard_limits)
    assert torch.equal(soft_limits, expected_soft_limits)


def _check_contact_sensor_config() -> None:
    from isaaclab.sensors.contact_sensor.contact_sensor_data import ContactSensorData

    from holosoma.simulator.isaacsim.isaacsim_articulation_cfg import build_contact_sensor_cfg

    cfg = build_contact_sensor_cfg(history_length=3, debug_vis=False)
    assert cfg.track_air_time is False
    assert cfg.history_length == 3
    assert cfg.update_period == 0.005
    assert cfg.force_threshold == 10.0
    assert cfg.debug_vis is False
    assert build_contact_sensor_cfg(history_length=3, debug_vis=True).debug_vis is True

    # These are the only ContactSensorData fields Holosoma reads. They remain available when
    # duration tracking is disabled; only the unused air/contact-time values stay unallocated.
    data = ContactSensorData()
    assert hasattr(data, "net_forces_w")
    assert hasattr(data, "net_forces_w_history")
    assert data.last_air_time is None
    assert data.current_air_time is None
    assert data.last_contact_time is None
    assert data.current_contact_time is None


def main() -> int:
    parser = argparse.ArgumentParser(description="IsaacSim articulation cfg assertion harness.")
    parser.add_argument("--result-file", default=None, help="write 'OK' here after all checks pass")
    args = parser.parse_args()

    from isaaclab.app import AppLauncher

    simulation_app = AppLauncher(headless=True).app
    try:
        _check_grouped_actuator_config()
        _check_dof_position_limits()
        _check_contact_sensor_config()
    except BaseException:
        print("ISAACSIM ARTICULATION CFG ASSERT FAILED:\n" + traceback.format_exc(), flush=True)
        simulation_app.close()
        return 1

    if args.result_file:
        Path(args.result_file).write_text("OK\n")
    simulation_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
