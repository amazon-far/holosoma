"""Headless checks for grouped Isaac Lab actuator and contact-sensor configuration."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import importlib
import sys
import traceback
from pathlib import Path

from tests.simulators._sim_harness import ISAAC_LANE_PACKAGES, class_type_is, isaac_lane_module

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


def _resolved_class(class_type):
    """The actual class behind a cfg's ``class_type``, on either Isaac Lab generation.

    2.3.2 stores the class. 3.0 stores an ``isaaclab.utils.string.ResolvableString`` holding
    ``"module:QualName"``; it is callable (so it can construct the object) but it is not the
    class, so ``inspect.signature`` on it describes ``ResolvableString.__call__`` rather than the
    constructor -- which makes a signature-based kwarg filter silently drop *everything*.
    """
    if isinstance(class_type, type):
        return class_type
    module_name, _, qual_name = str(class_type).partition(":")
    obj = importlib.import_module(module_name)
    for part in qual_name.split("."):
        obj = getattr(obj, part)
    return obj


def _instantiate_actuator(cfg, joint_names):
    import inspect

    import torch

    count = len(joint_names)
    defaults = torch.zeros((2, count), dtype=torch.float32)
    kwargs = {
        "cfg": copy.deepcopy(cfg),
        "joint_names": joint_names,
        "joint_ids": slice(None),
        "num_envs": 2,
        "device": "cpu",
        "stiffness": defaults,
        "damping": defaults,
        "armature": defaults,
        "friction": defaults,
        "dynamic_friction": defaults,
        "viscous_friction": defaults,
        "effort_limit": torch.full_like(defaults, 1.0e6),
        "velocity_limit": torch.full_like(defaults, 1.0e6),
    }
    # Pass only what this Isaac Lab's actuator constructor actually accepts. 2.3.2 took the
    # per-joint physical properties (armature, friction, dynamic_friction, viscous_friction)
    # through `__init__`; 3.0's `IdealPDActuator.__init__` takes only the gains and limits, so
    # passing the full 2.3.2 set raises `unexpected keyword argument 'armature'`.
    #
    # Filtering here is safe *for this harness*, which checks how the cfg was populated rather
    # than what the actuator does with it. It is NOT a statement that the dropped properties are
    # applied: 3.0 silently ignores `ActuatorBaseCfg.armature` / `.friction` entirely, which the
    # 6.1 backend compensates for in `IsaacSim._write_joint_drive_properties`.
    actuator_cls = _resolved_class(cfg.class_type)
    accepted = set(inspect.signature(actuator_cls).parameters)
    return actuator_cls(**{k: v for k, v in kwargs.items() if k in accepted})


def _check_grouped_actuator_config(lane: str) -> None:
    import torch
    from isaaclab.actuators import IdealPDActuator
    from isaaclab.utils.types import ArticulationActions

    build_robot_articulation_cfg = isaac_lane_module(lane, "isaacsim_articulation_cfg").build_robot_articulation_cfg

    robot = _synthetic_robot_config()
    articulation_cfg = build_robot_articulation_cfg(robot)
    assert list(articulation_cfg.actuators) == ["all_dofs"]
    grouped_cfg = articulation_cfg.actuators["all_dofs"]
    assert class_type_is(grouped_cfg.class_type, IdealPDActuator)
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
        expected = dict(zip(robot.dof_names, values, strict=False))
        assert getattr(grouped_cfg, property_name) == expected, f"{property_name} mapping changed"

    # Isaac Lab supplies articulation joints in simulator order. Reverse that order to prove
    # every property follows its joint name rather than the source-list position.
    simulator_names = list(reversed(robot.dof_names))
    grouped = _instantiate_actuator(grouped_cfg, simulator_names)
    for property_name, values in expected_properties.items():
        if not hasattr(grouped, property_name):
            # Isaac Lab 3.0's actuator keeps only the gains and limits; `armature`, `friction`,
            # `dynamic_friction` and `viscous_friction` were dropped from both its constructor
            # and its state. The cfg-level assertion above still covers what holosoma controls
            # (that each value is mapped to the right joint name), and the 6.1 backend applies
            # the dropped properties to the simulation itself in
            # `IsaacSim._write_joint_drive_properties`, because 3.0 otherwise ignores them.
            continue
        source_mapping = dict(zip(robot.dof_names, values, strict=False))
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
    legacy_cfg_by_name = dict(zip(robot.dof_names, _legacy_actuator_cfgs(robot), strict=False))
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


def _check_dof_position_limits(lane: str) -> None:
    import torch

    from holosoma.simulator.shared.dof_limits import build_dof_limits_from_config

    robot = _synthetic_robot_config()
    hard_limits, soft_limits, _, _ = build_dof_limits_from_config(robot, len(robot.dof_names), "cpu")

    expected_hard_limits = torch.tensor(
        list(zip(robot.dof_pos_lower_limit_list, robot.dof_pos_upper_limit_list, strict=False)),
        dtype=torch.float32,
    )
    expected_soft_limits = torch.tensor(
        [[-float(index) - 0.5, float(index) + 2.5] for index in range(len(robot.dof_names))],
        dtype=torch.float32,
    )
    assert torch.equal(hard_limits, expected_hard_limits)
    assert torch.equal(soft_limits, expected_soft_limits)


def _check_contact_sensor_config(lane: str) -> None:
    from isaaclab.sensors.contact_sensor.contact_sensor_data import ContactSensorData

    build_contact_sensor_cfg = isaac_lane_module(lane, "isaacsim_articulation_cfg").build_contact_sensor_cfg

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
    # Isaac Lab 3.0 renamed these to `net_normal_forces_w*`. The old names survive as properties
    # but are PhysX-unavailable (3.0 cannot report a combined normal+friction force on PhysX, so
    # they fall back and warn), which makes `hasattr` False rather than absent. Accept either
    # spelling: what matters is that holosoma has *some* per-body contact force to read, which is
    # the contract this line is pinning. The 6.1 backend reads the explicit normal-force name.
    # Probed on the *class*, not the instance: in 3.0 these are properties over backing storage
    # that is unset until the sensor initialises, so reading one off a bare `ContactSensorData()`
    # raises and `hasattr` reports False even though the attribute exists. Looking at the class
    # finds the descriptor without invoking it.
    for base in ("net_forces_w", "net_forces_w_history"):
        renamed = base.replace("net_", "net_normal_", 1)
        assert hasattr(type(data), base) or hasattr(type(data), renamed), (
            f"ContactSensorData exposes neither {base!r} nor its 3.x {renamed!r} equivalent"
        )
    # The duration fields are checked for *presence*, not for a None value.
    #
    # On 2.3.2 `ContactSensorData` is a dataclass whose duration fields default to None when
    # duration tracking is off, so `is None` was the natural assertion. On 3.0 they are
    # properties over per-backend storage that is not allocated until the sensor initialises, so
    # reading one off a bare instance raises `AttributeError: no attribute '_last_air_time'`.
    #
    # Worth knowing why a bare `ContactSensorData()` is backend-specific at all: 3.0's data
    # classes are factory-dispatched, and with no `SimulationContext` alive the factory resolves
    # to **Newton** -- the traceback for this line came from `isaaclab_newton`, not
    # `isaaclab_physx`. So this construction cannot meaningfully assert PhysX storage semantics;
    # what holosoma actually depends on is that the names exist, which is what is pinned here.
    for duration_field in ("last_air_time", "current_air_time", "last_contact_time", "current_contact_time"):
        assert hasattr(type(data), duration_field), f"ContactSensorData lost {duration_field!r}"


def main() -> int:
    parser = argparse.ArgumentParser(description="IsaacSim articulation cfg assertion harness.")
    parser.add_argument("--result-file", default=None, help="write 'OK' here after all checks pass")
    # Isaac-only harness, but not generation-only: it imports configuration builders that
    # live inside a backend package, so it must be told which Isaac Sim lane to exercise.
    # Defaults to the 5.1 lane so existing invocations and CI are unchanged.
    parser.add_argument("--simulator", default="isaacsim", choices=sorted(ISAAC_LANE_PACKAGES))
    args = parser.parse_args()

    from isaaclab.app import AppLauncher

    simulation_app = AppLauncher(headless=True).app
    try:
        _check_grouped_actuator_config(args.simulator)
        _check_dof_position_limits(args.simulator)
        _check_contact_sensor_config(args.simulator)
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
