"""Headless checks for grouped Isaac Lab actuator and contact-sensor configuration."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import sys
import tempfile
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


def _legacy_actuator_cfgs(robot, *, backend):
    from isaaclab.actuators import IdealPDActuatorCfg

    effort_key = "actuator_effort_limit" if backend.is_isaacsim6 else "effort_limit"
    velocity_key = "actuator_velocity_limit" if backend.is_isaacsim6 else "velocity_limit"
    return [
        IdealPDActuatorCfg(
            joint_names_expr=[joint_name],
            **{
                effort_key: robot.dof_effort_limit_list[index],
                velocity_key: robot.dof_vel_limit_list[index],
            },
            stiffness=0,
            damping=0,
            armature=robot.dof_armature_list[index],
            friction=robot.dof_joint_friction_list[index],
        )
        for index, joint_name in enumerate(robot.dof_names)
    ]


def _instantiate_actuator(cfg, joint_names, *, backend):
    import torch
    from isaaclab.utils import string_to_callable

    count = len(joint_names)
    defaults = torch.zeros((2, count), dtype=torch.float32)
    joint_properties = {}
    if not backend.is_isaacsim6:
        joint_properties = {
            "armature": defaults,
            "friction": defaults,
            "dynamic_friction": defaults,
            "viscous_friction": defaults,
        }
    actuator_class = string_to_callable(cfg.class_type) if isinstance(cfg.class_type, str) else cfg.class_type
    effort_key = "actuator_effort_limit" if backend.is_isaacsim6 else "effort_limit"
    velocity_key = "actuator_velocity_limit" if backend.is_isaacsim6 else "velocity_limit"
    return actuator_class(
        cfg=copy.deepcopy(cfg),
        joint_names=joint_names,
        joint_ids=slice(None),
        num_envs=2,
        device="cpu",
        stiffness=defaults,
        damping=defaults,
        **joint_properties,
        **{
            effort_key: torch.full_like(defaults, 1.0e6),
            velocity_key: torch.full_like(defaults, 1.0e6),
        },
    )


def _check_grouped_actuator_config(*, backend) -> None:
    import torch
    from isaaclab.actuators import IdealPDActuator
    from isaaclab.utils import string_to_callable
    from isaaclab.utils.types import ArticulationActions

    from holosoma.simulator.isaacsim.isaacsim_articulation_cfg import build_robot_articulation_cfg

    native = backend.is_isaacsim6
    robot = _synthetic_robot_config()
    articulation_cfg = build_robot_articulation_cfg(robot, backend=backend)
    assert list(articulation_cfg.actuators) == ["all_dofs"]
    grouped_cfg = articulation_cfg.actuators["all_dofs"]
    actuator_class = (
        string_to_callable(grouped_cfg.class_type)
        if isinstance(grouped_cfg.class_type, str)
        else grouped_cfg.class_type
    )
    assert actuator_class is IdealPDActuator
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
        expected_mapping = dict(zip(robot.dof_names, values, strict=False))
        cfg_name = (
            f"actuator_{property_name}"
            if native and property_name in {"effort_limit", "velocity_limit"}
            else property_name
        )
        assert getattr(grouped_cfg, cfg_name) == expected_mapping, f"{property_name} mapping changed"
    if native:
        assert grouped_cfg.joint_effort_limit is None, "authored solver effort limits must remain authoritative"

    # Isaac Lab supplies articulation joints in simulator order. Reverse that order to prove
    # every property follows its joint name rather than the source-list position.
    simulator_names = list(reversed(robot.dof_names))
    grouped = _instantiate_actuator(grouped_cfg, simulator_names, backend=backend)
    for property_name, values in expected_properties.items():
        source_mapping = dict(zip(robot.dof_names, values, strict=False))
        expected = torch.tensor([[source_mapping[name] for name in simulator_names]] * 2)
        if native and property_name in {"armature", "friction"}:
            # These values moved from actuator runtime buffers to articulation
            # joint properties. The live DOF-order harness checks their application.
            from isaaclab.actuators import resolve_joint_parameter

            actual = resolve_joint_parameter(getattr(grouped_cfg, property_name), 0.0, simulator_names, 2, "cpu")
            assert torch.equal(actual, expected), f"{property_name} resolved by source order"
            continue
        runtime_name = (
            f"actuator_{property_name}"
            if native and property_name in {"effort_limit", "velocity_limit"}
            else property_name
        )
        assert torch.equal(getattr(grouped, runtime_name), expected), f"{property_name} resolved by source order"

    # Compare the grouped model against the former one-model-per-joint implementation. Include
    # unsaturated commands and both signs of all-joint saturation.
    effort_limits = torch.tensor([robot.dof_effort_limit_list[robot.dof_names.index(name)] for name in simulator_names])
    commanded = torch.stack((effort_limits * 0.5, -effort_limits * 1.25))
    grouped_action = ArticulationActions(
        joint_positions=torch.zeros_like(commanded),
        joint_velocities=torch.zeros_like(commanded),
        joint_efforts=commanded.clone(),
    )
    grouped_output = grouped.compute(grouped_action, torch.zeros_like(commanded), torch.zeros_like(commanded))

    legacy_applied = torch.empty_like(commanded)
    legacy_computed = torch.empty_like(commanded)
    legacy_cfg_by_name = dict(zip(robot.dof_names, _legacy_actuator_cfgs(robot, backend=backend), strict=False))
    for simulator_index, joint_name in enumerate(simulator_names):
        legacy = _instantiate_actuator(legacy_cfg_by_name[joint_name], [joint_name], backend=backend)
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

    # Exercise the other sign at both boundaries; a one-sided clip must fail.
    commanded = torch.stack((effort_limits * 1.25, -effort_limits * 0.5))
    action = ArticulationActions(
        joint_positions=torch.zeros_like(commanded),
        joint_velocities=torch.zeros_like(commanded),
        joint_efforts=commanded.clone(),
    )
    output = grouped.compute(action, torch.zeros_like(commanded), torch.zeros_like(commanded))
    assert torch.equal(grouped.computed_effort, commanded)
    assert torch.equal(grouped.applied_effort[0], effort_limits)
    assert torch.equal(grouped.applied_effort[1], commanded[1])
    assert torch.equal(output.joint_efforts, grouped.applied_effort)


def _check_dof_position_limits() -> None:
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


def _check_preserved_fixed_joint_config(*, backend) -> None:
    """Native import must retain dont_collapse bodies without disabling ordinary merging."""
    from holosoma.config_values.robot import g1_29dof, t1_29dof_waist_wrist
    from holosoma.simulator.isaacsim.isaacsim_articulation_cfg import build_robot_articulation_cfg

    native = backend.is_isaacsim6
    for robot in (g1_29dof, t1_29dof_waist_wrist):
        assert robot.asset.collapse_fixed_joints is True
        spawn = build_robot_articulation_cfg(robot, backend=backend).spawn
        assert spawn.merge_fixed_joints is (not native), "configured foot contact bodies must survive native import"

    with tempfile.TemporaryDirectory() as directory:
        urdf_path = Path(directory) / "fixed_joint_fixture.urdf"
        for joint_type, preservation, collapse, expected_merge in (
            ("fixed", "", True, True),
            ("fixed", 'dont_collapse="false"', True, True),
            ("revolute", 'dont_collapse="true"', True, True),
            ("fixed", 'dont_collapse="true"', True, not native),
            ("fixed", 'dont_collapse="TrUe"', True, not native),
            ("fixed", 'dont_collapse="1"', True, not native),
            ("fixed", 'dont_collapse="true"', False, False),
        ):
            urdf_path.write_text(
                f'<robot name="fixture"><link name="parent"/><link name="child"/>'
                f'<joint name="fixture_joint" type="{joint_type}" {preservation}>'
                '<parent link="parent"/><child link="child"/></joint></robot>'
            )
            robot = dataclasses.replace(
                g1_29dof,
                asset=dataclasses.replace(
                    g1_29dof.asset,
                    asset_root=directory,
                    urdf_file=urdf_path.name,
                    collapse_fixed_joints=collapse,
                ),
            )
            spawn = build_robot_articulation_cfg(robot, backend=backend).spawn
            assert spawn.merge_fixed_joints is expected_merge, (
                f"fixed-joint import policy changed for {joint_type=}, {preservation=}, {collapse=}"
            )


def _check_contact_sensor_config(*, backend) -> None:
    from holosoma.simulator.isaacsim.isaacsim_articulation_cfg import build_contact_sensor_cfg

    native = backend.is_isaacsim6
    if native:
        from isaaclab_physx.sensors import ContactSensorData
    else:
        from isaaclab.sensors.contact_sensor.contact_sensor_data import ContactSensorData

    cfg = build_contact_sensor_cfg(debug_vis=False, backend=backend)
    assert cfg.track_air_time is False
    # 0.005 would hold the sensor stale across substeps at any fps above 200.
    assert cfg.update_period == 0.0
    assert cfg.force_threshold == 10.0
    assert cfg.debug_vis is False
    assert build_contact_sensor_cfg(debug_vis=True, backend=backend).debug_vis is True

    # Force data remains available when duration tracking is disabled. Lab 3
    # allocates proxy buffers explicitly; Lab 2 uses a dataclass container.
    data = ContactSensorData()
    if native:
        data.create_buffers(
            num_envs=2,
            num_sensors=3,
            num_filter_shapes=0,
            history_length=cfg.history_length,
            track_pose=cfg.track_pose,
            track_air_time=cfg.track_air_time,
            track_contact_points=cfg.track_contact_points,
            track_friction_forces=cfg.track_friction_forces,
            supports_total_contact_forces=True,
            device="cpu",
        )
        assert data.net_normal_forces_w.torch.shape == (2, 3, 3)
        assert data.net_forces_w.torch.shape == (2, 3, 3)
    assert hasattr(data, "net_forces_w")
    assert data.last_air_time is None
    assert data.current_air_time is None
    assert data.last_contact_time is None
    assert data.current_contact_time is None


def main() -> int:
    parser = argparse.ArgumentParser(description="IsaacSim articulation cfg assertion harness.")
    parser.add_argument("--simulator", choices=["isaacsim", "isaacsim6"], default="isaacsim")
    parser.add_argument("--result-file", default=None, help="write 'OK' here after all checks pass")
    args = parser.parse_args()

    from holosoma.simulator.isaacsim.backends import get_backend

    backend = get_backend("isaacsim6" if args.simulator == "isaacsim6" else "isaacsim5")
    simulation_app = backend.launch_app({"headless": True})
    try:
        _check_grouped_actuator_config(backend=backend)
        _check_dof_position_limits()
        _check_preserved_fixed_joint_config(backend=backend)
        _check_contact_sensor_config(backend=backend)
    except BaseException:
        print("ISAACSIM ARTICULATION CFG ASSERT FAILED:\n" + traceback.format_exc(), flush=True)
        if backend.is_isaacsim6:
            simulation_app.close(exit_code=1)
        else:
            simulation_app.close()
        return 1

    if args.result_file:
        Path(args.result_file).write_text("OK\n")
    simulation_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
