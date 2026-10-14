"""Headless checks for grouped Isaac Lab actuator, fixed-base weld and contact-sensor configuration."""

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
        dof_pos_limits=dict(
            zip(
                joint_names,
                zip([float(-2 * (index + 1)) for index in range(33)], [float(2 * (index + 2)) for index in range(33)]),
            )
        ),
        dof_effort_limits=dict(zip(joint_names, [float(10 + index) for index in range(33)])),
        dof_vel_limits=dict(zip(joint_names, [float(20 + index) for index in range(33)])),
        dof_armature=dict(zip(joint_names, [0.01 * index for index in range(33)])),
        dof_joint_friction=dict(zip(joint_names, [0.1 * index for index in range(33)])),
        init_state=dataclasses.replace(
            g1_29dof.init_state,
            default_joint_positions=dict.fromkeys(joint_names, 0.0),
        ),
        # g1's joint tags name g1 joints; the synthetic DOF set has none of them.
        joint_tags={},
        # This synthetic articulation has no SDK motors.
        bridge=dataclasses.replace(g1_29dof.bridge, controlled_dof_names=(), excluded_dof_names=()),
    )


def _legacy_actuator_cfgs(robot):
    from isaaclab.actuators import IdealPDActuatorCfg

    return [
        IdealPDActuatorCfg(
            joint_names_expr=[joint_name],
            effort_limit=robot.dof_effort_limits[joint_name],
            velocity_limit=robot.dof_vel_limits[joint_name],
            stiffness=0,
            damping=0,
            armature=robot.dof_armature[joint_name],
            friction=robot.dof_joint_friction[joint_name],
        )
        for joint_name in robot.dof_names
    ]


def _instantiate_actuator(cfg, joint_names, *, joint_ids=None, armature=0.0, friction=0.0):
    import torch

    count = len(joint_names)
    defaults = torch.zeros((2, count), dtype=torch.float32)
    return cfg.class_type(
        cfg=copy.deepcopy(cfg),
        joint_names=joint_names,
        joint_ids=slice(None) if joint_ids is None else joint_ids,
        num_envs=2,
        device="cpu",
        stiffness=defaults,
        damping=defaults,
        armature=torch.full_like(defaults, armature),
        friction=torch.full_like(defaults, friction),
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
        "effort_limit": robot.dof_effort_limits,
        "velocity_limit": robot.dof_vel_limits,
        "armature": robot.dof_armature,
        "friction": robot.dof_joint_friction,
        "stiffness": dict.fromkeys(robot.dof_names, 0.0),
        "damping": dict.fromkeys(robot.dof_names, 0.0),
    }
    for property_name, values in expected_properties.items():
        expected = values
        assert getattr(grouped_cfg, property_name) == (
            0.0 if property_name in ("stiffness", "damping") else expected
        ), f"{property_name} mapping changed"

    # Isaac Lab supplies articulation joints in simulator order. Reverse that order to prove
    # every property follows its joint name rather than the source-list position.
    simulator_names = list(reversed(robot.dof_names))
    grouped = _instantiate_actuator(grouped_cfg, simulator_names)
    for property_name, values in expected_properties.items():
        source_mapping = values
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


def _check_none_maps_defer_to_the_asset() -> None:
    """A ``None`` property map lets IsaacLab retain the USD/URDF values."""
    from holosoma.simulator.isaacsim.isaacsim_articulation_cfg import build_robot_articulation_cfg

    robot = dataclasses.replace(
        _synthetic_robot_config(), dof_effort_limits=None, dof_vel_limits=None, dof_armature=None
    )
    grouped_cfg = build_robot_articulation_cfg(robot).actuators["all_dofs"]
    assert grouped_cfg.effort_limit is None
    assert grouped_cfg.velocity_limit is None
    assert grouped_cfg.armature is None
    assert grouped_cfg.friction == robot.dof_joint_friction


def _check_joint_names_are_literal() -> None:
    import torch
    from isaaclab.utils.string import resolve_matching_names

    from holosoma.simulator.isaacsim.isaacsim_articulation_cfg import build_robot_articulation_cfg

    robot = _synthetic_robot_config()
    names = ["joint.0", "jointx0", "joint[0]"]
    robot = dataclasses.replace(
        robot,
        dof_names=names,
        joint_tags={},
        dof_pos_limits=None,
        dof_effort_limits=dict(zip(names, [3.0, 5.0, 7.0])),
        dof_vel_limits=None,
        dof_armature=None,
        dof_joint_friction=None,
        init_state=dataclasses.replace(robot.init_state, default_joint_positions=dict.fromkeys(names, 0.0)),
    )
    cfg = build_robot_articulation_cfg(robot)
    actuator_cfg = cfg.actuators["all_dofs"]
    native_names = list(reversed(names))
    ids, selected = resolve_matching_names(actuator_cfg.joint_names_expr, native_names, preserve_order=True)
    assert selected == names and ids == [2, 1, 0]
    actuator = _instantiate_actuator(actuator_cfg, native_names)
    torch.testing.assert_close(actuator.effort_limit, torch.tensor([[7.0, 5.0, 3.0]] * 2))
    _, initialized = resolve_matching_names(list(cfg.init_state.joint_pos), native_names, preserve_order=True)
    assert initialized == names


def _check_sparse_properties_and_cached_limits() -> None:
    from types import SimpleNamespace

    import torch

    from holosoma.simulator.isaacsim.isaacsim import IsaacSim
    from holosoma.simulator.isaacsim.isaacsim_articulation_cfg import build_robot_articulation_cfg

    robot = dataclasses.replace(
        _synthetic_robot_config(),
        dof_pos_limits={"test_joint_02": (-0.3, 0.4)},
        dof_effort_limits={"test_joint_00": 3.0},
        dof_vel_limits={"#pair": 5.0},
        dof_armature={"test_joint_02": 0.1},
        dof_joint_friction=None,
        joint_tags={"#pair": ["test_joint_00", "test_joint_01", "test_joint_00"]},
    )
    native_names = list(reversed(robot.dof_names))
    configs = build_robot_articulation_cfg(robot).actuators
    actuators = {}
    covered: set[str] = set()
    for key, cfg in configs.items():
        names = [name for name in native_names if name in cfg.joint_names_expr]
        assert not covered.intersection(names)
        covered.update(names)
        ids = [native_names.index(name) for name in names]
        actuator = _instantiate_actuator(cfg, names, joint_ids=ids, armature=0.6, friction=0.4)
        actuators[key] = actuator
        for i, name in enumerate(names):
            assert float(actuator.effort_limit[0, i]) == (3.0 if name == "test_joint_00" else 1.0e6)
            assert float(actuator.velocity_limit[0, i]) == (5.0 if name in robot.joint_tags["#pair"] else 1.0e6)
            torch.testing.assert_close(
                actuator.armature[:, i], torch.full((2,), 0.1 if name == "test_joint_02" else 0.6)
            )
            torch.testing.assert_close(actuator.friction[:, i], torch.full((2,), 0.4))
    assert covered == set(robot.dof_names)

    # Exercise the real getter with real actuator models and a recorded PhysX write boundary.
    writes = []
    sim = IsaacSim.__new__(IsaacSim)
    sim.robot_config = robot
    sim.sim_device = "cpu"
    sim.num_envs = 2
    sim.dof_ids = [native_names.index(name) for name in robot.dof_names]
    sim._robot = SimpleNamespace(
        num_joints=len(native_names),
        actuators=actuators,
        data=SimpleNamespace(joint_pos_limits=torch.tensor([[[-1.0, 1.0]] * len(native_names)] * 2)),
        write_joint_position_limit_to_sim=lambda values, **kwargs: writes.append((values.clone(), kwargs)),
    )
    sim._initialize_dof_limit_properties()
    limits = sim.get_dof_limits_properties()
    assert sim.get_dof_limits_properties() is limits
    assert len(writes) == 1
    torch.testing.assert_close(limits[0][2], torch.tensor([-0.3, 0.4]))
    torch.testing.assert_close(limits[0][1], torch.tensor([-1.0, 1.0]))
    torch.testing.assert_close(limits[2][:2], torch.tensor([3.0, 1.0e6]))
    for actual, write_options in writes:
        assert write_options["joint_ids"] == sim.dof_ids
        torch.testing.assert_close(actual, limits[0].unsqueeze(0).expand(2, -1, -1))


def _check_fixed_base_weld() -> None:
    """``fix_base_link`` welds the root on both spawn paths (URDF convert and authored USD); a floating base
    leaves ``fix_root_link`` unset so an authored USD keeps its own root setting."""
    import dataclasses

    import isaaclab.sim as sim_utils

    from holosoma.simulator.isaacsim.isaacsim_articulation_cfg import build_robot_articulation_cfg

    robot = _synthetic_robot_config()

    def spawn_for(fix_base_link: bool, usd_file: str | None):
        asset = dataclasses.replace(robot.asset, fix_base_link=fix_base_link, usd_file=usd_file)
        return build_robot_articulation_cfg(dataclasses.replace(robot, asset=asset)).spawn

    fixed_urdf = spawn_for(True, None)
    assert isinstance(fixed_urdf, sim_utils.UrdfFileCfg)
    assert fixed_urdf.fix_base is True
    assert fixed_urdf.articulation_props.fix_root_link is True
    assert "spawn_from_urdf_with_visual_colors" in getattr(fixed_urdf.func, "__name__", ""), fixed_urdf.func

    fixed_usd = spawn_for(True, "g1/g1_29dof.usd")
    assert isinstance(fixed_usd, sim_utils.UsdFileCfg)
    assert fixed_usd.articulation_props.fix_root_link is True

    floating_urdf = spawn_for(False, None)
    assert isinstance(floating_urdf, sim_utils.UrdfFileCfg)
    assert floating_urdf.fix_base is False
    assert not floating_urdf.articulation_props.fix_root_link

    floating_usd = spawn_for(False, "g1/g1_29dof.usd")
    assert isinstance(floating_usd, sim_utils.UsdFileCfg)
    assert not floating_usd.articulation_props.fix_root_link


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
        _check_none_maps_defer_to_the_asset()
        _check_joint_names_are_literal()
        _check_sparse_properties_and_cached_limits()
        _check_fixed_base_weld()
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
