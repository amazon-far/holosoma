"""Live contact contracts for Classic MuJoCo, MuJoCo Warp, and Isaac Sim."""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import pytest

if sys.path and sys.path[0].endswith("simulators"):
    sys.path.pop(0)
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from holosoma.hooks import Phase  # noqa: E402
from tests.simulators._sim_harness import build_run_sim_config, launch_simulator, run_and_hard_exit  # noqa: E402


def _build(backend: str, num_envs: int, device: str | None = None, *, configured_contacts: bool = True):
    import torch

    from holosoma.config_types.scene import PhysicsConfig, RigidObjectConfig, SceneConfig
    from holosoma.config_types.sensor import ContactSensorConfig, ContactTargetConfig

    def actor(name: str):
        return ContactTargetConfig(target_kind="actor", target=name)

    terrain = ContactTargetConfig(target_kind="terrain")
    sensors = {
        "pairs": ContactSensorConfig(
            sources=[actor("top"), actor("floor_box")],
            counterparts=[actor("pedestal"), terrain],
        ),
        "pairs_reversed": ContactSensorConfig(
            sources=[actor("top"), actor("floor_box")],
            counterparts=[terrain, actor("pedestal")],
        ),
        "__holosoma_contact_0": ContactSensorConfig(sources=[actor("floor_box")]),
        "reaction": ContactSensorConfig(sources=[actor("pedestal")], counterparts=[actor("top")]),
        "floor_total": ContactSensorConfig(sources=[actor("floor_box")], counterparts=[terrain], force_mode="total"),
        "robot_ground": ContactSensorConfig(counterparts=[terrain]),
        "body_ground": ContactSensorConfig(
            sources=[
                ContactTargetConfig(target_kind="robot_link", target="torso_link"),
                ContactTargetConfig(target_kind="robot_link", target="left_ankle_roll_link"),
            ],
            counterparts=[terrain],
        ),
        "body_unfiltered": ContactSensorConfig(
            sources=[
                ContactTargetConfig(target_kind="robot_link", target="torso_link"),
                ContactTargetConfig(target_kind="robot_link", target="left_ankle_roll_link"),
            ]
        ),
        "box_robot": ContactSensorConfig(sources=[actor("top")], counterparts=[ContactTargetConfig()]),
        "box_foot": ContactSensorConfig(
            sources=[actor("top")],
            counterparts=[ContactTargetConfig(target_kind="robot_link", target="left_ankle_roll_link")],
        ),
        "foot_box": ContactSensorConfig(
            sources=[ContactTargetConfig(target_kind="robot_link", target="left_ankle_roll_link")],
            counterparts=[actor("top")],
        ),
        "marker": ContactSensorConfig(
            sources=[ContactTargetConfig(target_kind="robot_link", target="left_foot_contact_point")]
        ),
        "to_marker": ContactSensorConfig(
            sources=[actor("floor_box")],
            counterparts=[ContactTargetConfig(target_kind="robot_link", target="left_foot_contact_point")],
        ),
        "body_pair": ContactSensorConfig(
            sources=[ContactTargetConfig(target_kind="robot_link", target="left_ankle_roll_link")],
            counterparts=[ContactTargetConfig(target_kind="robot_link", target="right_ankle_roll_link")],
        ),
    }
    config = build_run_sim_config(
        backend, "empty", "g1-29dof", "terrain_locomotion_plane", sensors=sensors if configured_contacts else {}
    )
    asset = RigidObjectConfig(
        urdf_file="holosoma/data/scene_objects/boxes/small_box.urdf",
        usd_file="holosoma/data/scene_objects/boxes/small_box.usda",
    )
    scene = SceneConfig(
        rigid_objects={
            "pedestal": dataclasses.replace(asset, position=[2, 0, 0.3], fixed=True),
            "top": dataclasses.replace(asset, position=[2, 0, 0.6], physics=PhysicsConfig(mass=2)),
            "floor_box": dataclasses.replace(asset, position=[2.5, 0, 0.3], physics=PhysicsConfig(mass=3)),
        }
    )
    config = dataclasses.replace(
        config,
        scene=scene,
        device=device or ("cpu" if backend == "mujoco" else "cuda:0"),
        training=dataclasses.replace(config.training, num_envs=num_envs, headless=True),
        simulator=dataclasses.replace(
            config.simulator, config=dataclasses.replace(config.simulator.config, debug_viz=False)
        ),
    )
    sim, device, _ = launch_simulator(config, device=config.device)
    sim.set_headless(True)
    sim.setup()
    sim.setup_terrain()
    sim.load_assets()
    origins = torch.zeros(num_envs, 3, device=device)
    origins[:, 0] = torch.arange(num_envs, device=device) * 8
    initial = config.robot.init_state
    root = torch.tensor(initial.root_state_list(), device=device)
    sim.create_envs(num_envs, origins, root)
    sim.prepare_sim()
    return sim


def _control_step(sim):
    sim.hooks.emit(Phase.FRAME_BEGIN)
    for _ in range(sim.simulator_config.sim.control_decimation_steps):
        sim.simulate_at_each_physics_step()
    sim.hooks.emit(Phase.FRAME_END)


def _settle_boxes(sim):
    import torch

    ids = torch.arange(sim.num_envs, device=sim.sim_device)
    heights = torch.tensor([0.4, 0.05], device=sim.sim_device).repeat_interleave(sim.num_envs)
    stable_substeps = 0
    decimation = sim.simulator_config.sim.control_decimation_steps
    # Resting PhysX bodies retain small motion: allow 1 mm/s and 0.01 rad/s.
    linear_limit, angular_limit = 1e-3, 1e-2
    # Use body motion to establish equilibrium independently of the force measurement.
    for frame in range(600):
        sim.hooks.emit(Phase.FRAME_BEGIN)
        for _ in range(decimation):
            sim.simulate_at_each_physics_step()
            states = sim.get_actor_states(["top", "floor_box"], ids)
            settled = (
                (torch.linalg.vector_norm(states[:, 7:10], dim=-1) < linear_limit).all()
                and (torch.linalg.vector_norm(states[:, 10:13], dim=-1) < angular_limit).all()
                and ((states[:, 2] - heights).abs() < 0.005).all()
            )
            stable_substeps = stable_substeps + 1 if settled else 0
        sim.hooks.emit(Phase.FRAME_END)
        if frame >= 199 and stable_substeps >= 10 * decimation:
            break
    else:
        raise AssertionError(f"Contact fixture did not settle: {states.tolist()}")

    # Keep an unobserved interval so host reads cannot hide a sampling synchronization error.
    for _ in range(20):
        _control_step(sim)
    states = sim.get_actor_states(["top", "floor_box"], ids)
    assert (torch.linalg.vector_norm(states[:, 7:10], dim=-1) < linear_limit).all(), states
    assert (torch.linalg.vector_norm(states[:, 10:13], dim=-1) < angular_limit).all(), states


def _assert_contract(sim):
    import torch

    from holosoma.utils.rotations import quat_apply

    _settle_boxes(sim)
    matrix = sim.get_contact_data("pairs", "force_matrix")
    assert matrix.shape == (sim.num_envs, 2, 2, 3), matrix.shape
    expected = torch.zeros_like(matrix)
    expected[:, 0, 0, 2] = 2 * 9.81
    expected[:, 1, 1, 2] = 3 * 9.81
    torch.testing.assert_close(matrix, expected, atol=0.5, rtol=0.02)
    torch.testing.assert_close(sim.get_contact_data("pairs_reversed", "force_matrix"), matrix[:, :, [1, 0]])
    torch.testing.assert_close(sim.get_contact_data("__holosoma_contact_0"), matrix[:, 1:].sum(dim=2))
    torch.testing.assert_close(sim.get_contact_data("reaction"), -matrix[:, 0, :1], atol=0.1, rtol=0.02)
    assert sim.contact_recorder.recorded_forces.shape == (
        sim.num_envs,
        sim.simulator_config.sim.control_decimation_steps,
        sim.num_bodies,
        3,
    )
    torch.testing.assert_close(sim.contact_recorder.latest_forces, sim.contact_forces)
    assert sim.get_contact_data("body_pair").shape == (sim.num_envs, 1, 3)
    assert torch.count_nonzero(sim.get_contact_data("body_pair")) == 0
    assert set(sim.sensor_manager.contact_names) == set(sim.sensor_config)
    assert sim.contact_forces.shape == (sim.num_envs, sim.num_bodies, 3)
    assert sim.contact_forces.dtype == torch.float32
    assert torch.count_nonzero(sim.get_contact_data("marker")) == 0
    assert torch.count_nonzero(sim.get_contact_data("to_marker")) == 0
    assert torch.count_nonzero(sim.get_contact_data("box_robot")) == 0
    robot_ground = sim.get_contact_data("robot_ground")[:, 0]
    assert torch.all(torch.linalg.vector_norm(robot_ground, dim=-1) > 1)
    # On a horizontal plane, normal forces match the vertical component of legacy total forces.
    torch.testing.assert_close(robot_ground[:, 2], sim.contact_forces[:].sum(dim=1)[:, 2], atol=1.0, rtol=0.05)
    torch.testing.assert_close(robot_ground[:, :2], torch.zeros_like(robot_ground[:, :2]), atol=0.1, rtol=0)
    torch.testing.assert_close(
        sim.get_contact_data("body_ground"), sim.get_contact_data("body_unfiltered"), atol=0.1, rtol=0.05
    )
    # Tangential slip distinguishes total force from the normal-only measurement.
    ids = torch.arange(sim.num_envs, device=sim.sim_device)
    states = sim.get_actor_states(["floor_box"], ids).clone()
    states[:, 7] = 1
    sim.set_actor_states(["floor_box"], ids, states)
    sim.simulate_at_each_physics_step()
    normal = sim.get_contact_data("pairs")[:, 1]
    total = sim.get_contact_data("floor_total")[:, 0]
    assert torch.all(normal[:, 2] > 1)
    assert torch.all(normal[:, 0].abs() < 0.1), normal
    assert torch.all(total[:, 0] < -0.1), total

    _settle_boxes(sim)

    # Reset only env 0. Rendering without a solve must not restore stale forces.
    untouched = sim.get_contact_data("pairs", "force_matrix")[1:].clone()
    one = torch.tensor([0], device=sim.sim_device)
    states = sim.get_actor_states(["top"], one).clone()
    states[:, 2] = 3
    states[:, 7:] = 0
    sim.set_actor_states(["top"], one, states)
    sim.hooks.emit(Phase.EPISODE_START, one)
    sim.render_sensors(one, 1)
    sim.render_sensors()
    assert torch.count_nonzero(sim.get_contact_data("pairs", "force_matrix")[0]) == 0
    torch.testing.assert_close(sim.get_contact_data("pairs", "force_matrix")[1:], untouched)
    _control_step(sim)
    assert torch.count_nonzero(sim.get_contact_data("pairs", "force_matrix")[0, 0]) == 0
    assert sim.get_contact_data("pairs", "force_matrix")[0, 1, 1, 2] > 1

    # A box touching the foot exercises positive robot and robot-link counterparts.
    foot = sim.body_names.index("left_ankle_roll_link")
    rotation = sim.rigid_body_quat_w[:, foot]
    offset = torch.tensor([0.075, 0, 0.015], device=sim.sim_device).repeat(sim.num_envs, 1)
    states = sim.get_actor_states(["top"], ids).clone()
    states[:, :3] = sim.rigid_body_pos_w[:, foot] + quat_apply(rotation, offset, w_last=True)
    states[:, 3:7] = rotation
    states[:, 7:] = 0
    sim.set_actor_states(["top"], ids, states)
    sim.simulate_at_each_physics_step()
    sim.render_sensors()
    box_foot = sim.get_contact_data("box_foot")
    assert torch.all(torch.linalg.vector_norm(box_foot[:, 0], dim=-1) > 1), box_foot
    assert torch.all(torch.linalg.vector_norm(sim.get_contact_data("box_robot")[:, 0], dim=-1) > 1)
    torch.testing.assert_close(sim.get_contact_data("foot_box"), -box_foot, atol=0.1, rtol=0.02)


def _assert_legacy_only(sim):
    import torch

    assert sim.sensor_manager.contact_names == []
    assert sim.get_sensor_names() == []
    _settle_boxes(sim)
    sim.hooks.emit(Phase.FRAME_BEGIN)
    for index in range(sim.simulator_config.sim.control_decimation_steps):
        sim.simulate_at_each_physics_step()
        forces = sim.contact_forces[:].clone()
        assert forces.shape == (sim.num_envs, sim.num_bodies, 3)
        assert torch.all(forces.norm(dim=-1).sum(dim=1) > 1)
        torch.testing.assert_close(sim.contact_recorder.recorded_forces[:, index], forces)
        torch.testing.assert_close(sim.contact_recorder.latest_forces, forces)
    sim.hooks.emit(Phase.FRAME_END)
    assert sim.sensor_manager.contact_names == []


def _assert_isaac_setup_errors(sim):
    from holosoma.config_types.sensor import ContactSensorConfig, ContactTargetConfig
    from holosoma.simulator.isaacsim.contact_setup import build_contacts

    cases = [
        (
            ContactSensorConfig(sources=[ContactTargetConfig(target_kind="actor", target="missing_actor")]),
            "not registered",
        ),
        (
            ContactSensorConfig(counterparts=[ContactTargetConfig(target_kind="robot_link", target="missing_link")]),
            "expected one matching prim, found 0",
        ),
        (
            ContactSensorConfig(
                counterparts=[
                    ContactTargetConfig(),
                    ContactTargetConfig(target_kind="robot_link", target="left_ankle_roll_link"),
                ]
            ),
            "counterpart groups overlap",
        ),
    ]
    used_bodies = {body for reader in sim.contact_readers.values() for group in reader.source_bodies for body in group}
    assert set(sim.contact_views) == used_bodies
    assert any(len(sensor.cfg.filter_prim_paths_expr) > 1 for sensor in sim.contact_views.values())
    original_config, original_readers, original_views = sim.sensor_config, sim.contact_readers, sim.contact_views
    try:
        for config, expected in cases:
            sim.sensor_config = {"invalid": config}
            with pytest.raises(ValueError, match=expected):
                build_contacts(sim)
    finally:
        sim.sensor_config, sim.contact_readers, sim.contact_views = original_config, original_readers, original_views


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulator", choices=["mujoco", "mjwarp", "isaacsim"], required=True)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--device")
    parser.add_argument("--legacy-only", action="store_true")
    parser.add_argument("--result-file", type=Path, required=True)
    args = parser.parse_args()
    sim = _build(args.simulator, args.num_envs, args.device, configured_contacts=not args.legacy_only)
    try:
        if args.legacy_only:
            _assert_legacy_only(sim)
        else:
            _assert_contract(sim)
        if args.simulator == "isaacsim" and not args.legacy_only:
            _assert_isaac_setup_errors(sim)
        args.result_file.write_text("OK\n")
        print(f"Contact sensor contracts passed: {args.simulator}, {args.num_envs} environments.")
        return 0
    finally:
        sim.hooks.emit(Phase.CLOSE)


if __name__ == "__main__":
    run_and_hard_exit(main)
