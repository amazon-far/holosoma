"""Live engine/renderer matrix: independent physics, sparse properties and camera geometry."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

if sys.path and sys.path[0].endswith("simulators"):
    sys.path.pop(0)

from holosoma.config_types.simulator import IsaacLab3Config
from tests.simulators._native_runtime_assertions import assert_native_runtime
from tests.simulators._sim_harness import build_run_sim_config, launch_simulator, run_and_hard_exit, step


def main() -> int:
    started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulator", choices=["isaaclab3"], required=True)
    parser.add_argument("--physics", choices=["physx", "newton"], required=True)
    parser.add_argument("--renderer", choices=["isaac_rtx", "newton_warp", "ovrtx"], required=True)
    parser.add_argument("--result-file", required=True)
    args = parser.parse_args()

    from tests.simulators import _camera_presets

    config = build_run_sim_config(
        args.simulator,
        "panel-target",
        "g1-29dof",
        "terrain_locomotion_plane",
        sensors=_camera_presets.front_cam_depth,
    )
    native = IsaacLab3Config(physics_backend=args.physics, renderer=args.renderer)
    simulator = dataclasses.replace(
        config.simulator,
        config=dataclasses.replace(
            config.simulator.config,
            sim=dataclasses.replace(config.simulator.config.sim, isaaclab3=native),
        ),
    )
    config = dataclasses.replace(
        config,
        simulator=simulator,
        device="cuda:0",
        training=dataclasses.replace(config.training, num_envs=2),
    )
    sim, device, _app = launch_simulator(config, device="cuda:0")
    launched = time.perf_counter()
    sim.set_headless(True)
    sim.setup()
    sim.setup_terrain()
    sim.load_assets()

    import torch
    import warp as wp
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.utils.string import string_to_callable

    from holosoma.config_types.sensor import CameraSensorConfig
    from holosoma.simulator.isaacsim.events import (
        randomize_body_com,
        randomize_rigid_body_inertia,
        randomize_rigid_body_mass,
        randomize_rigid_body_material,
    )
    from holosoma.utils.sampler import STAGE_SETUP, DistributionSpec, TermSampler
    from tests.simulators.camera_geometry_assert import _check_geometry

    init = config.robot.init_state
    initial = torch.tensor(
        list(init.pos) + list(init.rot) + list(init.lin_vel) + list(init.ang_vel),
        device=device,
        dtype=torch.float32,
    )
    sim.create_envs(2, torch.zeros(2, 3, device=device), initial)
    sim.prepare_sim()
    ready = time.perf_counter()
    sim.install_plugins()
    manager = string_to_callable(sim.sim.cfg.physics.class_type)
    assert ("newton" in manager.__module__) == (args.physics == "newton"), manager
    if args.renderer != "isaac_rtx":
        from importlib.machinery import PathFinder

        import isaacsim
        from isaaclab.utils.version import has_kit, standalone_importers_available

        assert not has_kit(), "Native renderers must run without the Kit runtime"
        assert standalone_importers_available(), "Standalone importers must be discoverable by the SDK"
        namespace = PathFinder.find_spec("isaacsim")
        assert namespace is not None and namespace.origin is None, "Kit's regular root must remain hidden"
        assert type(isaacsim.__spec__.loader).__name__ == "NamespaceLoader", "Use ordinary Python namespace discovery"
        if args.physics == "physx":
            assert manager.__module__.startswith("isaaclab_ov."), manager
            scene_prim = sim.sim.stage.GetPrimAtPath(sim.sim.cfg.physics_prim_path)
            assert scene_prim.GetAttribute("physxScene:solverType").Get() == "TGS"
            assert (
                scene_prim.GetAttribute("physxScene:maxPositionIterationCount").Get()
                == config.simulator.config.sim.physx.num_position_iterations
            )
    camera = sim.tiled_cameras["front_cam"]
    assert camera.cfg.renderer_cfg.renderer_type == args.renderer, camera.cfg.renderer_cfg
    robot = sim._robot
    spawn_joints = sim.dof_pos.clone()
    if args.physics == "newton":
        model = manager.get_model()
        ground_ids = [index for index, label in enumerate(model.shape_label) if label.startswith("/World/ground/")]
        assert ground_ids, "Newton must import the shared ground independently of the environment subtree"
        worlds = wp.to_torch(model.shape_world)
        assert bool((worlds[ground_ids] == -1).all()), worlds[ground_ids]

    # Read the native engine directly. The production adapter never supplies the oracle.
    def masses():
        if args.physics == "newton":
            return wp.to_torch(robot.root_view.get_attribute("body_mass", manager.get_model())[:, 0]).clone()
        if args.renderer != "isaac_rtx":
            return wp.to_torch(robot.root_view.get_attribute("articulation_body_mass")).clone()
        return wp.to_torch(robot.root_view.get_masses()).clone()

    def materials():
        if args.physics == "newton":
            model = manager.get_model()
            native = torch.stack(
                (wp.to_torch(model.shape_material_mu), wp.to_torch(model.shape_material_restitution)), dim=-1
            )
            robot_shapes = torch.tensor(
                ["/Robot/" in label for label in model.shape_label], device=native.device, dtype=torch.bool
            )
            worlds = wp.to_torch(model.shape_world)
            # Labels/worlds are independent of the adapter's articulation ownership map.
            return torch.stack([native[robot_shapes & (worlds == row)] for row in range(2)]).clone()
        if args.renderer != "isaac_rtx":
            return wp.to_torch(robot.root_view.get_attribute("articulation_shape_friction_and_restitution")).clone()
        return wp.to_torch(robot.root_view.get_material_properties()).clone()

    def body_property(name):
        if args.physics == "newton":
            return wp.to_torch(
                robot.root_view.get_attribute(
                    {"coms": "body_com", "inertias": "body_inertia"}[name], manager.get_model()
                )[:, 0]
            ).clone()
        if args.renderer != "isaac_rtx":
            attribute = {"coms": "articulation_body_com_pose", "inertias": "articulation_body_inertia"}[name]
            return wp.to_torch(robot.root_view.get_attribute(attribute)).clone()
        return wp.to_torch(getattr(robot.root_view, f"get_{name}")()).clone()

    before_mass = masses()
    randomize_rigid_body_mass(
        sim,
        torch.tensor([1]),
        SceneEntityCfg("robot", body_ids=slice(None)),
        mass=[1.25, 1.25],
        operation="scale",
        recompute_inertia=False,
        sampler=TermSampler.bind(base_seed=31, term_name="matrix-mass", stage=STAGE_SETUP),
    )
    expected_mass = before_mass.clone()
    expected_mass[1] *= 1.25
    torch.testing.assert_close(masses(), expected_mass, rtol=1e-5, atol=1e-6)

    # Resolve the independent engine order by name, rather than the adapter's mapping.
    body_name = "pelvis"
    raw_body = robot.backend_body_names.index(body_name)
    public_body = robot.body_names.index(body_name)
    target = SceneEntityCfg("robot", body_ids=[public_body])
    before_com = body_property("coms")[..., :3]
    randomize_body_com(
        sim,
        torch.tensor([1]),
        target,
        specs=[DistributionSpec.parse([x, x]) for x in (0.01, -0.02, 0.03)],
        operation="add",
        sampler=TermSampler.bind(base_seed=31, term_name="matrix-com", stage=STAGE_SETUP),
    )
    expected_com = before_com.clone()
    expected_com[1, raw_body] += torch.tensor([0.01, -0.02, 0.03], device=expected_com.device)
    torch.testing.assert_close(body_property("coms")[..., :3], expected_com, rtol=0, atol=1e-6)
    before_inertia = body_property("inertias")
    randomize_rigid_body_inertia(
        sim,
        torch.tensor([1]),
        target,
        specs=[DistributionSpec.parse([1.1, 1.1])] * 6,
        operation="scale",
        sampler=TermSampler.bind(base_seed=31, term_name="matrix-inertia", stage=STAGE_SETUP),
    )
    expected_inertia = before_inertia.clone()
    expected_inertia[1, raw_body] *= 1.1
    torch.testing.assert_close(body_property("inertias"), expected_inertia, rtol=1e-5, atol=1e-6)
    if args.physics == "newton":
        model = manager.get_model()
        inverse = wp.to_torch(robot.root_view.get_attribute("body_inv_inertia", model)[:, 0])
        torch.testing.assert_close(
            inverse[1, raw_body],
            torch.linalg.inv(expected_inertia[1, raw_body]),
            rtol=1e-5,
            atol=1e-5,
        )

    before_material = materials()
    global_materials = None
    selected_shapes = None
    if args.physics == "newton":
        model = manager.get_model()
        global_materials = torch.stack(
            (wp.to_torch(model.shape_material_mu), wp.to_torch(model.shape_material_restitution)), dim=-1
        ).clone()
        robot_shapes = torch.tensor(
            ["/Robot/" in label for label in model.shape_label],
            device=global_materials.device,
            dtype=torch.bool,
        )
        # Native replication may retain template labels in every world.
        selected_shapes = torch.nonzero(robot_shapes & (wp.to_torch(model.shape_world) == 1), as_tuple=True)[0]
        assert selected_shapes.numel() == before_material.shape[1] > 0
        global_materials[selected_shapes, 0] = 0.37
        global_materials[selected_shapes, 1] = 0.23
    randomize_rigid_body_material(
        sim,
        torch.tensor([1]),
        SceneEntityCfg("robot"),
        static_friction=[0.37, 0.37],
        dynamic_friction=None,
        restitution=[0.23, 0.23],
        num_buckets=None,
        sampler=TermSampler.bind(base_seed=31, term_name="matrix-material", stage=STAGE_SETUP),
    )
    expected_material = before_material.clone()
    expected_material[1, :, 0] = 0.37
    expected_material[1, :, -1] = 0.23
    if args.physics == "physx":
        # The shared writer preserves PhysX's dynamic <= static friction rule.
        expected_material[1, :, 1].clamp_(max=0.37)
    torch.testing.assert_close(materials(), expected_material, rtol=0, atol=1e-6)
    if global_materials is not None:
        actual_global = torch.stack(
            (wp.to_torch(model.shape_material_mu), wp.to_torch(model.shape_material_restitution)), dim=-1
        )
        changed = torch.nonzero((actual_global - global_materials).abs().amax(dim=-1) > 1e-6, as_tuple=True)[0]
        if changed.numel():
            worlds = wp.to_torch(model.shape_world)
            print(
                "UNEXPECTED_MATERIAL_SHAPES",
                [
                    (i, model.shape_label[i], int(worlds[i]), actual_global[i].tolist(), global_materials[i].tolist())
                    for i in changed.tolist()
                ],
                flush=True,
            )
        torch.testing.assert_close(actual_global, global_materials, rtol=0, atol=1e-6)

    # Forward propagates an authored root state without integrating gravity or advancing time.
    ids = torch.arange(2, device=device)
    pose = sim.get_actor_states(["robot"], ids).clone()
    pose[:, :3] = sim.env_origins + torch.tensor(list(init.pos), device=device)
    pose[:, 2] += 0.2
    pose[:, 7:] = 0.0
    sim.set_actor_states(["robot"], ids, pose)
    count = sim.sim.get_physics_step_count()
    time_before = sim.backend.physics_time(sim.sim)
    sim.forward_kinematics()
    assert sim.sim.get_physics_step_count() == count
    assert sim.backend.physics_time(sim.sim) == time_before
    torch.testing.assert_close(sim.get_actor_states(["robot"], ids), pose, rtol=0, atol=2e-5)
    start_z = sim.get_actor_states(["robot"], ids)[:, 2].clone()
    step(sim, 10)
    falling = sim.get_actor_states(["robot"], ids)
    assert bool((falling[:, 2] < start_z - 0.002).all()), falling
    assert bool((falling[:, 9] < -0.1).all()), falling
    assert sim.sim.get_physics_step_count() == count + 10
    assert abs(sim.backend.physics_time(sim.sim) - time_before - 10 * sim.sim_dt) < 1e-6

    # Restore the camera mount before inspecting a known panel, after real physical stepping.
    for _ in range(5):
        pose[:, :3] = sim.env_origins + torch.tensor(list(init.pos), device=device)
        pose[:, 7:] = 0
        sim.set_actor_states(["robot"], ids, pose)
        sim.dof_pos[ids] = spawn_joints
        sim.dof_vel[ids] = 0
        sim.forward_kinematics()
        sim.render_sensors()
    rgb = sim.get_camera_data("front_cam", "rgb")
    depth = sim.get_camera_data("front_cam", "depth")
    cfg = config.sensor["front_cam"]
    assert isinstance(cfg, CameraSensorConfig)
    assert tuple(rgb.shape) == (2, cfg.height, cfg.width, 3) and rgb.dtype == torch.uint8
    assert tuple(depth.shape) == (2, cfg.height, cfg.width, 1)
    distance = _camera_presets._PANEL_DISTANCE - cfg.mount.position[0] - 0.01
    for row in range(2):
        failures = _check_geometry(rgb[row], cfg, distance, _camera_presets._PANEL_HALF_SIZE, f"env{row}")
        assert not failures, failures
        center = float(depth[row, cfg.height // 2, cfg.width // 2, 0])
        assert abs(center - distance) < 0.035, (row, center, distance)

    # Sparse model writes remain effective after stepping; reading them must not advance physics.
    torch.testing.assert_close(masses(), expected_mass, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(materials(), expected_material, rtol=0, atol=1e-6)
    torch.testing.assert_close(body_property("coms")[..., :3], expected_com, rtol=0, atol=1e-6)
    torch.testing.assert_close(body_property("inertias"), expected_inertia, rtol=1e-5, atol=1e-6)
    assert torch.isfinite(sim._robot_contact_forces()).all()

    # Both torque signs must move the named joint and clip at its configured model limit.
    joint_name = "left_elbow_joint"
    configured_joint = config.robot.dof_names.index(joint_name)
    native_joint = robot.joint_names.index(joint_name)
    limit = config.robot.dof_effort_limit_list[configured_joint]
    for sign in (1.0, -1.0):
        pose[:, :3] = sim.env_origins + torch.tensor(list(init.pos), device=device)
        pose[:, 2] += 1.0
        pose[:, 7:] = 0.0
        sim.set_actor_states(["robot"], ids, pose)
        sim.dof_pos[ids] = spawn_joints
        sim.dof_vel[ids] = 0.0
        sim.forward_kinematics()
        command = torch.zeros((2, sim.num_dof), device=device)
        command[:, configured_joint] = sign * 500.0
        sim.apply_torques_at_dof(command)
        step(sim, 3)
        torch.testing.assert_close(
            sim.backend.applied_effort(robot)[:, native_joint],
            torch.full((2,), sign * limit, device=device),
            rtol=0,
            atol=1e-5,
        )
        velocity = sim.dof_vel[:, configured_joint]
        assert bool((sign * velocity > 0.01).all()), velocity

    # A sliding, falling robot must generate real normal and tangential reactions.
    pose[:, :3] = sim.env_origins + torch.tensor(list(init.pos), device=device)
    pose[:, 7:] = 0.0
    pose[:, 7] = 0.6
    sim.set_actor_states(["robot"], ids, pose)
    sim.dof_pos[ids] = spawn_joints
    sim.dof_vel[ids] = 0.0
    sim.forward_kinematics()
    sim.apply_torques_at_dof(torch.zeros((2, sim.num_dof), device=device))
    peak_normal = torch.zeros(2, device=device)
    peak_tangential = torch.zeros(2, device=device)
    total_config = dataclasses.replace(
        config.simulator.config.sim, isaaclab3=dataclasses.replace(native, contact_force_mode="total")
    )
    # Observe the contact window: a bouncing robot can be airborne at the final tick.
    for _ in range(120):
        step(sim, 1)
        normal = sim.contact_sensor.data.net_normal_forces_w.torch.clone()
        total = sim.contact_sensor.data.net_forces_w.torch.clone()
        assert torch.isfinite(normal).all() and torch.isfinite(total).all()
        peak_normal = torch.maximum(peak_normal, normal.abs().amax(dim=(1, 2)))
        peak_tangential = torch.maximum(peak_tangential, (total - normal).abs().amax(dim=(1, 2)))
        torch.testing.assert_close(sim.backend.contact_forces(sim.contact_sensor, config.simulator.config.sim), normal)
        torch.testing.assert_close(sim.backend.contact_forces(sim.contact_sensor, total_config), total)
    assert bool((peak_normal > 1.0).all()), (peak_normal, sim.get_actor_states(["robot"], ids))
    assert bool((peak_tangential > 0.1).all()), peak_tangential
    dependencies = assert_native_runtime(physics=args.physics, uses_kit=args.renderer == "isaac_rtx")
    Path(args.result_file).with_suffix(".startup.json").write_text(
        json.dumps(
            {
                "physics": args.physics,
                "renderer": args.renderer,
                "launch_seconds": launched - started,
                "ready_seconds": ready - started,
                **dependencies,
            },
            indent=2,
        )
        + "\n"
    )
    if args.renderer != "isaac_rtx":
        from isaaclab.sim import SimulationContext

        from holosoma.utils.sim_utils import close_simulation_resources

        close_simulation_resources(sim, _app)
        assert SimulationContext.instance() is None, "Kit-less native resources remained open"
        assert not _app.is_running(), "closed Kit-less runtime still reports running"
    Path(args.result_file).write_text("OK")
    print(
        f"PASS: {args.physics} / {args.renderer}: independent physics, sparse mass/COM/inertia/material DR, "
        f"FK, RGB/depth, both-sign clipped actuation, contacts {peak_normal.tolist()}/{peak_tangential.tolist()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    if "ovrtx" in sys.argv or "newton_warp" in sys.argv:
        # Kit-less runs must support ordinary interpreter teardown.
        sys.exit(main())
    else:
        run_and_hard_exit(main)
