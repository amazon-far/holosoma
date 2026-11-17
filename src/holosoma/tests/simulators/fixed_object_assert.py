"""Real Newton fixture topology, contact, relocation and reset regression.

Reuses the support, scene-file, robot and camera fixtures. Fixed objects must
have zero dynamic DOFs. Downstream support/fall outcomes distinguish a real
physics relocation from an updated state cache or a rendered-only transform.
An authored kinematic body exercises the separate pose-driven SDK path.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
import tempfile
from pathlib import Path

if sys.path and sys.path[0].endswith("simulators"):
    sys.path.pop(0)

from tests.simulators import _camera_presets, _robot_presets, _scene_presets
from tests.simulators._native_runtime_assertions import (
    assert_native_runtime,
    assert_newton_fixed_objects,
    snapshot_native_physics,
)
from tests.simulators._sim_harness import (
    build_run_sim_config,
    configure_isaaclab3,
    launch_simulator,
    run_and_hard_exit,
    step,
    steps_for_seconds,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulator", choices=["isaaclab3"], default="isaaclab3")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--renderer", choices=["isaac_rtx", "ovrtx", "newton_warp"])
    parser.add_argument("--result-file", required=True)
    args = parser.parse_args()
    _robot_presets.register()
    _scene_presets.register()
    from holosoma.config_types.scene import ObjectPatternConfig, PhysicsConfig, RigidObjectConfig, SceneFileConfig
    from holosoma.config_types.sensor import SensorMountConfig

    config = build_run_sim_config("isaaclab3", "static-support", "onelink-box", "terrain_locomotion_plane")
    scene_file = _scene_presets.multibody.scene_files["scene"]
    temporary_scene = tempfile.TemporaryDirectory()
    static_path = str(Path(temporary_scene.name) / "native-static.usda")
    objects = {
        **config.scene.rigid_objects,
        **_camera_presets.panel_target.rigid_objects,
        # Zero mass is valid for a fixed fixture; no placeholder dynamic inertia.
        "massless": RigidObjectConfig(
            usd_file=_camera_presets._SMALL_BOX_USD,
            position=[4.0, 0.0, 0.3],
            fixed=True,
            physics=PhysicsConfig(mass=0.0),
        ),
        "kin_prop": dataclasses.replace(config.scene.rigid_objects["restbox"], position=[3.5, 0.0, 0.45]),
        "file_prop": dataclasses.replace(config.scene.rigid_objects["restbox"], position=[2.5, 0.0, 0.45]),
        "disabled_prop": dataclasses.replace(config.scene.rigid_objects["restbox"], position=[4.5, 0.0, 0.45]),
        "world_prop": dataclasses.replace(config.scene.rigid_objects["restbox"], position=[5.0, 0.0, 0.45]),
    }
    config = dataclasses.replace(
        config,
        scene=dataclasses.replace(
            config.scene,
            rigid_objects=objects,
            scene_files={
                "native": SceneFileConfig(usd_path=static_path, position=[4.5, 0.0, 0.3]),
                "fixed_file": dataclasses.replace(
                    scene_file,
                    position=[2.0, 0.0, 0.3],
                    include_patterns=["static_post"],
                    object_configs={"*": ObjectPatternConfig(fixed=True)},
                ),
                "kinematic_file": dataclasses.replace(
                    scene_file,
                    position=[3.0, 0.0, 0.3],
                    include_patterns=["static_post"],
                ),
            },
        ),
        training=dataclasses.replace(config.training, num_envs=args.num_envs, headless=True),
    )
    if args.renderer:
        camera = dataclasses.replace(
            _camera_presets.front_cam_depth["front_cam"],
            width=128,
            height=128,
            mount=SensorMountConfig(
                target_kind="world",
                position=[0.1, 0.0, _camera_presets._BASE_Z],
                orientation=_camera_presets._LOOK_FORWARD_WXYZ,
            ),
        )
        config = dataclasses.replace(config, sensor={"camera": camera})
    config = configure_isaaclab3(
        config, argparse.Namespace(simulator="isaaclab3", physics_backend="newton", renderer=args.renderer or "ovrtx")
    )

    def author_native_scene() -> None:
        # Activation selects the USD build/schemas before pxr imports. Author
        # the temporary reference before the simulator constructor loads it.
        from pxr import Usd, UsdGeom, UsdPhysics

        from holosoma.utils.path import resolve_asset_path

        stage = Usd.Stage.CreateNew(static_path)
        root = stage.DefinePrim("/multibody", "Xform")
        assert scene_file.usd_path is not None
        root.GetReferences().AddReference(resolve_asset_path(scene_file.usd_path, scene_file.asset_root))
        stage.SetDefaultPrim(root)
        UsdGeom.SetStageUpAxis(stage, "Z")
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        disabled = stage.GetPrimAtPath("/multibody/free_box")
        UsdPhysics.RigidBodyAPI(disabled).GetRigidBodyEnabledAttr().Set(False)
        anchored = stage.GetPrimAtPath("/multibody/static_post")
        # World attachment takes precedence over the pose-settable kinematic flag.
        UsdPhysics.FixedJoint.Define(stage, "/multibody/attachment").CreateBody1Rel().SetTargets([anchored.GetPath()])
        stage.GetRootLayer().Save()

    sim, device, _app = launch_simulator(config, device="cuda:0", before_construct=author_native_scene)
    sim.set_headless(True)
    sim.setup()
    sim.setup_terrain()
    sim.load_assets()
    import torch

    n = args.num_envs
    ids = torch.arange(n, device=device)
    init = config.robot.init_state
    sim.create_envs(
        n,
        torch.zeros(n, 3, device=device),
        torch.tensor([*init.pos, *init.rot, *init.lin_vel, *init.ang_vel], device=device),
    )
    sim.prepare_sim()
    origins = sim.env_origins
    fixed = ["post", "panel", "massless", "fixed_file_static_post", "native_free_box", "native_static_post"]
    assert_newton_fixed_objects(sim, fixed)
    from newton import BodyFlags, JointType

    model = sim.sim.physics_manager.get_model()
    kinematic = sim.scene.rigid_objects["kinematic_file_static_post"]
    assert not getattr(kinematic, "is_world_fixed", False)
    for body in [i for i, label in enumerate(model.body_label) if label.endswith("/kinematic_file_static_post")]:
        joint = next(i for i, child in enumerate(model.joint_child.numpy()) if child == body)
        assert model.joint_type.numpy()[joint] == JointType.FREE
        assert model.body_flags.numpy()[body] & int(BodyFlags.KINEMATIC)

    def local_states(name):
        states = sim.get_actor_states([name], ids).clone()
        states[:, :3] -= origins
        return states

    def settle():
        step(sim, steps_for_seconds(sim, 1.5))

    settle()
    for name in ["restbox", "file_prop", "kin_prop", "disabled_prop", "world_prop"]:
        states = local_states(name)
        assert torch.allclose(states[:, 2], torch.full_like(states[:, 2], 0.4), atol=0.02), (name, states)
        assert torch.linalg.vector_norm(states[:, 7:], dim=-1).max() < 0.01, (name, states)
    robot = sim.get_actor_states(["robot"], ids)
    assert torch.allclose(robot[:, 2] - origins[:, 2], torch.full((n,), 0.05, device=device), atol=0.02)
    assert torch.linalg.vector_norm(sim.contact_forces, dim=-1).amax() > 0, "robot-floor contacts disabled"
    scene_snapshot = sim.scene.get_state(is_relative=True)

    original = {name: sim.get_actor_states([name], ids).clone() for name in fixed}
    selected = torch.arange(0, n, 2, device=device)
    unchanged = torch.arange(1, n, 2, device=device)
    # A robot resting on furniture verifies that robot-fixture contacts remain active.
    robot[selected, :3] = original["post"][selected, :3] + torch.tensor([0.0, 0.0, 0.15], device=device)
    robot[selected, 7:] = 0
    prop = sim.get_actor_states(["restbox"], selected).clone()
    prop[:, 1] += 0.5
    prop[:, 7:] = 0
    sim.set_actor_states(["restbox"], selected, prop)
    sim.set_actor_states(["robot"], selected, robot[selected])
    settle()
    assert torch.allclose(
        local_states("robot")[selected, 2], torch.full_like(selected, 0.4, dtype=torch.float32), atol=0.02
    )
    assert torch.linalg.vector_norm(sim.contact_forces[selected], dim=-1).amax() > 0, (
        "robot-furniture contacts disabled"
    )

    if args.renderer:
        from tests.simulators.camera_follow_assert import _panel_median_depth, _red_mask

        def panel_extents(image):
            mask = _red_mask(image)
            return torch.stack([mask.any(dim=0).sum(), mask.any(dim=1).sum()]).float()

        def capture():
            before = snapshot_native_physics(sim)
            count = sim.sim.get_physics_step_count()
            sim.render_sensors()
            assert count == sim.sim.get_physics_step_count()
            for old, new in zip(before, snapshot_native_physics(sim), strict=True):
                assert torch.equal(old, new), "render advanced or rewrote physics"
            rgb = sim.get_camera_data("camera", "rgb")
            depth = sim.get_camera_data("camera", "depth")
            return rgb, depth, [_panel_median_depth(rgb[e], depth[e]) for e in range(n)]

        old_rgb, old_depth, before = capture()
        owned_rgb, owned_depth = old_rgb.clone(), old_depth.clone()
        old_extents = [panel_extents(old_rgb[e]) for e in range(n)]
        for distance in before:
            assert distance is not None and abs(distance - 0.89) < 0.04, before

    # Remove supports in a strided subset, using each environment's own pose.
    for name in ["post", "fixed_file_static_post", "kinematic_file_static_post"]:
        states = sim.get_actor_states([name], selected).clone()
        states[:, 1] += 0.5
        sim.set_static_body_pose([name], selected, states[:, :7])
    panel = original["panel"][selected].clone()
    panel[:, 0] += 0.25
    panel[:, 3:7] = torch.tensor([0.0, 0.0, math.sin(math.pi / 8), math.cos(math.pi / 8)], device=device)
    sim.set_actor_states(["panel"], selected, panel)
    settle()
    assert torch.allclose(
        local_states("robot")[selected, 2], torch.full_like(selected, 0.05, dtype=torch.float32), atol=0.02
    )
    for name in ["file_prop", "kin_prop"]:
        states = local_states(name)
        assert torch.allclose(states[selected, 2], torch.full_like(selected, 0.05, dtype=torch.float32), atol=0.02), (
            name
        )
        assert torch.allclose(states[unchanged, 2], torch.full_like(unchanged, 0.4, dtype=torch.float32), atol=0.02), (
            name
        )
    for name in fixed:
        assert torch.equal(sim.get_actor_states([name], unchanged), original[name][unchanged]), name
    assert_newton_fixed_objects(sim, fixed)
    if args.renderer:
        new_rgb, _, after = capture()
        for e, distance in enumerate(after):
            expected = 1.14 if e % 2 == 0 else 0.89
            assert distance is not None and abs(distance - expected) < 0.04, (e, distance, expected)
        # Depth must reflect relocation immediately. The default temporal RGB
        # reconstruction can retain the previous silhouette on the first
        # submission; require the rotated geometry after two further captures,
        # at the same physics state and with unchanged renderer quality.
        for _ in range(2):
            new_rgb, _, converged_depths = capture()
            for e, distance in enumerate(converged_depths):
                expected = 1.14 if e % 2 == 0 else 0.89
                assert distance is not None and abs(distance - expected) < 0.04, (e, distance, expected)
        for e in range(n):
            if e % 2 == 0:
                # Translation alone scales both extents by ~0.78; yawing
                # narrows one further to ~0.58. The mount's roll determines
                # which image axis it is. Check geometry, not quaternion readback.
                ratio = (panel_extents(new_rgb[e]) / old_extents[e]).min().item()
                assert 0.48 < ratio < 0.66, (e, ratio)
        assert torch.equal(old_rgb, owned_rgb) and torch.equal(old_depth, owned_depth), "retained image buffer changed"

    # Partial reset restores fixtures and falling props only in selected worlds.
    for name in fixed:
        sim.set_actor_states([name], selected, original[name][selected])
    for name, x in [("restbox", 1.5), ("file_prop", 2.5)]:
        states = sim.get_actor_states([name], selected).clone()
        states[:, :3] = origins[selected] + torch.tensor([x, 0.0, 0.45], device=device)
        states[:, 7:] = 0
        sim.set_actor_states([name], selected, states)
    settle()
    for name in ["restbox", "file_prop"]:
        assert torch.allclose(
            local_states(name)[selected, 2], torch.full_like(selected, 0.4, dtype=torch.float32), atol=0.02
        ), name
    assert_newton_fixed_objects(sim, fixed)
    if args.renderer:
        _, _, reset_depths = capture()
        assert all(distance is not None and abs(distance - 0.89) < 0.04 for distance in reset_depths), reset_depths
    # Exercise the SDK scene state contract, as well as Holosoma actor writes.
    selected_snapshot = {
        family: {name: {key: value[selected].clone() for key, value in state.items()} for name, state in actors.items()}
        for family, actors in scene_snapshot.items()
    }
    count = sim.sim.get_physics_step_count()
    sim.scene.reset_to(selected_snapshot, env_ids=selected, is_relative=True)
    sim.write_state_updates()
    assert count == sim.sim.get_physics_step_count(), "SDK state restoration advanced physics"
    settle()
    for name in ["restbox", "file_prop", "kin_prop", "disabled_prop", "world_prop"]:
        assert torch.allclose(
            local_states(name)[selected, 2], torch.full_like(selected, 0.4, dtype=torch.float32), atol=0.02
        ), name
    assert_newton_fixed_objects(sim, fixed)
    if not args.renderer:
        # AssetBase's normal stop/ready lifecycle must rebind recreated buffers.
        previous_model = model
        sim.sim.reset()
        assert sim.sim.physics_manager.get_model() is not previous_model
        assert_newton_fixed_objects(sim, fixed)
    assert_native_runtime(physics="newton", uses_kit=bool(args.renderer == "isaac_rtx"))
    # Use the shared harness lifecycle: invalidate physics assets before the
    # renderer registry closes. Launcher/Kit interpreter shutdown is handled
    # by run_and_hard_exit, as in the existing behavioral harnesses.
    assets = [sim.scene.rigid_objects[name] for name in fixed]
    sim.sim.stop()
    sim.sim.clear_instance()
    assert all(not asset.is_initialized for asset in assets)
    if args.renderer:
        assert torch.equal(old_rgb, owned_rgb) and torch.equal(old_depth, owned_depth), "teardown changed output"
    Path(args.result_file).write_text("OK\n")
    print("PASS: fixed topology, support, robot contacts, kinematic/dynamic behavior, partial reset and relocation")
    temporary_scene.cleanup()
    return 0


if __name__ == "__main__":
    run_and_hard_exit(main)
