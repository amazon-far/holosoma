"""UCA1 dataset regression: exact fixed topology, support and rendered relocation.

Loads the existing resolved scene and assets supplied with --scene-config. The
pytest wrapper skips explicitly when this external dataset is unavailable.
Detailed-mesh contact jitter is recorded separately from strict tray/toy settling.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

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
    configure_isaacsim6,
    launch_simulator,
    run_and_hard_exit,
    step,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--renderer", choices=["ovrtx", "isaac_rtx", "newton_warp"])
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--simulator", choices=["isaacsim6"], default="isaacsim6")
    parser.add_argument("--scene-config", type=Path, required=True)
    parser.add_argument("--result-file", type=Path, required=True)
    args = parser.parse_args()
    _robot_presets.register()
    _scene_presets.register()
    from holosoma.config_types.scene import ObjectPatternConfig, SceneConfig
    from holosoma.config_types.sensor import SensorMountConfig

    raw = json.loads(args.scene_config.read_text())
    scene = SceneConfig(**raw["scene"])
    # Remove positive-mass adaptation; preserve collision-authored cached room and all furniture/assets.
    room = dataclasses.replace(scene.scene_files["apartment"], object_configs={"*": ObjectPatternConfig(fixed=True)})
    objects = dict(scene.rigid_objects)
    objects["apartment_gaussian"] = dataclasses.replace(objects["apartment_gaussian"], physics=None)
    objects["marker"] = dataclasses.replace(
        _camera_presets.panel_target.rigid_objects["panel"], position=[4.74, -4.16, 1.4]
    )
    scene = dataclasses.replace(scene, rigid_objects=objects, scene_files={"apartment": room})
    config = build_run_sim_config("isaacsim6", "empty", "onelink-box", "terrain_locomotion_plane")
    robot = dataclasses.replace(
        config.robot, init_state=dataclasses.replace(config.robot.init_state, pos=[4.5, -2.45, 0.8])
    )
    config = dataclasses.replace(
        config,
        scene=scene,
        robot=robot,
        training=dataclasses.replace(config.training, num_envs=args.num_envs, headless=True),
    )
    config = configure_isaacsim6(
        config, argparse.Namespace(simulator="isaacsim6", physics_backend="newton", renderer=args.renderer or "ovrtx")
    )
    # Retain shared MJWarp capacity/settings used by native controls and the session.
    mujoco = config.simulator.config.sim.mujoco
    mujoco = dataclasses.replace(
        mujoco, warp=dataclasses.replace(mujoco.warp, njmax_per_env=4096, nconmax_per_env=512), iterations=100
    )
    config = dataclasses.replace(
        config,
        simulator=dataclasses.replace(
            config.simulator,
            config=dataclasses.replace(
                config.simulator.config, sim=dataclasses.replace(config.simulator.config.sim, mujoco=mujoco)
            ),
        ),
    )
    if args.renderer:
        camera = dataclasses.replace(
            _camera_presets.front_cam_depth["front_cam"],
            width=128,
            height=128,
            mount=SensorMountConfig(
                target_kind="world", position=[3.84, -4.16, 1.4], orientation=_camera_presets._LOOK_FORWARD_WXYZ
            ),
        )
        table_camera = dataclasses.replace(
            camera,
            # Offset from the red marker so the center rays actually see the tabletop.
            mount=SensorMountConfig(target_kind="world", position=[4.9, -4.16, 2.0], orientation=[1.0, 0.0, 0.0, 0.0]),
        )
        config = dataclasses.replace(config, sensor={"camera": camera, "table_camera": table_camera})
    sim, device, _app = launch_simulator(config, device="cuda:0")
    sim.set_headless(True)
    sim.setup()
    sim.setup_terrain()
    sim.load_assets()
    import torch

    num_envs = args.num_envs
    ids = torch.arange(num_envs, device=device)
    init = config.robot.init_state
    sim.create_envs(
        num_envs,
        torch.zeros(num_envs, 3, device=device),
        torch.tensor([*init.pos, *init.rot, *init.lin_vel, *init.ang_vel], device=device),
    )
    sim.prepare_sim()
    origins = sim.env_origins
    fixtures = [
        name for name in sim.scene.rigid_objects if getattr(sim.scene.rigid_objects[name], "is_world_fixed", False)
    ]
    assert_newton_fixed_objects(sim, fixtures)
    assert len(fixtures) == 21, (len(fixtures), fixtures)
    dynamic = [name for name, obj in scene.rigid_objects.items() if not obj.fixed]
    initial = {name: sim.get_actor_states([name], ids).clone() for name in sim.scene.rigid_objects}
    result: dict[str, Any] = {
        "renderer": args.renderer,
        "robot": "one-link",
        "num_envs": num_envs,
        "fixtures": fixtures,
        "samples": [],
    }

    def sample(tag: str) -> dict[str, torch.Tensor]:
        states = {name: sim.get_actor_states([name], ids).clone() for name in [*dynamic, "uca_dining_table", "robot"]}
        assert all(torch.isfinite(x).all() for x in states.values())
        result["samples"].append({"tag": tag, "states": {name: val.cpu().tolist() for name, val in states.items()}})
        args.result_file.with_suffix(".partial.json").write_text(json.dumps(result, indent=2) + "\n")
        return states

    def advance(count: int = 2000) -> None:
        step(sim, count)

    def capture(expected: list[float], table_expected: list[float]) -> tuple[torch.Tensor, torch.Tensor]:
        from tests.simulators.camera_follow_assert import _panel_median_depth

        before = snapshot_native_physics(sim)
        count = sim.sim.get_physics_step_count()
        sim.render_sensors()
        assert count == sim.sim.get_physics_step_count()
        assert all(torch.equal(x, y) for x, y in zip(before, snapshot_native_physics(sim), strict=True))
        rgb = sim.get_camera_data("camera", "rgb")
        depth = sim.get_camera_data("camera", "depth")
        distances = [_panel_median_depth(rgb[e], depth[e]) for e in range(num_envs)]
        assert all(x is not None and abs(x - y) < 0.045 for x, y in zip(distances, expected, strict=True)), distances
        table_depth = sim.get_camera_data("table_camera", "depth")
        table_depths = table_depth[:, 62:66, 62:66, 0].flatten(1).median(dim=1).values.cpu().tolist()
        result.setdefault("rendered_table_depths", []).append(table_depths)
        assert all(abs(x - y) < 0.05 for x, y in zip(table_depths, table_expected, strict=True)), (
            table_depths,
            table_expected,
        )
        return rgb, depth

    advance(4000)
    settled = sample("settled")
    # Check a ten-second window, not an endpoint or a noisy instantaneous
    # velocity on the detailed meshes. Record those velocities separately.
    for second in range(10):
        advance(200)
        stable = sample(f"support_window_{second + 1}")
        for name in dynamic:
            assert torch.allclose(stable[name][:, 2], initial[name][:, 2], atol=0.05), (name, stable[name])
            assert (stable[name][:, :3] - settled[name][:, :3]).norm(dim=-1).max() < 0.025, (
                name,
                settled[name],
                stable[name],
            )
    for name in dynamic:
        # Every prop must remain on its authored support. Four tray/toy props must
        # settle strictly; the detailed meshes have residual jitter also reproduced
        # with true fixed bodies in standalone CPU MuJoCo and MJWarp controls.
        if name in ["sheet_pan", "gumby", "office_sheet_pan", "fish"]:
            assert (stable[name][:, :3] - settled[name][:, :3]).norm(dim=-1).max() < 0.005, (
                name,
                settled[name],
                stable[name],
            )
            assert stable[name][:, 7:].norm(dim=-1).max() < 0.02, (name, stable[name])
        else:
            result.setdefault("residual_mesh_jitter", {})[name] = stable[name][:, 7:].cpu().tolist()
    for name in fixtures:
        assert torch.equal(sim.get_actor_states([name], ids), initial[name]), name
    assert torch.linalg.vector_norm(sim.contact_forces, dim=-1).max() > 0, "robot-floor contacts suppressed"
    assert torch.allclose(
        stable["robot"][:, 2] - origins[:, 2], torch.full((num_envs,), 0.05, device=device), atol=0.02
    )
    if args.renderer:
        old_rgb, old_depth = capture([0.89] * num_envs, [1.25] * num_envs)
        owned_rgb, owned_depth = old_rgb.clone(), old_depth.clone()
    chosen = ids[:1]
    others = ids[1:]
    table = initial["uca_dining_table"][chosen].clone()
    table[:, 1] += 10
    sim.set_static_body_pose(["uca_dining_table"], chosen, table[:, :7])
    marker = initial["marker"][chosen].clone()
    marker[:, 0] += 0.25
    sim.set_actor_states(["marker"], chosen, marker)
    advance()
    dropped = sample("table_removed")
    for name in ["sheet_pan", "gumby"]:
        assert dropped[name][0, 2] - origins[0, 2] < 0.1, (name, dropped[name])
        assert torch.allclose(dropped[name][others, :3], stable[name][others, :3], atol=0.005), (name, dropped[name])
    if args.renderer:
        capture([1.14] + [0.89] * (num_envs - 1), [2.0] + [1.25] * (num_envs - 1))
        assert torch.equal(old_rgb, owned_rgb) and torch.equal(old_depth, owned_depth)
    sim.set_static_body_pose(["uca_dining_table"], chosen, initial["uca_dining_table"][chosen, :7])
    sim.set_actor_states(["marker"], chosen, initial["marker"][chosen])
    for name in ["sheet_pan", "gumby"]:
        sim.set_actor_states([name], chosen, initial[name][chosen])
    advance()
    restored = sample("partial_reset")
    for name in ["sheet_pan", "gumby"]:
        assert torch.allclose(restored[name][:, :3], stable[name][:, :3], atol=0.006), (
            name,
            restored[name],
            stable[name],
        )
    if args.renderer:
        capture([0.89] * num_envs, [1.25] * num_envs)
    assert_newton_fixed_objects(sim, fixtures)
    assert_native_runtime(physics="newton", uses_kit=args.renderer == "isaac_rtx")
    args.result_file.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    sim.sim.stop()
    sim.sim.clear_instance()
    args.result_file.write_text("OK\n")
    print("PASS: full apartment zero-DOF fixtures, support and partial relocation/reset", flush=True)
    return 0


if __name__ == "__main__":
    run_and_hard_exit(main)
