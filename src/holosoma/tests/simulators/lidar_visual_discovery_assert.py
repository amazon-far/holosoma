# Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""Live Isaac Sim LiDAR visual-target discovery regression.

The fixture puts a guide-purpose CollisionAPI proxy in front of a visible, collision-enabled
surface on the same ray. Assertions use public LiDAR outputs so the test fails if default
discovery targets collision-only geometry, drops visual scene/robot geometry, omits the ground,
or rewrites an explicit target path.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from importlib.util import find_spec
from pathlib import Path

# Avoid shadowing the installed isaacsim package with tests/simulators/isaacsim.
if sys.path and sys.path[0].endswith("simulators"):
    sys.path.pop(0)
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from holosoma.config_types.scene import PhysicsConfig, RigidObjectConfig  # noqa: E402
from holosoma.simulator.base_simulator.hooks import Phase  # noqa: E402
from holosoma.utils.sim_utils import setup_simulation_environment  # noqa: E402
from tests.simulators import _camera_presets, _lidar_presets  # noqa: E402, F401
from tests.simulators._sim_harness import build_run_sim_config, run_and_hard_exit  # noqa: E402


def _check_exact_downward_return(sim, name: str, expected_ranges) -> list[str]:
    """Compare public range and XYZ buffers with one exact downward intersection."""
    import torch

    failures = []
    ranges = sim.get_lidar_data(name, "ranges")
    points = sim.get_lidar_data(name, "points")
    direction = torch.tensor([0.0, 0.0, -1.0], dtype=ranges.dtype, device=ranges.device)
    expected_points = direction.view(1, 1, 3) * expected_ranges.unsqueeze(-1)
    if not torch.allclose(ranges, expected_ranges, atol=2e-3, rtol=2e-3):
        failures.append(
            f"{name}: actual ranges {ranges[:, 0].tolist()} do not match "
            f"authored ranges {expected_ranges[:, 0].tolist()}"
        )
    if not torch.allclose(points, expected_points, atol=2e-3, rtol=2e-3):
        failures.append(f"{name}: public XYZ does not match the authored surface intersection")
    return failures


def _check_native_target_contract(sim) -> list[str]:
    """Supplement the return oracles with exact public-config target selection checks."""
    failures = []
    visual_path = "/World/envs/env_.*/visual_collision_target/Visuals/visible"
    proxy_path = "/World/envs/env_.*/visual_collision_target/Collisions/proxy"

    default_targets = {target.prim_expr for target in sim.scene.sensors["visual_target"].cfg.mesh_prim_paths}
    if visual_path not in default_targets:
        failures.append("visual_target: default discovery omitted the visible scene-object surface")
    if proxy_path in default_targets:
        failures.append("visual_target: default discovery included the collision-only proxy")
    if not any(path.startswith("/World/envs/env_.*/Robot/") for path in default_targets):
        failures.append("visual_target: default discovery omitted robot visual geometry")
    if not any(path == "/World/ground" or path.startswith("/World/ground/") for path in default_targets):
        failures.append("visual_target: default discovery omitted ground visual geometry")

    explicit_targets = {
        target.prim_expr for target in sim.scene.sensors["explicit_collision_target"].cfg.mesh_prim_paths
    }
    if explicit_targets != {proxy_path}:
        failures.append(
            f"explicit_collision_target: caller-provided mesh path was changed (actual={sorted(explicit_targets)})"
        )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=3)
    parser.add_argument("--result-file", default=None)
    args = parser.parse_args()

    isaaclab_spec = find_spec("isaaclab")
    isaaclab_roots = () if isaaclab_spec is None else (isaaclab_spec.submodule_search_locations or ())
    if not any(
        (Path(root) / "sensors" / "ray_caster" / "multi_mesh_ray_caster.py").is_file() for root in isaaclab_roots
    ):
        print("SKIP: Isaac Sim LiDAR requires IsaacLab 2.3.2 or newer")
        return 77

    sensors = {
        "visual_target": _lidar_presets.isaacsim_visual_target_lidar,
        "explicit_collision_target": _lidar_presets.isaacsim_explicit_collision_target_lidar,
        "ground_visual": _lidar_presets.isaacsim_ground_visual_lidar,
        "robot_visual": _lidar_presets.isaacsim_robot_visual_lidar,
    }
    config = build_run_sim_config(
        "isaacsim",
        "camera-target",
        "g1-29dof",
        "terrain_locomotion_plane",
        sensors=sensors,
    )
    visual_collision_asset = str(Path(__file__).with_name("data") / "lidar_visual_collision_target.usda")
    scene_objects = dict(config.scene.rigid_objects)
    scene_objects["visual_collision_target"] = RigidObjectConfig(
        usd_file=visual_collision_asset,
        position=_lidar_presets.VISUAL_COLLISION_TARGET_POSITION,
        fixed=True,
        physics=PhysicsConfig(),
    )
    config = dataclasses.replace(
        config,
        device="cuda:0",
        scene=dataclasses.replace(config.scene, rigid_objects=scene_objects),
        training=dataclasses.replace(config.training, num_envs=args.num_envs),
    )

    env, device, _app = setup_simulation_environment(config, device="cuda:0")
    sim = env.sim
    sim.set_headless(True)
    sim.setup()
    sim.setup_terrain()
    sim.load_assets()

    import torch

    env_origins = torch.zeros(args.num_envs, 3, device=device)
    env_origins[:, 0] = torch.arange(args.num_envs, device=device, dtype=torch.float32) * 5.0
    env_origins[:, 2] = torch.arange(args.num_envs, device=device, dtype=torch.float32) * 0.25
    init = config.robot.init_state
    base_init = torch.tensor(
        list(init.pos) + list(init.rot) + list(init.lin_vel) + list(init.ang_vel),
        device=device,
        dtype=torch.float32,
    )
    sim.create_envs(args.num_envs, env_origins, base_init)
    sim.prepare_sim()
    sim.hooks.emit(Phase.FRAME_END)

    failures = _check_native_target_contract(sim)
    visual_expected = torch.full(
        (args.num_envs, 1),
        _lidar_presets.VISUAL_TARGET_EXPECTED_RANGE,
        dtype=torch.float32,
        device=device,
    )
    explicit_expected = torch.full(
        (args.num_envs, 1),
        _lidar_presets.EXPLICIT_COLLISION_EXPECTED_RANGE,
        dtype=torch.float32,
        device=device,
    )
    failures += _check_exact_downward_return(sim, "visual_target", visual_expected)
    failures += _check_exact_downward_return(sim, "explicit_collision_target", explicit_expected)

    visual_ranges = sim.get_lidar_data("visual_target", "ranges")
    explicit_ranges = sim.get_lidar_data("explicit_collision_target", "ranges")
    expected_separation = _lidar_presets.VISUAL_TARGET_EXPECTED_RANGE - _lidar_presets.EXPLICIT_COLLISION_EXPECTED_RANGE
    if not torch.allclose(
        visual_ranges - explicit_ranges,
        torch.full_like(visual_ranges, expected_separation),
        atol=2e-3,
        rtol=2e-3,
    ):
        failures.append(
            "default discovery did not pass through the collision-only proxy while the explicit path still hit it"
        )

    ground_expected = torch.full(
        (args.num_envs, 1),
        2.0,
        dtype=torch.float32,
        device=device,
    )
    failures += _check_exact_downward_return(sim, "ground_visual", ground_expected)

    robot_ranges = sim.get_lidar_data("robot_visual", "ranges")
    robot_points = sim.get_lidar_data("robot_visual", "points")
    robot_ground = _lidar_presets.ROBOT_VISUAL_SENSOR_HEIGHT + env_origins[:, 2].to(
        device=robot_ranges.device, dtype=robot_ranges.dtype
    )
    if not torch.isfinite(robot_ranges).all():
        failures.append("robot_visual: opaque robot visuals produced non-finite returns")
    elif not torch.all(robot_ranges[:, 0] < robot_ground - 0.5):
        failures.append(
            "robot_visual: downward ray reached the ground instead of an opaque robot visual "
            f"(actual={robot_ranges[:, 0].tolist()}, ground={robot_ground.tolist()})"
        )
    robot_direction = torch.tensor(
        [0.0, 0.0, -1.0],
        dtype=robot_ranges.dtype,
        device=robot_ranges.device,
    )
    expected_robot_points = robot_direction.view(1, 1, 3) * robot_ranges.unsqueeze(-1)
    if not torch.allclose(robot_points, expected_robot_points, atol=2e-3, rtol=2e-3):
        failures.append("robot_visual: public XYZ does not agree with its visual-surface range")

    if args.result_file:
        with open(args.result_file, "w") as result:
            result.write("OK" if not failures else "FAIL\n" + "\n".join(failures))
    if failures:
        for failure in failures:
            print(f"[isaacsim] FAIL: {failure}")
        return 1
    print(
        "[isaacsim] PASS: default discovery returns visual robot/scene/ground surfaces, "
        "excludes the collision-only proxy, and preserves explicit targets"
    )
    return 0


if __name__ == "__main__":
    run_and_hard_exit(main)
