"""Isaac Sim ray-cast depth camera assertion harness (``isaacsim.renderer="raycast"``).

Uses the ``raycast_cam`` camera, which casts against the terrain only. ``--check frame`` asserts the
camera produces a depth frame without an RTX ``TiledCamera`` or any RTX sensor, and that the flat
ground is smooth row by row. ``--check rerender``
raises the robot in two envs without stepping, re-renders those envs, and asserts their ground depth
grows while the third env's frame is unchanged.

``--check scene`` renders a loaded object with two cameras, excludes it from one camera, moves it
without stepping, and checks that both cameras and LiDAR reuse the native scene mesh cache.

``--check dynamic-scene`` translates and rotates a free rigid object through physics steps, checks
depth against an analytic box intersection, and verifies exclusions and partial-frame isolation.

``--check default-selection`` hides the plane terrain's visual and uses the default mesh selection on
the pelvis-mounted camera: the ground stays selected and visible, and the pelvis is left out.

Exits 0 PASS / 1 FAIL / 77 SKIP.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from typing import Any

if sys.path and sys.path[0].endswith("simulators"):
    sys.path.pop(0)

from holosoma.utils.sim_utils import setup_simulation_environment
from tests.simulators._sim_harness import build_run_sim_config, run_and_hard_exit, step, steps_for_seconds

_NUM_ENVS = 3


def _check_frame(sim):
    import torch

    fails = []
    if sim.tiled_cameras:
        fails.append(f"RTX TiledCameras were built: {sorted(sim.tiled_cameras)}")
    if sim.sim.has_rtx_sensors():
        fails.append("RTX sensors are active with only a ray-cast camera configured")
    depth = sim.get_camera_data("raycast", "depth")
    if tuple(depth.shape) != (_NUM_ENVS, 64, 64, 1) or depth.dtype != torch.float32:
        fails.append(f"depth shape/dtype {tuple(depth.shape)} {depth.dtype} != ({_NUM_ENVS}, 64, 64, 1) float32")
    fails.extend(
        f"env{e}: the camera sees almost no ground"
        for e in range(_NUM_ENVS)
        if float(torch.isfinite(depth[e]).float().mean()) < 0.2
    )
    # Flat ground: a pinhole camera's 1/depth over a plane is affine in pixel coordinates, so within a
    # depth camera's working range (3 m) the affine fit must leave only sub-millimetre residuals.
    for e in range(_NUM_ENVS):
        image = depth[e, ..., 0].double().cpu()
        rows, cols = torch.nonzero(torch.isfinite(image) & (image < 3.0), as_tuple=True)
        if rows.numel() < 100:
            fails.append(f"env{e}: only {rows.numel()} ground pixels within 3 m")
            continue
        design = torch.stack([rows.double(), cols.double(), torch.ones_like(rows, dtype=torch.double)], dim=1)
        coeffs = torch.linalg.lstsq(design, (1.0 / image[rows, cols]).unsqueeze(1)).solution
        error = (image[rows, cols] - 1.0 / (design @ coeffs).flatten()).abs().max()
        if float(error) > 1e-3:
            fails.append(f"env{e}: flat ground is not flat in depth (off by up to {1e3 * float(error):.1f} mm)")
    return fails


def _check_rerender(sim):
    import torch

    fails = []
    before = sim.get_camera_data("raycast", "depth")[..., 0].clone()
    env_ids = torch.arange(_NUM_ENVS, device=sim.sim_device)
    moved = torch.tensor([1, 2], device=sim.sim_device)
    states = sim.get_actor_states(["robot"], env_ids).clone()
    states[moved, 2] += 0.3
    sim.set_actor_states(["robot"], moved, states[moved])
    sim.refresh_sim_tensors()
    sim.rerender_sensors(moved)
    after = sim.get_camera_data("raycast", "depth")[..., 0]

    if not torch.equal(after[0], before[0]):
        fails.append("env0: frame changed although env 0 was not re-rendered")
    for e in (1, 2):
        ground = torch.isfinite(before[e]) & torch.isfinite(after[e])
        if not bool(ground.any()):
            fails.append(f"env{e}: no ground visible before and after the move")
        elif float((after[e][ground] > 1.1 * before[e][ground]).float().mean()) < 0.8:
            fails.append(f"env{e}: ground depth did not grow after raising the robot by 0.3 m")
    return fails


def _check_scene(sim):
    import numpy as np
    import torch

    fails = _check_frame(sim)
    depth = sim.get_camera_data("raycast", "depth")[..., 0].clone()
    excluded = sim.get_camera_data("without_target", "depth")[..., 0].clone()
    # The 0.1 m cube is at x=.5 and the camera at x=.1; its front face is .35 m away.
    for e in range(_NUM_ENVS):
        center = float(depth[e, 32, 32])
        if abs(center - 0.35) > 0.01:
            fails.append(f"env{e}: loaded object's center depth {center:.4f} != .35 m")
        if float(excluded[e, 32, 32]) < 1.0:
            fails.append(f"env{e}: excluded object still appears in the second camera")

    # The depth geometry view must reuse the same mesh IDs as LiDAR, not upload its own copies.
    native = sim.scene.sensors["raycast"]
    lidar = sim.scene.sensors["scan"]
    if not np.array_equal(np.sort(native._mesh_ids_wp.numpy(), axis=1), np.sort(lidar._mesh_ids_wp.numpy(), axis=1)):
        fails.append("depth and LiDAR did not reuse the same scene mesh IDs")
    renderer = sim.raycast_cameras["raycast"]
    if renderer._meshes.mesh_ids is not native._mesh_ids_wp:
        fails.append("depth renderer copied the native scene mesh IDs")

    moved = torch.tensor([1], device=sim.sim_device)
    states = sim.get_actor_states(["target"], moved).clone()
    states[:, 0] += 0.2
    sim.set_actor_states(["target"], moved, states)
    sim.refresh_sim_tensors()
    sim.rerender_sensors(moved)
    after = sim.get_camera_data("raycast", "depth")[..., 0]
    if abs(float(after[1, 32, 32]) - 0.55) > 0.01:
        fails.append(f"env1: moved object's center depth {float(after[1, 32, 32]):.4f} != .55 m")
    if not torch.equal(after[[0, 2]], depth[[0, 2]]):
        fails.append("object rerender changed environments outside the requested subset")
    if not torch.equal(sim.get_camera_data("without_target", "depth")[..., 0], excluded):
        fails.append("moving an excluded object changed the second camera's frame")
    return fails


def _box_center_depth(sim, state):
    """Independent slab intersection for the fixture's 0.1 m cube; parameter is image-plane depth."""
    import math

    import numpy as np
    from scipy.spatial.transform import Rotation

    cfg = sim.sensor_config["raycast"]
    body = sim.find_rigid_body_indice(cfg.mount.target)
    body_position = sim.rigid_body_pos_w[1, body].cpu().numpy()
    body_rotation = Rotation.from_quat(sim.rigid_body_quat_w[1, body].cpu().numpy())
    origin = body_position + body_rotation.apply(cfg.mount.position)
    w, x, y, z = cfg.mount.orientation
    optical_rotation = body_rotation * Rotation.from_quat([x, y, z, w])
    focal = cfg.height / (2 * math.tan(math.radians(cfg.vertical_fov) / 2))
    # Optical -Z forward, +Y up. Leave the ray unnormalized so its parameter is z-depth.
    ray = optical_rotation.apply(
        [(cfg.width // 2 + 0.5 - cfg.width / 2) / focal, -(cfg.height // 2 + 0.5 - cfg.height / 2) / focal, -1]
    )
    box_rotation = Rotation.from_quat(state[3:7])
    origin = box_rotation.inv().apply(origin - state[:3])
    ray = box_rotation.inv().apply(ray)
    near, far = 0.0, np.inf
    for coordinate, direction in zip(origin, ray, strict=True):
        if abs(direction) < 1e-12:
            if abs(coordinate) > 0.05:
                return np.inf
            continue
        intersections = [(-0.05 - coordinate) / direction, (0.05 - coordinate) / direction]
        near, far = max(near, min(intersections)), min(far, max(intersections))
    return near if near <= far else np.inf


def _check_dynamic_scene(sim):
    import numpy as np
    import torch

    fails = []
    env_ids = torch.arange(_NUM_ENVS, device=sim.sim_device)
    moved = torch.tensor([1], device=sim.sim_device)
    robot_states = sim.get_actor_states(["robot"], env_ids).clone()
    states = sim.get_actor_states(["target"], env_ids).clone()
    states[:, :3] = sim.env_origins + torch.tensor([0.5, 0.0, 0.79], device=sim.sim_device)
    states[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=sim.sim_device)
    states[:, 7:] = 0.0
    states[1, 7] = 1.0  # Free-body translation, not a pose write after each step.
    states[1, 12] = 6.0  # Rotation changes the cube's ray intersection as well.
    sim.set_actor_states(["target"], env_ids, states)
    sim.refresh_sim_tensors()
    sim.render_sensors()
    before = sim.get_camera_data("raycast", "depth")[..., 0].clone()
    excluded = sim.get_camera_data("without_target", "depth")[..., 0].clone()
    native = sim.scene.sensors["raycast"]
    mesh_ids = native._mesh_ids_wp.numpy().copy()
    lidar = sim.scene.sensors["scan"]
    if not np.array_equal(np.sort(mesh_ids, axis=1), np.sort(lidar._mesh_ids_wp.numpy(), axis=1)):
        fails.append("dynamic actor: depth and LiDAR did not share mesh IDs")

    elapsed = 0.0
    for frame in range(3):
        if frame:
            count = steps_for_seconds(sim, 0.04)
            step(sim, count)
            elapsed += count * sim.sim_dt
            # Hold the camera mount fixed; only the target's motion should change its depth.
            sim.set_actor_states(["robot"], env_ids, robot_states)
            sim.refresh_sim_tensors()
            sim.rerender_sensors(moved)
        state = sim.get_actor_states(["target"], moved)[0].cpu().numpy()
        expected = _box_center_depth(sim, state)
        after = sim.get_camera_data("raycast", "depth")[..., 0]
        actual = float(after[1, 32, 32])
        if not np.isfinite(expected) or not np.isfinite(actual) or abs(actual - expected) > 0.002:
            fails.append(f"dynamic actor t={elapsed:.3f}: center depth {actual:.5f} != analytic {expected:.5f} m")
        if not torch.equal(after[[0, 2]], before[[0, 2]]):
            fails.append("dynamic actor: partial rerender changed unrequested environments")
        if not torch.equal(sim.get_camera_data("without_target", "depth")[..., 0], excluded):
            fails.append("dynamic actor: excluded camera changed")
        if not np.array_equal(native._mesh_ids_wp.numpy(), mesh_ids):
            fails.append("dynamic actor: movement replaced cached mesh IDs")
        print(
            f"[isaacsim] dynamic target t={elapsed:.3f}: position={state[:3].tolist()} "
            f"quaternion_xyzw={state[3:7].tolist()} depth={actual:.5f} analytic={expected:.5f}"
        )
    if state[0] - float(states[1, 0]) < 0.05 or state[2] - float(states[1, 2]) > -0.005:
        fails.append("dynamic actor did not translate and fall under physics")
    if abs(state[5]) < 0.1:
        fails.append("dynamic actor did not rotate under physics")
    if abs(actual - float(before[1, 32, 32])) < 0.05:
        fails.append("dynamic actor motion did not change captured depth")
    return fails


def _check_default_selection(sim):
    import torch

    fails = []
    paths = [target.prim_expr for target in sim.raycast_cameras["raycast"].mesh_provider.cfg.mesh_prim_paths]
    if not any(path.startswith("/World/ground") for path in paths):
        fails.append("the hidden ground was not selected")
    if any("/Robot/pelvis/" in path for path in paths):
        fails.append("the default selection includes the camera's pelvis mount body")
    if not any("/Robot/" in path for path in paths):
        fails.append("the default selection dropped the rest of the robot")
    depth = sim.get_camera_data("raycast", "depth")[..., 0]
    fails.extend(
        f"env{e}: the camera sees almost no ground with the terrain visual hidden"
        for e in range(_NUM_ENVS)
        if float(torch.isfinite(depth[e]).float().mean()) < 0.2
    )
    return fails


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", required=True, choices=["frame", "rerender", "scene", "dynamic-scene", "default-selection"]
    )
    parser.add_argument("--result-file", default=None, help="write OK/FAIL here before teardown")
    args = parser.parse_args()

    from tests.simulators import _camera_presets

    sensors: dict[str, Any] = dict(_camera_presets.raycast_cam)
    scene_name = "empty"
    if args.check in ("scene", "dynamic-scene"):
        from holosoma.config_types.sensor import (
            IsaacSimCameraConfig,
            IsaacSimRaycastMeshConfig,
            LidarBodyFilterConfig,
            LidarSensorConfig,
        )

        _camera_presets.register()
        scene_name = "camera-target"
        camera = sensors["raycast"]
        sensors = {
            "raycast": dataclasses.replace(
                camera,
                isaacsim=IsaacSimCameraConfig(
                    renderer="raycast",
                    raycast_meshes=IsaacSimRaycastMeshConfig(
                        include_paths=["/World"],
                        exclude_paths=["/World/envs/env_.*/Robot"],
                    ),
                ),
            ),
            "without_target": dataclasses.replace(
                camera,
                isaacsim=IsaacSimCameraConfig(
                    renderer="raycast",
                    raycast_meshes=IsaacSimRaycastMeshConfig(
                        exclude_paths=["/World/envs/env_.*/Robot", "/World/envs/env_.*/target"]
                    ),
                ),
            ),
            "scan": LidarSensorConfig(
                mount=camera.mount,
                body_filter=LidarBodyFilterConfig(target_kind="robot"),
            ),
        }

    if args.check == "default-selection":
        from holosoma.config_types.sensor import IsaacSimCameraConfig

        sensors = {
            "raycast": dataclasses.replace(sensors["raycast"], isaacsim=IsaacSimCameraConfig(renderer="raycast"))
        }

    config = build_run_sim_config("isaacsim", scene_name, "g1-29dof", "terrain_locomotion_plane", sensors=sensors)
    if args.check == "default-selection":
        terrain = dataclasses.replace(config.terrain.terrain_term, hide_visual=True)
        config = dataclasses.replace(config, terrain=dataclasses.replace(config.terrain, terrain_term=terrain))
    if args.check == "dynamic-scene":
        objects = dict(config.scene.rigid_objects)
        objects["target"] = dataclasses.replace(objects["target"], fixed=False)
        config = dataclasses.replace(config, scene=dataclasses.replace(config.scene, rigid_objects=objects))
    config = dataclasses.replace(
        config, device="cuda:0", training=dataclasses.replace(config.training, num_envs=_NUM_ENVS)
    )
    env, device, _app = setup_simulation_environment(config, device="cuda:0")
    sim = env.sim
    sim.set_headless(True)
    sim.setup()
    sim.setup_terrain()
    sim.load_assets()
    import torch

    init = config.robot.init_state
    base_init = torch.tensor(
        list(init.pos) + list(init.rot) + list(init.lin_vel) + list(init.ang_vel), device=device, dtype=torch.float32
    )
    sim.create_envs(_NUM_ENVS, torch.zeros(_NUM_ENVS, 3, device=device), base_init)
    sim.prepare_sim()
    sim.install_plugins()

    # Pin each robot upright at its env origin, then capture one frame from that state.
    env_ids = torch.arange(_NUM_ENVS, device=device)
    robot_states = sim.get_actor_states(["robot"], env_ids).clone()
    robot_states[:, :3] = sim.env_origins + torch.tensor(list(init.pos), device=device)
    robot_states[:, 3:7] = torch.tensor(list(init.rot), device=device)
    robot_states[:, 7:] = 0.0
    sim.set_actor_states(["robot"], env_ids, robot_states)
    step(sim, max(2, steps_for_seconds(sim, 0.05)))
    sim.set_actor_states(["robot"], env_ids, robot_states)
    sim.refresh_sim_tensors()
    sim.render_sensors()

    checks = {
        "frame": _check_frame,
        "rerender": _check_rerender,
        "scene": _check_scene,
        "dynamic-scene": _check_dynamic_scene,
        "default-selection": _check_default_selection,
    }
    fails = checks[args.check](sim)
    for failure in fails:
        print(f"[isaacsim] FAIL: {failure}")
    if args.result_file:
        with open(args.result_file, "w") as fh:
            fh.write("OK" if not fails else "FAIL\n" + "\n".join(fails))
    if fails:
        return 1
    print(f"[isaacsim] PASS: ray-cast camera {args.check} checks")
    return 0


if __name__ == "__main__":
    run_and_hard_exit(main)
