# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""Cross-backend mounting randomization: real depth/range changes and env independence."""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
from pathlib import Path

if sys.path and sys.path[0].endswith("simulators"):
    sys.path.pop(0)

from tests.simulators._sim_harness import build_run_sim_config, launch_simulator, run_and_hard_exit, step


def _sensors():
    from holosoma.config_types.sensor import (
        CameraSensorConfig,
        CustomLidarRayPatternConfig,
        LidarSensorConfig,
        SensorMountConfig,
    )

    mounts = {
        "world": SensorMountConfig(target_kind="world", position=[3.0, 0.0, 2.0]),
        "link": SensorMountConfig(target_kind="robot_link", target="pelvis", position=[3.0, 0.0, 1.5]),
        "actor": SensorMountConfig(target_kind="actor", target="panel", position=[3.0, 0.0, 1.5]),
    }
    sensors = {}
    for kind, mount in mounts.items():
        sensors[f"{kind}_camera"] = CameraSensorConfig(mount=mount, width=65, height=65, data_types=["depth"], far=20.0)
        sensors[f"{kind}_lidar"] = LidarSensorConfig(
            mount=mount, pattern=CustomLidarRayPatternConfig(ray_directions=[[0.0, 0.0, -1.0]]), far=20.0
        )
    return sensors


def _rotate(quat, vector):
    """Independent XYZW quaternion-vector rotation for expected plane intersections."""
    import torch

    xyz = quat[:, :3]
    twice_cross = 2.0 * torch.cross(xyz, vector, dim=-1)
    return vector + quat[:, 3:] * twice_cross + torch.cross(xyz, twice_cross, dim=-1)


def _check(sim, offsets, label: str) -> None:
    import torch

    ids = torch.arange(sim.num_envs, device=sim.sim_device)
    identity = torch.tensor([0.0, 0.0, 0.0, 1.0], device=sim.sim_device).repeat(sim.num_envs, 1)
    link_index = sim.find_rigid_body_indice("pelvis")
    actor = sim.get_actor_states(["panel"], ids)
    parents = {
        "world": (sim.env_origins, identity),
        "link": (sim.rigid_body_pos_w[:, link_index], sim.rigid_body_quat_w[:, link_index]),
        "actor": (actor[:, :3], actor[:, 3:7]),
    }
    down = torch.tensor([0.0, 0.0, -1.0], device=sim.sim_device).repeat(sim.num_envs, 1)
    for name, pose in offsets.items():
        parent_pos, parent_quat = parents[name.split("_")[0]]
        origin = parent_pos + _rotate(parent_quat, pose[:, :3])
        direction = _rotate(parent_quat, _rotate(pose[:, 3:], down))
        expected = origin[:, 2] / -direction[:, 2]
        assert torch.all(direction[:, 2] < -0.5), f"{label}/{name}: test ray no longer points toward the ground"
        if name.startswith("world_") and sim.simulator_config.name == "mujoco":
            for env_id in range(sim.num_envs):
                data = sim.backend.get_render_data(env_id)
                if name.endswith("camera"):
                    position = data.cam_xpos[sim.root_model.camera(name).id]
                else:
                    position = data.site_xpos[sim.root_model.site(f"holosoma_lidar_mount_{name}").id]
                    position = position + sim.env_origins[env_id].cpu().numpy()
                torch.testing.assert_close(
                    torch.as_tensor(position, device=sim.sim_device, dtype=origin.dtype),
                    origin[env_id],
                    atol=1e-5,
                    rtol=0,
                    msg=f"{label}/{name}: native mount position",
                )
        if name.endswith("camera"):
            actual = sim.get_camera_data(name, "depth")[:, 32, 32, 0]
        else:
            actual = sim.get_lidar_data(name, "ranges")[:, 0]
            points = sim.get_lidar_data(name, "points")[:, 0]
            torch.testing.assert_close(points[:, :2], torch.zeros_like(points[:, :2]), atol=1e-4, rtol=0)
            torch.testing.assert_close(points[:, 2], -actual)
        print(f"{label}/{name}: actual={actual.tolist()} expected={expected.tolist()}")
        torch.testing.assert_close(actual, expected, atol=0.035, rtol=0.015, msg=f"{label}/{name}")
        torch.testing.assert_close(sim.sensor_manager.get_mount_pose(name), pose)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulator", required=True, choices=["mujoco", "mjwarp", "isaacsim"])
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--result-file", required=True)
    args = parser.parse_args()

    from tests.simulators import _camera_presets

    config = build_run_sim_config(
        "mujoco" if args.simulator == "mjwarp" else args.simulator,
        "panel-target",
        "g1-29dof",
        "terrain_locomotion_plane",
        sensors=_sensors(),
    )
    if args.simulator == "mjwarp":
        config = _camera_presets.as_mjwarp(config)
    device = "cpu" if args.simulator == "mujoco" else "cuda:0"
    config = dataclasses.replace(
        config, device=device, training=dataclasses.replace(config.training, num_envs=args.num_envs, headless=True)
    )
    sim, device, _app = launch_simulator(config, device=device)
    manager = sim.sensor_manager
    sim.set_headless(True)
    sim.setup()
    sim.setup_terrain()
    sim.load_assets()

    import torch

    ids = torch.arange(args.num_envs, device=device)
    origins = torch.zeros(args.num_envs, 3, device=device)
    origins[:, 0] = 5.0 + ids * 10.0
    origins[:, 1] = -2.0
    origins[:, 2] = 0.25 + ids * 0.25
    init = config.robot.init_state
    base_init = torch.tensor(
        list(init.pos) + list(init.rot) + list(init.lin_vel) + list(init.ang_vel), device=device, dtype=torch.float32
    )
    sim.create_envs(args.num_envs, origins, base_init)
    sim.prepare_sim()
    assert sim.sensor_manager is manager, "native setup replaced the public registry"
    initial = {name: sim.sensor_manager.get_mount_pose(name) for name in sim.get_sensor_names()}
    offsets = {name: pose.clone() for name, pose in initial.items()}
    root_states = sim.get_actor_states(["robot"], ids).clone()
    joints = sim.dof_pos.clone()
    failures = []
    if args.simulator == "mjwarp":
        graph = sim.backend.step_graph
        arrays = [
            sim.backend.mjw_model.cam_pos,
            sim.backend.mjw_model.cam_quat,
            sim.backend.mjw_model.site_pos,
            sim.backend.mjw_model.site_quat,
        ]
        pointers = [array.ptr for array in arrays]
        write_stream = torch.cuda.Stream(device=device)

    def capture(label: str) -> None:
        # Hold the robot at a known pose while running the actual physics graph. An unactuated
        # robot can otherwise fall far enough during render warmup to point its sensors upward.
        for _ in range(4):
            sim.set_actor_states(["robot"], ids, root_states)
            sim.dof_pos[:] = joints
            sim.dof_vel[:] = 0.0
            sim.forward_kinematics()
            step(sim, 1)
        sim.render_sensors()
        _check(sim, offsets, label)
        if args.simulator == "mjwarp":
            assert sim.backend.step_graph is graph, "mount updates recaptured the physics graph"
            assert [
                sim.backend.mjw_model.cam_pos.ptr,
                sim.backend.mjw_model.cam_quat.ptr,
                sim.backend.mjw_model.site_pos.ptr,
                sim.backend.mjw_model.site_quat.ptr,
            ] == pointers, "mount updates replaced arrays referenced by captured graphs"

    try:
        capture("initial")
        initial_outputs = {
            name: (
                sim.get_camera_data(name, "depth") if name.endswith("camera") else sim.get_lidar_data(name, "ranges")
            ).clone()
            for name in initial
        }
        for name, pose in initial.items():
            sim.sensor_manager.set_mount_pose(name, pose)
        capture("unchanged offsets")
        for name, output in initial_outputs.items():
            if name.startswith("world_"):
                actual = (
                    sim.get_camera_data(name, "depth")
                    if name.endswith("camera")
                    else sim.get_lidar_data(name, "ranges")
                )
                torch.testing.assert_close(actual, output)
        for name, pose in offsets.items():
            old_output = (
                sim.get_camera_data(name, "depth") if name.endswith("camera") else sim.get_lidar_data(name, "ranges")
            ).clone()
            selected = ids[:1]
            pose[selected, 2] += 0.4
            clock_before = sim.time()
            sim.sensor_manager.set_mount_pose(name, pose[selected], selected)
            assert sim.time() == clock_before, "mount updates advanced simulation time"
            cached = (
                sim.get_camera_data(name, "depth") if name.endswith("camera") else sim.get_lidar_data(name, "ranges")
            )
            torch.testing.assert_close(cached, old_output)
        capture("translated subset")
        # Orientation updates on a different subset catch quaternion-order and shared-array bugs.
        selected = ids[-1:]
        angle = math.radians(30.0)
        rotation = torch.tensor([0.0, math.sin(angle / 2), 0.0, math.cos(angle / 2)], device=device)
        for name, pose in offsets.items():
            pose[selected, 3:] = rotation
            if args.simulator == "mjwarp":
                # Produce pose rows and call the writer on a non-default Torch stream. The
                # subsequent native graph/render must wait for the correct stream handoff.
                write_stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(write_stream):
                    sim.sensor_manager.set_mount_pose(name, pose[selected].clone(), selected)
                torch.cuda.current_stream(device).wait_stream(write_stream)
            else:
                sim.sensor_manager.set_mount_pose(name, pose[selected], selected)
        capture("rotated subset")
        root_states[:, 2] += 0.2
        parent_angle = math.radians(10.0)
        parent_rotation = torch.tensor(
            [0.0, math.sin(parent_angle / 2), 0.0, math.cos(parent_angle / 2)], device=device
        )
        root_states[:, 3:7] = parent_rotation
        panel_states = sim.get_actor_states(["panel"], ids).clone()
        panel_states[:, 2] += 0.2
        panel_states[:, 3:7] = parent_rotation
        sim.set_actor_states(["panel"], ids, panel_states)
        capture("moved parents")
        # Repeated writes followed by captured steps must keep reading the same model arrays.
        for name, pose in initial.items():
            sim.sensor_manager.set_mount_pose(name, pose)
            offsets[name] = pose.clone()
        capture("restored offsets")
    except AssertionError as exc:
        failures.append(str(exc))
    Path(args.result_file).write_text("FAIL\n" + "\n".join(failures) if failures else "OK")
    if failures:
        print("\n".join(failures))
        return 1
    print(f"PASS: runtime camera/LiDAR mounting poses on {args.simulator}, {args.num_envs} env(s)")
    return 0


if __name__ == "__main__":
    run_and_hard_exit(main)
