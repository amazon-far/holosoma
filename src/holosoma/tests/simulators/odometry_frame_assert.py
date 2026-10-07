"""Cross-backend regression for arbitrary-frame ROS2 odometry kinematics.

The test attaches a nontrivial frame transform to the G1 pelvis and launches the robot with
translation and rotation, clear of contacts. It checks:

1. The plugin's frame pose against an independent NumPy transform composition.
2. Each backend's pelvis link-origin velocity against central differences of pelvis poses.
3. The plugin's child-frame twist against central differences of the transformed frame poses.

The pelvis inertial COM is 7.6 cm away from its link origin, so the body-velocity check catches
backends that pair link-origin pose with COM velocity.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
from pathlib import Path

# Running this file directly puts tests/simulators first, where the isaacsim test package would
# shadow NVIDIA's isaacsim package.
if sys.path and sys.path[0].endswith("tests/simulators"):
    sys.path.pop(0)

import numpy as np

from holosoma.config_types.plugin import ROS2OdometryPluginConfig
from holosoma.simulator.shared.ros2_plugins import ROS2OdometryPlugin
from holosoma.utils.sim_utils import setup_simulation_environment
from tests.simulators._sim_harness import build_run_sim_config, run_and_hard_exit

BODY_NAME = "pelvis"
FRAME_POSITION_B = np.array([0.31, -0.17, 0.23], dtype=np.float64)
FRAME_ORIENTATION_WXYZ = np.array([0.81, 0.22, -0.31, 0.43], dtype=np.float64)
FRAME_ORIENTATION_WXYZ /= np.linalg.norm(FRAME_ORIENTATION_WXYZ)

ROOT_POSITION_W = np.array([0.4, -0.3, 3.0], dtype=np.float64)
ROOT_ORIENTATION_XYZW = np.array([0.19, -0.27, 0.14, 0.91], dtype=np.float64)
ROOT_ORIENTATION_XYZW /= np.linalg.norm(ROOT_ORIENTATION_XYZW)
ROOT_LINEAR_VELOCITY_W = np.array([0.7, -0.4, 0.3], dtype=np.float64)
ROOT_ANGULAR_VELOCITY_W = np.array([2.1, -1.6, 1.2], dtype=np.float64)

POSE_POSITION_ATOL = 2e-4
POSE_ROTATION_ATOL = 3e-4
LINEAR_VELOCITY_ATOL = 6e-2
ANGULAR_VELOCITY_ATOL = 8e-2


def _rotation_xyzw(quaternion: np.ndarray) -> np.ndarray:
    """Quaternion-to-matrix conversion independent of holosoma's rotation helpers."""
    x, y, z, w = quaternion / np.linalg.norm(quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rotation_wxyz(quaternion: np.ndarray) -> np.ndarray:
    return _rotation_xyzw(quaternion[[1, 2, 3, 0]])


def _angular_velocity_world(rotation_minus: np.ndarray, rotation_plus: np.ndarray, dt: float) -> np.ndarray:
    """World angular velocity from the rotation taking the minus pose to the plus pose."""
    delta = rotation_plus @ rotation_minus.T
    cos_theta = float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0))
    theta = math.acos(cos_theta)
    vee = np.array(
        [
            delta[2, 1] - delta[1, 2],
            delta[0, 2] - delta[2, 0],
            delta[1, 0] - delta[0, 1],
        ]
    )
    if theta < 1e-7:
        rotation_vector = 0.5 * vee
    else:
        rotation_vector = theta * vee / (2.0 * math.sin(theta))
    return rotation_vector / (2.0 * dt)


def _build(simulator: str):
    config = build_run_sim_config(
        simulator,
        scene="empty",
        robot="g1-29dof",
        terrain="terrain_locomotion_plane",
    )
    device = "cpu" if simulator == "mujoco" else "cuda:0"
    config = dataclasses.replace(
        config,
        device=device,
        training=dataclasses.replace(config.training, num_envs=1, headless=True),
    )

    import torch

    env, resolved_device, _ = setup_simulation_environment(config, device=device)
    sim = env.sim
    sim.set_headless(True)
    sim.setup()
    sim.setup_terrain()
    sim.load_assets()
    origins = torch.zeros(1, 3, device=resolved_device)
    init = config.robot.init_state
    base_init = torch.tensor(
        list(init.pos) + list(init.rot) + list(init.lin_vel) + list(init.ang_vel),
        device=resolved_device,
    )
    sim.create_envs(1, origins, base_init)
    sim.prepare_sim()
    sim.install_plugins()
    return sim


def _seed_motion(sim) -> None:
    import torch

    env_ids = torch.tensor([0], dtype=torch.long, device=sim.sim_device)
    root = sim.robot_root_states[:].clone()
    root[0, 0:3] = torch.as_tensor(ROOT_POSITION_W, dtype=root.dtype, device=root.device)
    root[0, 3:7] = torch.as_tensor(ROOT_ORIENTATION_XYZW, dtype=root.dtype, device=root.device)
    root[0, 7:10] = torch.as_tensor(ROOT_LINEAR_VELOCITY_W, dtype=root.dtype, device=root.device)
    root[0, 10:13] = torch.as_tensor(ROOT_ANGULAR_VELOCITY_W, dtype=root.dtype, device=root.device)
    sim.robot_root_states[:] = root
    # Passing no state uses the backend's live robot_root_states view/proxy on all backends.
    sim.set_actor_root_state_tensor_robots(env_ids)
    sim.write_state_updates()
    sim.refresh_sim_tensors()


def _make_plugin(sim) -> ROS2OdometryPlugin:
    cfg = ROS2OdometryPluginConfig(
        body_name=BODY_NAME,
        position=FRAME_POSITION_B.tolist(),
        orientation=FRAME_ORIENTATION_WXYZ.tolist(),
        child_frame_id="regression_frame",
    )
    # _read_frame_state has no ROS dependency. Avoid __init__, which creates an rclpy node; unit
    # tests separately cover copying this state into nav_msgs/Odometry.
    plugin = ROS2OdometryPlugin.__new__(ROS2OdometryPlugin)
    plugin.cfg = cfg
    plugin.simulator = sim
    plugin._body_index = None
    return plugin


def _sample(sim, plugin: ROS2OdometryPlugin) -> dict[str, np.ndarray]:
    body_index = sim.find_rigid_body_indice(BODY_NAME)
    frame_pos, frame_quat, frame_lin, frame_ang, _ = plugin._read_frame_state()
    return {
        "body_pos": sim.rigid_body_pos_w[0, body_index].detach().cpu().numpy().astype(np.float64),
        "body_rot": _rotation_xyzw(sim.rigid_body_quat_w[0, body_index].detach().cpu().numpy().astype(np.float64)),
        "body_lin": sim.rigid_body_lin_vel_w[0, body_index].detach().cpu().numpy().astype(np.float64),
        "body_ang": sim.rigid_body_ang_vel_w[0, body_index].detach().cpu().numpy().astype(np.float64),
        "frame_pos": np.asarray(frame_pos, dtype=np.float64),
        "frame_rot": _rotation_xyzw(np.asarray(frame_quat, dtype=np.float64)),
        "frame_lin": np.asarray(frame_lin, dtype=np.float64),
        "frame_ang": np.asarray(frame_ang, dtype=np.float64),
    }


def _advance(sim) -> None:
    sim.simulate_at_each_physics_step()
    sim.refresh_sim_tensors()


def _norm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(vector))


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-backend arbitrary-frame odometry regression.")
    parser.add_argument("--simulator", required=True, choices=["mujoco", "mjwarp", "isaacgym", "isaacsim"])
    parser.add_argument("--result-file", type=Path)
    args = parser.parse_args()

    sim = _build(args.simulator)
    _seed_motion(sim)
    plugin = _make_plugin(sim)

    # Let backend caches and the articulation settle onto the seeded state, then collect a central
    # difference stencil around the middle sample.
    _advance(sim)
    minus = _sample(sim, plugin)
    _advance(sim)
    middle = _sample(sim, plugin)
    _advance(sim)
    plus = _sample(sim, plugin)
    dt = float(sim.sim_dt)

    frame_rotation_b = _rotation_wxyz(FRAME_ORIENTATION_WXYZ)
    expected_frame_pos = middle["body_pos"] + middle["body_rot"] @ FRAME_POSITION_B
    expected_frame_rot = middle["body_rot"] @ frame_rotation_b
    pose_position_error = _norm(middle["frame_pos"] - expected_frame_pos)
    pose_rotation_error = _norm(middle["frame_rot"] - expected_frame_rot)

    body_lin_fd = (plus["body_pos"] - minus["body_pos"]) / (2.0 * dt)
    body_ang_fd = _angular_velocity_world(minus["body_rot"], plus["body_rot"], dt)
    body_linear_error = _norm(middle["body_lin"] - body_lin_fd)
    body_angular_error = _norm(middle["body_ang"] - body_ang_fd)

    frame_lin_world_fd = (plus["frame_pos"] - minus["frame_pos"]) / (2.0 * dt)
    frame_ang_world_fd = _angular_velocity_world(minus["frame_rot"], plus["frame_rot"], dt)
    frame_lin_fd = middle["frame_rot"].T @ frame_lin_world_fd
    frame_ang_fd = middle["frame_rot"].T @ frame_ang_world_fd
    frame_linear_error = _norm(middle["frame_lin"] - frame_lin_fd)
    frame_angular_error = _norm(middle["frame_ang"] - frame_ang_fd)

    checks = {
        "pose_position": pose_position_error <= POSE_POSITION_ATOL,
        "pose_rotation": pose_rotation_error <= POSE_ROTATION_ATOL,
        "body_linear": body_linear_error <= LINEAR_VELOCITY_ATOL,
        "body_angular": body_angular_error <= ANGULAR_VELOCITY_ATOL,
        "frame_linear": frame_linear_error <= LINEAR_VELOCITY_ATOL,
        "frame_angular": frame_angular_error <= ANGULAR_VELOCITY_ATOL,
    }
    print(
        f"ODOMETRY_KINEMATICS backend={args.simulator} dt={dt:.6f} "
        f"pose_pos_err={pose_position_error:.6g} pose_rot_err={pose_rotation_error:.6g} "
        f"body_lin_err={body_linear_error:.6g} body_ang_err={body_angular_error:.6g} "
        f"frame_lin_err={frame_linear_error:.6g} frame_ang_err={frame_angular_error:.6g}"
    )
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        print(f"FAIL: {args.simulator} arbitrary-frame odometry checks failed: {failed}")
        return 1

    if args.result_file is not None:
        args.result_file.write_text("OK\n")
    print(f"PASS: {args.simulator} arbitrary-frame pose and twist")
    return 0


if __name__ == "__main__":
    run_and_hard_exit(main)
