# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""Native camera depth and LiDAR range checks, including a mount fused into its parent."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from holosoma.config_types.full_sim import FullSimConfig  # noqa: E402
from holosoma.config_types.sensor import (  # noqa: E402
    CameraSensorConfig,
    CustomLidarRayPatternConfig,
    LidarSensorConfig,
    SensorMountConfig,
)
from holosoma.config_values.simulator import mujoco as mujoco_config  # noqa: E402
from holosoma.simulator.mujoco.backends.classic_backend import ClassicBackend  # noqa: E402
from holosoma.simulator.mujoco.mujoco import MuJoCo  # noqa: E402
from holosoma.simulator.mujoco.scene_manager import ActorSpecMeta, MujocoSceneManager  # noqa: E402
from holosoma.simulator.mujoco.sensor_setup import create_sensors, initialize_camera_mounts  # noqa: E402
from holosoma.simulator.shared.sensor_manager import SensorManager  # noqa: E402
from holosoma.utils.safe_torch_import import torch  # noqa: E402

if TYPE_CHECKING:
    from holosoma.config_types.sensor import SensorConfig

pytestmark = pytest.mark.mujoco_classic


def _make_sim(model: Any, sensors: dict[str, SensorConfig], origins: torch.Tensor) -> MuJoCo:
    """Run the real sensor pipeline on a small native scene, without an unrelated robot lifecycle."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    config = cast("FullSimConfig", SimpleNamespace(training=SimpleNamespace(num_envs=1)))
    instance = MuJoCo.__new__(MuJoCo)
    instance.root_model, instance.root_data = model, data
    instance.backend = ClassicBackend(model, data, config, "cpu")
    instance.num_envs, instance.sim_device = 1, "cpu"
    instance.env_origins = origins
    instance.simulator_config = mujoco_config.config
    instance.sensor_config = sensors
    instance.kinematic_playback = False
    cfg = instance.simulator_config.sim
    instance.sensor_manager = SensorManager("cpu", cfg.fps / cfg.control_decimation_steps)
    create_sensors(instance)
    initialize_camera_mounts(instance)
    return instance


@pytest.fixture(params=["world", "actor", "robot_link", "fused_link"])
def sim(request: pytest.FixtureRequest) -> Iterator[MuJoCo]:
    scene = MujocoSceneManager.__new__(MujocoSceneManager)
    scene.world_spec = mujoco.MjSpec.from_string(
        """
        <mujoco>
          <option gravity="0 0 0"/>
          <worldbody>
            <geom type="plane" size="10 10 0.1"/>
            <body name="unit_root" pos="0 0 1">
              <freejoint/>
              <geom type="sphere" size="0.1"/>
              <body name="unit_mount" pos="0.1 0.2 0.3" quat="0.707106781 0 0 0.707106781">
                <geom type="sphere" size="0.1"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    scene.world_spec.compiler.fusestatic = request.param == "fused_link"
    scene.robot_spec_meta = ActorSpecMeta(prefix="unit_", root_body="root", body_names=["mount"])
    scene.rigid_object_root_bodies = {"fixture": "unit_mount"}
    scene.scene_file_bodies = {}
    if request.param == "world":
        mount = SensorMountConfig(target_kind="world", position=[3.0, 0.0, 2.0])
    elif request.param == "actor":
        mount = SensorMountConfig(target_kind="actor", target="fixture", position=[3.0, 0.0, 1.0])
    else:
        mount = SensorMountConfig(target_kind="robot_link", target="mount", position=[3.0, 0.0, 1.0])
    camera = CameraSensorConfig(mount=mount, width=65, height=65, data_types=["depth"], far=20.0)
    lidar = LidarSensorConfig(mount=mount, pattern=CustomLidarRayPatternConfig(ray_directions=[[0.0, 0.0, -1.0]]))
    scene.add_cameras({"camera": camera})
    scene.add_lidar_sites({"lidar": lidar})
    model = scene.world_spec.compile()
    origins = torch.tensor([[5.0, 0.0, 0.25]]) if request.param == "world" else torch.zeros(1, 3)
    instance = _make_sim(model, {"camera": camera, "lidar": lidar}, origins)
    try:
        yield instance
    finally:
        instance.backend.close()


def _check_plane_returns(sim: MuJoCo, expected: float) -> None:
    sim.render_sensors()
    depth = float(sim.get_camera_data("camera", "depth")[0, 32, 32, 0])
    distance = float(sim.get_lidar_data("lidar", "ranges")[0, 0])
    assert depth == pytest.approx(expected, abs=0.015)
    assert distance == pytest.approx(expected, abs=1e-4)
    torch.testing.assert_close(
        sim.get_lidar_data("lidar", "points")[0, 0], torch.tensor([0.0, 0.0, -expected]), atol=1e-4, rtol=1e-4
    )


def _check_native_mount_frame(sim: MuJoCo, pose: torch.Tensor, angle: float) -> None:
    """Check all position axes and optical axes against the independently authored mount frame."""
    assert sim.root_model is not None and sim.root_data is not None
    world = sim.sensor_config["camera"].mount.target_kind == "world"
    expected_position = (
        pose[:3].numpy() + np.array([5.0, 0.0, 0.25])
        if world
        else np.array([0.1 - float(pose[1]), 0.2 + float(pose[0]), 1.3 + float(pose[2])])
    )
    c, s = math.cos(angle), math.sin(angle)
    local_rotation = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    mount_rotation = np.eye(3) if world else np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    expected_rotation = mount_rotation @ local_rotation
    cam_id = sim.root_model.camera("camera").id
    site_id = sim.root_model.site("holosoma_lidar_mount_lidar").id
    np.testing.assert_allclose(sim.root_data.cam_xpos[cam_id], expected_position, atol=1e-6)
    origin = sim.root_data.site_xpos[site_id] + (sim.env_origins[0].numpy() if world else 0.0)
    np.testing.assert_allclose(origin, expected_position, atol=1e-6)
    np.testing.assert_allclose(sim.root_data.cam_xmat[cam_id].reshape(3, 3), expected_rotation, atol=1e-6)
    np.testing.assert_allclose(sim.root_data.site_xmat[site_id].reshape(3, 3), expected_rotation, atol=1e-6)


def test_mount_update_changes_native_outputs_without_advancing_physics(sim: MuJoCo) -> None:
    initial = sim.sensor_manager.get_mount_pose("camera")
    initial_height = 2.25 if sim.sensor_config["camera"].mount.target_kind == "world" else 2.3
    # Check setup before any public pose write can hide an incorrect native initial pose.
    _check_native_mount_frame(sim, initial[0], 0.0)
    _check_plane_returns(sim, initial_height)
    initial_frame = sim.get_camera_data("camera", "depth").clone()
    initial_scan = sim.get_lidar_data("lidar", "ranges").clone()
    for name in ("camera", "lidar"):
        sim.sensor_manager.set_mount_pose(name, initial)
    _check_native_mount_frame(sim, initial[0], 0.0)
    _check_plane_returns(sim, initial_height)
    torch.testing.assert_close(sim.get_camera_data("camera", "depth"), initial_frame)
    torch.testing.assert_close(sim.get_lidar_data("lidar", "ranges"), initial_scan)
    old_frame = sim.get_camera_data("camera", "depth").clone()
    old_scan = sim.get_lidar_data("lidar", "ranges").clone()
    assert sim.root_data is not None
    qpos, qvel, sim_time = sim.root_data.qpos.copy(), sim.root_data.qvel.copy(), sim.time()
    angle = math.radians(30.0)
    pose = initial[0].clone()
    pose[2] += 0.4
    pose[3:] = torch.tensor([0.0, math.sin(angle / 2), 0.0, math.cos(angle / 2)])
    for name in ("camera", "lidar"):
        sim.sensor_manager.set_mount_pose(name, pose)
        torch.testing.assert_close(sim.sensor_manager.get_mount_pose(name)[0], pose)
    _check_native_mount_frame(sim, pose, angle)
    np.testing.assert_array_equal(sim.root_data.qpos, qpos)
    np.testing.assert_array_equal(sim.root_data.qvel, qvel)
    assert sim.time() == sim_time
    torch.testing.assert_close(sim.get_camera_data("camera", "depth"), old_frame)
    torch.testing.assert_close(sim.get_lidar_data("lidar", "ranges"), old_scan)
    _check_plane_returns(sim, (initial_height + 0.4) / math.cos(angle))
    # Reset native dynamic state without recreating the model: randomized offsets must survive.
    assert sim.root_model is not None
    mujoco.mj_resetData(sim.root_model, sim.root_data)
    sim.backend.forward_kinematics()
    _check_native_mount_frame(sim, pose, angle)
    _check_plane_returns(sim, (initial_height + 0.4) / math.cos(angle))
    for name in ("camera", "lidar"):
        torch.testing.assert_close(sim.sensor_manager.get_mount_pose(name)[0], pose)
    for name in ("camera", "lidar"):
        sim.sensor_manager.set_mount_pose(name, initial)
    _check_plane_returns(sim, initial_height)


def test_sensor_keeps_following_its_parent_after_offset_changes(sim: MuJoCo) -> None:
    initial = sim.sensor_manager.get_mount_pose("camera")
    pose = initial.clone()
    pose[:, 2] += 0.4
    for name in ("camera", "lidar"):
        sim.sensor_manager.set_mount_pose(name, pose)
    assert sim.root_data is not None
    sim.root_data.qpos[2] += 0.5
    sim.backend.forward_kinematics()
    world = sim.sensor_config["camera"].mount.target_kind == "world"
    _check_plane_returns(sim, 2.65 if world else 3.2)
    # A pose update must also see qpos writes that have not yet had an FK refresh. Updating just
    # the camera avoids a second sensor writer accidentally supplying the missing body refresh.
    sim.root_data.qpos[2] += 0.25
    sim.sensor_manager.set_mount_pose("camera", pose)
    _check_plane_returns(sim, 2.65 if world else 3.45)
    for name in ("camera", "lidar"):
        torch.testing.assert_close(sim.sensor_manager.get_mount_pose(name), pose)


def test_rgb_mount_translation_and_rotation_match_projected_landmarks(tmp_path: Path) -> None:
    """Save actual before/after RGB frames and constrain their translation and roll independently."""
    from PIL import Image

    scene = MujocoSceneManager.__new__(MujocoSceneManager)
    scene.world_spec = mujoco.MjSpec.from_string(
        """
        <mujoco>
          <visual><global offwidth="640" offheight="480"/></visual>
          <asset>
            <texture name="checker" type="2d" builtin="checker" width="512" height="512"
                     rgb1="0.82 0.82 0.78" rgb2="0.38 0.40 0.42"/>
            <material name="floor" texture="checker" texrepeat="12 12" reflectance="0"/>
          </asset>
          <worldbody>
            <light pos="0 0 5" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
            <geom type="plane" size="4 4 0.1" material="floor"/>
            <geom name="red_landmark" type="box" pos="-0.6 0.35 0.03"
                  size="0.28 0.10 0.01" rgba="0.95 0.04 0.03 1"/>
            <geom type="box" pos="0.7 -0.4 0.15" size="0.16 0.22 0.15" rgba="0.03 0.18 0.95 1"/>
            <geom type="cylinder" pos="0.65 0.65 0.10" size="0.16 0.10" rgba="0.95 0.8 0.02 1"/>
            <geom type="box" pos="-0.45 -0.65 0.10" size="0.12 0.20 0.10" rgba="0.02 0.8 0.12 1"/>
          </worldbody>
        </mujoco>
        """
    )
    camera = CameraSensorConfig(
        mount=SensorMountConfig(target_kind="world", position=[0.0, 0.0, 3.0]),
        width=640,
        height=480,
        vertical_fov=55.0,
        data_types=["rgb"],
    )
    scene.add_cameras({"camera": camera})
    instance = _make_sim(scene.world_spec.compile(), {"camera": camera}, torch.zeros(1, 3))
    observations = []
    try:
        pose = torch.tensor([0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0])
        for label, x, angle_deg in (("before", 0.0, 0.0), ("translated", 0.6, 0.0), ("after", 0.6, 25.0)):
            angle = math.radians(angle_deg)
            pose[0] = x
            pose[3:] = torch.tensor([0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2)])
            before_time = instance.time()
            instance.sensor_manager.set_mount_pose("camera", pose)
            assert instance.time() == before_time
            instance.render_sensors()
            rgb = instance.get_camera_data("camera")[0].numpy()
            Image.fromarray(rgb).save(tmp_path / f"{label}.png")

            # The red box's top is at z=0.04. Project its center with an independent pinhole
            # calculation, and measure the long axis from actual red pixels.
            red = (rgb[:, :, 0].astype(int) > rgb[:, :, 1].astype(int) + 40) & (
                rgb[:, :, 0].astype(int) > rgb[:, :, 2].astype(int) + 40
            )
            rows, cols = np.nonzero(red)
            assert len(rows) > 100, f"{label}: landmark is not visible"
            centroid = np.array([cols.mean(), rows.mean()])
            c, s = math.cos(angle), math.sin(angle)
            local_xy = np.array([[c, s], [-s, c]]) @ np.array([-0.6 - x, 0.35])
            focal = camera.height / (2 * math.tan(math.radians(camera.vertical_fov) / 2))
            expected = np.array([(camera.width - 1) / 2, (camera.height - 1) / 2])
            expected += focal * local_xy * np.array([1, -1]) / (3.0 - 0.04)
            np.testing.assert_allclose(centroid, expected, atol=2.0)
            _, axes = np.linalg.eigh(np.cov(np.stack([cols, rows])))
            axis = axes[:, -1]
            observed_angle = (math.degrees(math.atan2(axis[1], axis[0])) + 90) % 180 - 90
            assert observed_angle == pytest.approx(angle_deg, abs=1.5)
            observations.append(
                {
                    "frame": f"{label}.png",
                    "pose_xyzw": pose.tolist(),
                    "landmark_centroid_px": centroid.tolist(),
                    "expected_centroid_px": expected.tolist(),
                    "landmark_angle_deg": observed_angle,
                    "expected_angle_deg": angle_deg,
                }
            )
            print(f"{label}: pose={pose.tolist()}, centroid={centroid.tolist()}, roll={observed_angle:.2f} deg")
        (tmp_path / "poses-and-measurements.json").write_text(json.dumps(observations, indent=2) + "\n")
    finally:
        instance.backend.close()
