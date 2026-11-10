# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""Native camera depth and LiDAR range checks, including a mount fused into its parent."""

from __future__ import annotations

import math
from collections.abc import Iterator
from types import SimpleNamespace
from typing import cast

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
from holosoma.simulator.mujoco.sensor_setup import create_sensors  # noqa: E402
from holosoma.utils.safe_torch_import import torch  # noqa: E402

pytestmark = pytest.mark.mujoco_classic


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
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    config = cast("FullSimConfig", SimpleNamespace(training=SimpleNamespace(num_envs=1)))
    instance = MuJoCo.__new__(MuJoCo)
    instance.root_model, instance.root_data = model, data
    instance.backend = ClassicBackend(model, data, config, "cpu")
    instance.num_envs, instance.sim_device = 1, "cpu"
    instance.env_origins = torch.tensor([[5.0, 0.0, 0.25]]) if request.param == "world" else torch.zeros(1, 3)
    instance.simulator_config = mujoco_config.config
    instance.sensor_config = {"camera": camera, "lidar": lidar}
    instance.kinematic_playback = False
    create_sensors(instance)
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
    initial = sim.get_sensor_mount_pose("camera")
    initial_height = 2.25 if sim.sensor_config["camera"].mount.target_kind == "world" else 2.3
    for name in ("camera", "lidar"):
        sim.set_sensor_mount_pose(name, initial)
    _check_native_mount_frame(sim, initial[0], 0.0)
    _check_plane_returns(sim, initial_height)
    old_frame = sim.get_camera_data("camera", "depth").clone()
    old_scan = sim.get_lidar_data("lidar", "ranges").clone()
    assert sim.root_data is not None
    qpos, qvel, sim_time = sim.root_data.qpos.copy(), sim.root_data.qvel.copy(), sim.time()
    angle = math.radians(30.0)
    pose = initial[0].clone()
    pose[2] += 0.4
    pose[3:] = torch.tensor([0.0, math.sin(angle / 2), 0.0, math.cos(angle / 2)])
    for name in ("camera", "lidar"):
        sim.set_sensor_mount_pose(name, pose)
        torch.testing.assert_close(sim.get_sensor_mount_pose(name)[0], pose)
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
        torch.testing.assert_close(sim.get_sensor_mount_pose(name)[0], pose)
    for name in ("camera", "lidar"):
        sim.set_sensor_mount_pose(name, initial)
    _check_plane_returns(sim, initial_height)


def test_sensor_keeps_following_its_parent_after_offset_changes(sim: MuJoCo) -> None:
    initial = sim.get_sensor_mount_pose("camera")
    pose = initial.clone()
    pose[:, 2] += 0.4
    for name in ("camera", "lidar"):
        sim.set_sensor_mount_pose(name, pose)
    assert sim.root_data is not None
    sim.root_data.qpos[2] += 0.5
    sim.backend.forward_kinematics()
    world = sim.sensor_config["camera"].mount.target_kind == "world"
    _check_plane_returns(sim, 2.65 if world else 3.2)
    for name in ("camera", "lidar"):
        torch.testing.assert_close(sim.get_sensor_mount_pose(name), pose)
