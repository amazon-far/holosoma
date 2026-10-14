"""Engine regressions for robot property overrides, fixed-root poses, and named sensor mounts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

torch = pytest.importorskip("torch")
mujoco = pytest.importorskip("mujoco")

from holosoma.config_types.robot import RobotConfig, RobotInitState  # noqa: E402
from holosoma.config_types.scene import RigidObjectConfig, SceneConfig  # noqa: E402
from holosoma.config_types.sensor import (  # noqa: E402
    CameraSensorConfig,
    LidarBodyFilterConfig,
    LidarSensorConfig,
    SensorMountConfig,
)
from holosoma.config_values.run_sim import mujoco as mujoco_preset  # noqa: E402
from holosoma.simulator.mujoco.scene_manager import (  # noqa: E402
    LIDAR_FILTER_SITE_PREFIX,
    LIDAR_MOUNT_SITE_PREFIX,
    MujocoSceneManager,
)
from tests.simulators._robot_presets import test_arm_fixed as arm_preset  # noqa: E402
from tests.simulators.mujoco._build import build_classic_sim  # noqa: E402

if TYPE_CHECKING:
    from holosoma.config_types.simulator import SimulatorConfig
    from holosoma.managers.terrain.base import TerrainTermBase

pytestmark = pytest.mark.mujoco_classic

_HINGE_XML = """
<mujoco>
  <compiler angle="radian"/>
  <default><geom contype="0" conaffinity="0"/></default>
  <worldbody>
    <body name="base">
      <geom size=".01"/>
      <body name="link">
        <joint name="hinge" axis="0 0 1" range="-1 1" armature="{armature}"/>
        <geom type="capsule" size=".01" fromto="0 0 0 1 0 0" mass="1"/>
      </body>
    </body>
  </worldbody>
  <actuator><motor name="hinge" joint="hinge" ctrlrange="-100 100"/></actuator>
</mujoco>
"""


def _robot(tmp_path: Path, xml: str) -> RobotConfig:
    asset = tmp_path / "hinge.xml"
    asset.write_text(xml)
    return RobotConfig(
        dof_names=["hinge"],
        body_names=["base", "link"],
        init_state=RobotInitState(position=[0.0, 0.0, 2.0], default_joint_positions={"hinge": 0.0}),
        asset=replace(arm_preset.asset, xml_file=str(asset)),
        dof_armature={"hinge": 10.0},
    )


def test_armature_override_matches_authored_constraint_physics(tmp_path: Path) -> None:
    """The public load path must produce the dynamics of an independently compiled asset."""
    robot = _robot(tmp_path, _HINGE_XML.format(armature=0.0))
    sim = build_classic_sim(robot=robot)
    try:
        reference = mujoco.MjModel.from_xml_string(_HINGE_XML.format(armature=10.0))
        reference.opt.timestep = sim.sim_dt
        reference_data = mujoco.MjData(reference)
        qpos_id = sim.dof_qpos_addrs[0]
        qvel_id = sim.dof_qvel_addrs[0]
        sim.root_data.qpos[qpos_id] = reference_data.qpos[0] = 1.1
        mujoco.mj_forward(sim.root_model, sim.root_data)
        mujoco.mj_forward(reference, reference_data)

        assert sim.root_data.ncon == reference_data.ncon == 0
        assert abs(reference_data.qfrc_constraint[0]) > 1.0
        np.testing.assert_allclose(sim.root_data.qfrc_constraint[qvel_id], reference_data.qfrc_constraint[0], rtol=1e-6)
        np.testing.assert_allclose(sim.root_data.qacc[qvel_id], reference_data.qacc[0], rtol=1e-6)
        np.testing.assert_allclose(sim.root_model.dof_invweight0[qvel_id], reference.dof_invweight0[0])
        for _ in range(10):
            sim.simulate_at_each_physics_step()
            mujoco.mj_step(reference, reference_data)
        np.testing.assert_allclose(sim.root_data.qpos[qpos_id], reference_data.qpos[0], rtol=1e-6)
        np.testing.assert_allclose(sim.root_data.qvel[qvel_id], reference_data.qvel[0], rtol=1e-6)
    finally:
        sim.close()


@pytest.mark.parametrize("authored_orientation", ['quat="1 1 0 0"', 'euler="1.5707963267948966 0 0"'])
def test_fixed_base_spawn_overrides_authored_root_orientation(tmp_path: Path, authored_orientation: str) -> None:
    """Root, body and actor reads agree before any reset, which must then preserve the pose."""
    xml = _HINGE_XML.format(armature=10.0).replace(
        '<body name="base">', f'<body name="base" pos=".2 -.3 .1" {authored_orientation}>'
    )
    robot = _robot(tmp_path, xml)
    robot = replace(
        robot,
        body_names=["link", "base"],
        init_state=replace(robot.init_state, position=[1.0, 2.0, 2.0], orientation=[2**-0.5, 0.0, 0.0, 2**-0.5]),
    )
    sim = build_classic_sim(robot=robot)
    try:
        expected = torch.tensor(robot.init_state.root_state_list()).unsqueeze(0)
        env_ids = torch.tensor([0])
        root_id = sim.root_model.body("robot_base").id
        sim.refresh_sim_tensors()
        np.testing.assert_allclose(sim.root_data.xquat[root_id], robot.init_state.orientation, atol=1e-7)
        np.testing.assert_allclose(sim.root_data.xpos[root_id], robot.init_state.position, atol=1e-7)
        torch.testing.assert_close(sim.robot_root_states, expected)
        torch.testing.assert_close(sim.get_actor_states(["robot"], env_ids), expected)
        torch.testing.assert_close(sim.rigid_body_quat_w[:, 1], expected[:, 3:7])
        torch.testing.assert_close(sim.rigid_body_pos_w[:, 1], expected[:, :3])

        poses_before = sim.root_data.xpos.copy(), sim.root_data.xquat.copy()
        sim.set_actor_root_state_tensor_robots(env_ids, sim.robot_root_states)
        np.testing.assert_allclose(sim.root_data.xpos, poses_before[0], atol=1e-7)
        np.testing.assert_allclose(sim.root_data.xquat, poses_before[1], atol=1e-7)
        sim.simulate_at_each_physics_step()
        torch.testing.assert_close(sim.get_actor_states(["robot"], env_ids), expected)
    finally:
        sim.close()


def test_fixed_robot_placement_preserves_static_object_authored_orientation(tmp_path: Path) -> None:
    """A static object's composed asset orientation must survive the fixed robot's placement."""
    asset = tmp_path / "box.xml"
    asset.write_text(
        '<mujoco><worldbody><body name="box" pos=".2 -.3 .1" quat="1 1 0 0">'
        '<geom type="box" size=".1 .2 .3"/></body></worldbody></mujoco>'
    )
    obj = RigidObjectConfig(
        xml_file=str(asset),
        fixed=True,
        position=[4.0, 3.0, 1.0],
        orientation=[2**-0.5, 0.0, 0.0, 2**-0.5],
    )
    sim = build_classic_sim(
        SceneConfig(rigid_objects={"prop": obj}), robot=_robot(tmp_path, _HINGE_XML.format(armature=10.0))
    )
    try:
        body_id = sim.root_model.body("prop_box").id
        geom_id = sim.root_model.body_geomadr[body_id]
        env_ids = torch.tensor([0])
        # Configured yaw followed by authored roll is [w,x,y,z] = [.5,.5,.5,.5].
        expected_quat = np.full(4, 0.5)
        expected_matrix = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        for _ in range(3):
            np.testing.assert_allclose(sim.root_data.xpos[body_id], obj.position, atol=1e-7)
            np.testing.assert_allclose(sim.root_data.xquat[body_id], expected_quat, atol=1e-7)
            np.testing.assert_allclose(sim.root_data.geom_xmat[geom_id].reshape(3, 3), expected_matrix, atol=1e-7)
            actor = sim.get_actor_states(["prop"], env_ids)[0]
            torch.testing.assert_close(actor[:3], torch.tensor(obj.position))
            torch.testing.assert_close(actor[3:7], torch.full((4,), 0.5))
            sim.simulate_at_each_physics_step()
    finally:
        sim.close()


@pytest.mark.parametrize("sensor_kind", ["camera", "lidar_mount", "lidar_filter"])
def test_sensor_names_attach_to_the_physical_link(tmp_path: Path, sensor_kind: str) -> None:
    robot = _robot(tmp_path, _HINGE_XML.format(armature=10.0))
    robot = replace(
        robot,
        body_names=["link", "base"],
        body_tags={},
    )
    manager = MujocoSceneManager(cast("SimulatorConfig", mujoco_preset.config))
    manager.add_robot(cast("TerrainTermBase", SimpleNamespace(geom=None)), robot)
    mount = SensorMountConfig(target="link", position=[0.1, 0.2, 0.3], orientation=[1.0, 0.0, 1.0, 0.0])
    if sensor_kind == "camera":
        manager.add_cameras({"sensor": CameraSensorConfig(mount=mount)})
    elif sensor_kind == "lidar_mount":
        manager.add_lidar_sites({"sensor": LidarSensorConfig(mount=mount)})
    else:
        manager.add_lidar_sites(
            {
                "sensor": LidarSensorConfig(
                    mount=SensorMountConfig(target_kind="world"),
                    body_filter=LidarBodyFilterConfig(target_kind="robot_link", target="link"),
                )
            }
        )
        # The filter marker has an identity transform on the selected body.
        mount = SensorMountConfig(target="link")

    model = manager.compile()
    data = mujoco.MjData(model)
    native_body_id = model.body("robot_link").id
    if sensor_kind == "camera":
        sensor_id = model.camera("sensor").id
        assert model.cam_bodyid[sensor_id] == native_body_id
        positions, matrices = data.cam_xpos, data.cam_xmat
    else:
        prefix = LIDAR_MOUNT_SITE_PREFIX if sensor_kind == "lidar_mount" else LIDAR_FILTER_SITE_PREFIX
        sensor_id = model.site(prefix + "sensor").id
        assert model.site_bodyid[sensor_id] == native_body_id
        positions, matrices = data.site_xpos, data.site_xmat

    local_rotation = np.empty(9)
    mujoco.mju_quat2Mat(local_rotation, np.asarray(mount.orientation))
    for angle in (0.0, 0.4):
        data.qpos[model.joint("robot_hinge").qposadr[0]] = angle
        mujoco.mj_forward(model, data)
        link_rotation = data.xmat[native_body_id].reshape(3, 3)
        np.testing.assert_allclose(
            positions[sensor_id], data.xpos[native_body_id] + link_rotation @ np.asarray(mount.position), atol=1e-7
        )
        np.testing.assert_allclose(
            matrices[sensor_id].reshape(3, 3), link_rotation @ local_rotation.reshape(3, 3), atol=1e-7
        )
