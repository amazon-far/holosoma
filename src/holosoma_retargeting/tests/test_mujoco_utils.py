"""Tests for robot-generic MuJoCo frame resolution."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest
from holosoma_retargeting.src.mujoco_utils import (
    mujoco_frame_jacobians,
    mujoco_frame_pose,
    resolve_mujoco_frame,
)
from scipy.spatial.transform import Rotation

import holosoma_retargeting

MODEL_DIR = Path(holosoma_retargeting.__file__).parent / "models" / "g1"


def _frame_test_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="shared" pos="0.1 0.2 0.3">
              <geom name="shared" type="sphere" size="0.01" pos="0.2 0 0"/>
              <site name="shared" pos="0.4 0 0"/>
              <geom name="geom_only" type="sphere" size="0.01" pos="0 0.2 0"/>
              <site name="site_only" pos="0 0 0.3"/>
              <body name="body_only" pos="0 0.4 0"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )


def test_geom_distance_uses_cutoff_as_out_of_range_sentinel() -> None:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <geom name="ground" type="plane" size="1 1 0.1"/>
            <body pos="0 0 1">
              <geom name="ball" type="sphere" size="0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ground_id = model.geom("ground").id
    ball_id = model.geom("ball").id

    fromto = np.full(6, np.nan)
    cutoff = 0.1
    distance = mujoco.mj_geomDistance(model, data, ground_id, ball_id, cutoff, fromto)

    assert distance == pytest.approx(cutoff)
    np.testing.assert_array_equal(fromto, np.zeros(6))

    in_range_cutoff = 1.0
    distance = mujoco.mj_geomDistance(model, data, ground_id, ball_id, in_range_cutoff, fromto)

    assert distance == pytest.approx(0.9)
    assert distance < in_range_cutoff
    assert np.linalg.norm(fromto[3:] - fromto[:3]) == pytest.approx(distance)


def test_resolve_mujoco_frame_supports_all_kinds_and_precedence() -> None:
    model = _frame_test_model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    assert resolve_mujoco_frame(model, "shared").kind == "body"
    assert resolve_mujoco_frame(model, "body_only").kind == "body"
    assert resolve_mujoco_frame(model, "geom_only").kind == "geom"
    site_ref = resolve_mujoco_frame(model, "site_only")
    assert site_ref.kind == "site"
    assert site_ref.body_id == resolve_mujoco_frame(model, "shared").body_id

    site_position, site_rotation = mujoco_frame_pose(model, data, site_ref)
    np.testing.assert_allclose(site_position, [0.1, 0.2, 0.6])
    np.testing.assert_allclose(site_rotation, np.eye(3))


def test_resolve_mujoco_frame_reports_all_supported_kinds() -> None:
    with pytest.raises(KeyError, match="body, geom, or site"):
        resolve_mujoco_frame(_frame_test_model(), "missing")


@pytest.mark.parametrize("side", ["left", "right"])
def test_metatarsal_site_jacobian_matches_finite_difference(side: str) -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_DIR / "g1_29dof.xml"))
    data = mujoco.MjData(model)
    qpos = model.qpos0.copy()
    qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)

    site_ref = resolve_mujoco_frame(model, f"{side}_ankle_roll_metatarsal_site")
    jacp, _jacr, position, _rotation = mujoco_frame_jacobians(model, data, site_ref)
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_ankle_roll_joint")
    qpos_index = int(model.jnt_qposadr[joint_id])
    dof_index = int(model.jnt_dofadr[joint_id])

    epsilon = 1e-7
    data.qpos[:] = qpos
    data.qpos[qpos_index] += epsilon
    mujoco.mj_forward(model, data)
    displaced, _ = mujoco_frame_pose(model, data, site_ref)

    np.testing.assert_allclose(jacp[:, dof_index], (displaced - position) / epsilon, atol=1e-7)


@pytest.mark.parametrize("model_name", ["g1_29dof.xml", "g1_29dof_spherehand.xml"])
def test_metatarsal_sites_follow_contact_sphere_midpoints(model_name: str) -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_DIR / model_name))
    data = mujoco.MjData(model)
    rng = np.random.default_rng(42)

    for _ in range(10):
        qpos = model.qpos0.copy()
        qpos[:3] += rng.uniform(-0.2, 0.2, size=3)
        quaternion_xyzw = Rotation.random(random_state=rng).as_quat()
        qpos[3:7] = quaternion_xyzw[[3, 0, 1, 2]]
        for joint_id in range(model.njnt):
            if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
                continue
            qpos_index = int(model.jnt_qposadr[joint_id])
            qpos[qpos_index] = rng.uniform(*model.jnt_range[joint_id])
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)

        for side in ("left", "right"):
            site_position, _ = mujoco_frame_pose(model, data, f"{side}_ankle_roll_metatarsal_site")
            sphere_positions = [
                mujoco_frame_pose(model, data, f"{side}_ankle_roll_sphere_{index}_link")[0]
                for index in (3, 4)
            ]
            np.testing.assert_allclose(site_position, np.mean(sphere_positions, axis=0), atol=1e-12)
