"""Unit tests for retargeting constraint activation."""

from __future__ import annotations

from unittest.mock import Mock

import cvxpy as cp
import mujoco
import numpy as np
import pytest
from holosoma_retargeting.config_types.retargeter import FootLockConfig
from holosoma_retargeting.src.interaction_mesh_retargeter import InteractionMeshRetargeter


def _retargeter_with_non_penetration(enabled: bool) -> InteractionMeshRetargeter:
    retargeter = object.__new__(InteractionMeshRetargeter)
    retargeter.activate_obj_non_penetration = enabled
    retargeter.q_a_indices = np.array([0, 1], dtype=int)
    retargeter.penetration_tolerance = 1e-3
    return retargeter


def _retargeter_with_collision_pair() -> tuple[InteractionMeshRetargeter, tuple[int, int]]:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <geom name="ground" type="plane" size="1 1 0.1"/>
            <body name="robot_body" pos="0 0 1">
              <geom name="robot_collision" type="sphere" size="0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    ground_id = model.geom("ground").id
    robot_id = model.geom("robot_collision").id
    pair = (ground_id, robot_id)

    retargeter = object.__new__(InteractionMeshRetargeter)
    retargeter.robot_model = model
    retargeter.robot_data = mujoco.MjData(model)
    retargeter.collision_detection_threshold = 0.1
    retargeter.object_name = "tracked_object"
    retargeter._geom_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "" for geom_id in range(model.ngeom)
    ]
    retargeter._prefilter_pairs_with_mj_collision = lambda _threshold: {pair}
    retargeter._compute_jacobian_for_contact_relative = Mock(return_value=np.zeros(model.nv))
    retargeter._self_collision_enabled = True
    retargeter._self_collision_windows = None
    retargeter._self_collision_geom_pairs = [pair]
    retargeter._sc_last_vis_frame = -1
    retargeter.visualize = False
    return retargeter, pair


def test_environment_non_penetration_constraints_are_disabled_by_config() -> None:
    retargeter = _retargeter_with_non_penetration(False)

    constraints = retargeter._environment_non_penetration_constraints(
        cp.Variable(2),
        {"ground-contact": np.zeros(2)},
        {"ground-contact": -0.1},
    )

    assert constraints == []


def test_environment_non_penetration_constraints_are_enabled_by_default() -> None:
    retargeter = _retargeter_with_non_penetration(True)

    constraints = retargeter._environment_non_penetration_constraints(
        cp.Variable(2),
        {"ground-contact": np.zeros(2)},
        {"ground-contact": -0.1},
    )

    assert len(constraints) == 1


def test_foot_lock_windows_return_per_window_floor_height() -> None:
    retargeter = object.__new__(InteractionMeshRetargeter)
    retargeter._init_foot_lock(
        FootLockConfig(
            enable=True,
            windows={
                "L_Toe": [(10, 20, 0.12), (30, 40)],
                "R_Toe": [(15, 25, 0.56)],
            },
            z_floor=0.34,
        )
    )

    assert retargeter._is_foot_locked_in_window("left_ankle_link", 15) == pytest.approx(0.12)
    assert retargeter._is_foot_locked_in_window("left_ankle_link", 35) == pytest.approx(0.34)
    assert retargeter._is_foot_locked_in_window("right_ankle_link", 20) == pytest.approx(0.56)
    assert retargeter._is_foot_locked_in_window("left_ankle_link", 25) is None
    assert retargeter._is_foot_locked_in_window("torso_link", 15) is None


def test_collision_queries_reject_cutoff_equal_results(monkeypatch: pytest.MonkeyPatch) -> None:
    retargeter, _pair = _retargeter_with_collision_pair()

    def cutoff_result(_model, _data, _geom_a, _geom_b, cutoff, fromto):
        fromto[:] = 0.0
        return cutoff

    monkeypatch.setattr(mujoco, "mj_geomDistance", cutoff_result)

    environment_js, environment_phis = retargeter._update_jacobians_and_phis_from_q(np.zeros(retargeter.robot_model.nq))
    self_js, self_phis = retargeter._compute_self_collision_constraints(frame_idx=0)

    assert environment_js == {}
    assert environment_phis == {}
    assert self_js == {}
    assert self_phis == {}
    retargeter._compute_jacobian_for_contact_relative.assert_not_called()


def test_collision_queries_accept_results_strictly_below_cutoff(monkeypatch: pytest.MonkeyPatch) -> None:
    retargeter, pair = _retargeter_with_collision_pair()
    distance = retargeter.collision_detection_threshold - 0.01

    def in_range_result(_model, _data, _geom_a, _geom_b, _cutoff, fromto):
        fromto[:] = [0.0, 0.0, 0.0, 0.0, 0.0, distance]
        return distance

    monkeypatch.setattr(mujoco, "mj_geomDistance", in_range_result)

    environment_js, environment_phis = retargeter._update_jacobians_and_phis_from_q(np.zeros(retargeter.robot_model.nq))
    self_js, self_phis = retargeter._compute_self_collision_constraints(frame_idx=0)

    assert set(environment_js) == {pair}
    assert environment_phis[pair] == pytest.approx(distance)
    self_key = ("self", *pair)
    assert set(self_js) == {self_key}
    assert self_phis[self_key] == pytest.approx(distance)
    assert retargeter._compute_jacobian_for_contact_relative.call_count == 2
