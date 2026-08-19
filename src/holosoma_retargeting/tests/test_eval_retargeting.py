"""Tests for cutoff handling in retargeting evaluation."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
from holosoma_retargeting.evaluation.eval_retargeting import RetargetingEvaluator
from holosoma_retargeting.src.mujoco_utils import resolve_mujoco_frame


@pytest.mark.parametrize(
    ("distance", "expected_preservation"),
    [(0.1, 0.0), (0.09, 1.0)],
)
def test_terrain_contact_precision_rejects_cutoff_sentinel(
    monkeypatch: pytest.MonkeyPatch,
    distance: float,
    expected_preservation: float,
) -> None:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="robot_body">
              <geom name="robot_collision" type="sphere" size="0.1"/>
            </body>
            <body pos="0 0 1">
              <geom name="multi_boxes_object" type="sphere" size="0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    evaluator = object.__new__(RetargetingEvaluator)
    evaluator.robot_model = model
    evaluator.robot_data = mujoco.MjData(model)
    evaluator.object_name = "multi_boxes"
    evaluator.collision_detection_threshold = 0.1
    evaluator.contact_threshold = 0.1
    evaluator.joints_mapping = {"RightHand": "robot_body"}
    evaluator._frame_refs = {"robot_body": resolve_mujoco_frame(model, "robot_body")}
    evaluator._obj_VW = np.zeros((1, 3))
    evaluator.detect_demo_contact = lambda _joints, _names: ["RightHand"]

    def distance_result(_model, _data, _geom_a, _geom_b, _cutoff, fromto):
        fromto[:] = 0.0
        return distance

    monkeypatch.setattr(mujoco, "mj_geomDistance", distance_result)

    preservation = evaluator.evaluate_terrain_contact_precision(
        human_joints_motion=np.zeros((1, 1, 3)),
        q_trajectory=np.zeros((1, model.nq)),
        joint_names=("RightHand",),
    )

    assert preservation == expected_preservation
