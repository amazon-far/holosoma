"""Tests for Xsens-to-USD conversion."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace

import h5py
import numpy as np
import pytest

pytest.importorskip("pxr")

from holosoma_retargeting.data_utils.xsens_hdf5 import SegmentPoseSet, XsensHdf5Calibration
from holosoma_retargeting.usd import open_usd_stage, read_kinematic_tree_from_stage
from holosoma_retargeting.xsens.usd_conversion import convert_xsens_hdf5_to_usd

from .test_avatar_mesh import _write_avatar_hdf5
from .test_kinematic_model import synthetic_calibration


def _write_hdf5(path, calibration: XsensHdf5Calibration) -> None:
    landmarks_cm = OrderedDict(
        (
            segment,
            OrderedDict(
                (name, (np.asarray(value) * 100.0).tolist())
                for name, value in calibration.landmarks_m[segment].items()
            ),
        )
        for segment in calibration.segment_names
    )
    with h5py.File(path, "w") as hdf5_file:
        stream = hdf5_file.require_group("xsens-segments").create_group("body_position_xyz_m")
        stream.attrs["segment_names_body"] = repr(list(calibration.segment_names))
        stream.attrs["segment_mesh_points_body_xyz_cm"] = repr(landmarks_cm)
        stream.attrs["mvn_version"] = calibration.mvn_version or ""
        stream.attrs["mvnx_version"] = calibration.mvnx_version or ""
        poses = hdf5_file.create_group("xsens-segments-tpose")
        poses.create_dataset("body_position_Tpose_xyz_m", data=calibration.tpose.positions_m)
        poses.create_dataset("body_orientation_Tpose_quaternion_wijk", data=calibration.tpose.quaternions_wijk)
        joints = hdf5_file.create_group("xsens-joints")
        joint_stream = joints.create_group("body_joint_angles_eulerZXY_xyz_rad")
        joint_stream.attrs["joint_names_body"] = repr(list(calibration.joint_names))
        joint_stream.attrs["joint_rotation_order_body"] = repr(
            [(name, calibration.joint_rotation_metadata[name].components) for name in calibration.joint_names]
        )


def test_each_hdf5_produces_an_independent_canonical_usd(tmp_path) -> None:
    first_calibration = synthetic_calibration()
    shifted_pose = SegmentPoseSet(
        first_calibration.tpose.positions_m + np.array([0.01, 0.0, 0.0]),
        first_calibration.tpose.quaternions_wijk,
        "Tpose",
    )
    second_calibration = replace(first_calibration, tpose=shifted_pose)
    first_hdf5 = tmp_path / "sequence_01.hdf5"
    second_hdf5 = tmp_path / "sequence_02.hdf5"
    _write_hdf5(first_hdf5, first_calibration)
    _write_hdf5(second_hdf5, second_calibration)

    first = convert_xsens_hdf5_to_usd(first_hdf5, include_visuals=False)
    second = convert_xsens_hdf5_to_usd(second_hdf5, include_visuals=False)

    assert first.output_path.name == "sequence_01_xsens_model.usda"
    assert second.output_path.name == "sequence_02_xsens_model.usda"
    assert first.output_path != second.output_path
    assert first.calibration_fingerprint != second.calibration_fingerprint
    assert first.body_count == second.body_count == 24
    assert first.joint_count == second.joint_count == 23
    model = read_kinematic_tree_from_stage(open_usd_stage(first.output_path))
    assert model.bodies[-1].name == "TennisRacket"
    assert model.joints[-1].name == "RightHandTennisRacketOrigin"
    assert all("Sword" not in body.name for body in model.bodies)
    assert all("Sword" not in joint.name for joint in model.joints)


def test_refined_avatar_visuals_are_exported_and_round_trip(tmp_path) -> None:
    hdf5_path = tmp_path / "visual_subject.hdf5"
    output_path = tmp_path / "visual_subject.usda"
    _write_avatar_hdf5(hdf5_path)

    convert_xsens_hdf5_to_usd(hdf5_path, output_path)
    stage = open_usd_stage(output_path)
    model = read_kinematic_tree_from_stage(stage)
    body_map = model.body_map()

    assert {mesh.name for mesh in body_map["RightHand"].meshes} >= {
        "righthand_palm",
        "righthand_dorsal_panel",
        "righthand_palm_pad",
        "righthand_finger_1",
        "righthand_fingernail_1",
        "righthand_finger_joint_1",
        "righthand_thumb",
        "righthand_thumbnail",
    }
    orientation_cues = [mesh for mesh in body_map["RightHand"].meshes if mesh.category == "orientation_cue"]
    assert len(orientation_cues) == 7
    assert {mesh.name for mesh in body_map["RightUpperLeg"].meshes} == {
        "rightupperleg_shell",
        "rightupperleg_outer_stripe",
        "rightupperleg_joint_collar",
    }
    assert {mesh.name for mesh in body_map["RightFoot"].meshes} >= {
        "rightfoot_shoe",
        "rightfoot_outsole",
    }
    assert {mesh.name for mesh in body_map["TennisRacket"].meshes} == {
        "racket_grip",
        "racket_frame",
        "racket_strings",
    }
    assert stage.GetPrimAtPath("/XsensAvatar/Looks").IsValid()
    assert stage.GetPrimAtPath("/XsensAvatar/Looks/racket_49_56_59").IsValid()
