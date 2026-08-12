"""Tests for Xsens avatar mesh construction."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace

import numpy as np
from holosoma_retargeting.kinematics.model import rotate_vector
from holosoma_retargeting.xsens.avatar_mesh import (
    build_tennis_racket_meshes,
    build_xsens_avatar_meshes,
    expected_racket_tpose_from_right_hand,
    load_xsens_avatar_proportions,
    validate_avatar_mesh_parts,
)
from holosoma_retargeting.xsens.kinematic_model import XSENS_JOINT_SPECS, XSENS_RACKET_SOURCE_SEGMENT


def _write_avatar_hdf5(hdf5_path, *, scale: float = 1.0) -> tuple[str, ...]:
    h5py = __import__("h5py")
    names = [
        "Pelvis",
        "L5",
        "L3",
        "T12",
        "T8",
        "Neck",
        "Head",
        "RightShoulder",
        "RightUpperArm",
        "RightForeArm",
        "RightHand",
        "LeftShoulder",
        "LeftUpperArm",
        "LeftForeArm",
        "LeftHand",
        "RightUpperLeg",
        "RightLowerLeg",
        "RightFoot",
        "RightToe",
        "LeftUpperLeg",
        "LeftLowerLeg",
        "LeftFoot",
        "LeftToe",
        XSENS_RACKET_SOURCE_SEGMENT,
    ]
    # Use the real metadata schema, but compact synthetic dimensions.
    endpoint_names = {
        "L5": "jL4L3",
        "L3": "jL1T12",
        "T12": "jT9T8",
        "T8": "jT1C7",
        "Neck": "jC1Head",
        "RightShoulder": "jRightShoulder",
        "RightUpperArm": "jRightElbow",
        "RightForeArm": "jRightWrist",
        "RightHand": "pRightTopOfHand",
        "LeftShoulder": "jLeftShoulder",
        "LeftUpperArm": "jLeftElbow",
        "LeftForeArm": "jLeftWrist",
        "LeftHand": "pLeftTopOfHand",
        "RightUpperLeg": "jRightKnee",
        "RightLowerLeg": "jRightAnkle",
        "RightFoot": "jRightBallFoot",
        "RightToe": "pRightToe",
        "LeftUpperLeg": "jLeftKnee",
        "LeftLowerLeg": "jLeftAnkle",
        "LeftFoot": "jLeftBallFoot",
        "LeftToe": "pLeftToe",
    }
    metadata = []
    for name in names:
        values = [("origin", [0.0, 0.0, 0.0])]
        if name in endpoint_names:
            axis = [0.0, 20.0, 0.0] if "Arm" in name or "Hand" in name or "Shoulder" in name else [0.0, 0.0, 20.0]
            if name.startswith("Right") and axis[1] != 0.0:
                axis[1] *= -1.0
            values.append((endpoint_names[name], axis))
        metadata.append((name, values))
    metadata[0] = (
        "Pelvis",
        [
            ("pHipOrigin", [0.0, 0.0, 0.0]),
            ("jRightHip", [0.0, -8.0, 0.0]),
            ("jLeftHip", [0.0, 8.0, 0.0]),
            ("jL5S1", [0.0, 0.0, 10.0]),
            ("front", [6.0, 0.0, 5.0]),
            ("back", [-6.0, 0.0, 5.0]),
        ],
    )
    metadata[names.index("Head")] = (
        "Head",
        [
            ("jC1Head", [0.0, 0.0, 0.0]),
            ("pTopOfHead", [0.0, 0.0, 18.0]),
            ("pRightAuricularis", [0.0, -6.0, 8.0]),
            ("pLeftAuricularis", [0.0, 6.0, 8.0]),
            ("pBackOfHead", [-8.0, 0.0, 8.0]),
        ],
    )
    for side in ("Right", "Left"):
        hand = f"{side}Hand"
        metadata[names.index(hand)] = (
            hand,
            [
                ("origin", [0.0, 0.0, 0.0]),
                (f"p{side}TopOfHand", [0.0, -18.0 if side == "Right" else 18.0, 0.0]),
                (f"p{side}Pinky", [-4.0, -11.0 if side == "Right" else 11.0, 0.0]),
                (f"p{side}HandPalm", [2.0, -9.0 if side == "Right" else 9.0, -1.0]),
            ],
        )
        upper_leg = f"{side}UpperLeg"
        lateral_sign = -1.0 if side == "Right" else 1.0
        metadata[names.index(upper_leg)] = (
            upper_leg,
            [
                ("origin", [0.0, 0.0, 0.0]),
                (f"j{side}Knee", [0.0, 0.0, -42.0]),
                (f"p{side}GreaterTrochanter", [0.0, lateral_sign * 7.0, -3.0]),
                (f"p{side}KneeLatEpicondyle", [-1.0, lateral_sign * 4.0, -42.0]),
                (f"p{side}KneeMedEpicondyle", [-1.0, -lateral_sign * 3.6, -42.0]),
                (f"p{side}Patella", [3.8, 0.0, -41.8]),
            ],
        )
        lower_leg = f"{side}LowerLeg"
        metadata[names.index(lower_leg)] = (
            lower_leg,
            [
                ("origin", [0.0, 0.0, 0.0]),
                (f"j{side}Ankle", [0.0, 0.0, -40.0]),
                (f"p{side}LatMalleolus", [0.0, lateral_sign * 3.0, -40.0]),
                (f"p{side}MedMalleolus", [0.0, -lateral_sign * 3.0, -40.0]),
                (f"p{side}TibialTub", [4.8, 0.0, -4.0]),
                (f"p{side}Fibula", [0.0, lateral_sign * 3.2, -27.0]),
            ],
        )
        foot = f"{side}Foot"
        metadata[names.index(foot)] = (
            foot,
            [
                ("origin", [0.0, 0.0, 0.0]),
                (f"j{side}BallFoot", [18.0, 0.0, -7.0]),
                (f"p{side}HeelCenter", [-4.0, 0.0, -9.0]),
                (f"p{side}FirstMetatarsal", [13.0, 4.2, -9.0]),
                (f"p{side}FifthMetatarsal", [13.0, -4.2, -9.0]),
                (f"p{side}TopOfFoot", [12.0, 0.0, -2.0]),
            ],
        )
        toe = f"{side}Toe"
        metadata[names.index(toe)] = (
            toe,
            [("origin", [0.0, 0.0, 0.0]), (f"p{side}Toe", [7.0, 0.0, -1.5])],
        )

    metadata_by_segment = OrderedDict((name, OrderedDict(values)) for name, values in metadata)
    for spec in XSENS_JOINT_SPECS[:-1]:
        metadata_by_segment[spec.parent_segment].setdefault(spec.landmark, [0.0, 0.0, 0.0])
        metadata_by_segment[spec.child_segment].setdefault(spec.landmark, [0.0, 0.0, 0.0])
    scaled_metadata = OrderedDict(
        (
            name,
            OrderedDict(
                (point_name, (np.asarray(value) * scale).tolist())
                for point_name, value in values.items()
            ),
        )
        for name, values in metadata_by_segment.items()
    )
    metadata_text = repr(scaled_metadata)
    positions = np.zeros((len(names), 3))
    for spec in XSENS_JOINT_SPECS[:-1]:
        parent_position = positions[names.index(spec.parent_segment)]
        parent_anchor = np.asarray(scaled_metadata[spec.parent_segment][spec.landmark]) / 100.0
        child_anchor = np.asarray(scaled_metadata[spec.child_segment][spec.landmark]) / 100.0
        positions[names.index(spec.child_segment)] = parent_position + parent_anchor - child_anchor
    hand_position = positions[names.index("RightHand")]
    hand_palm = np.asarray(scaled_metadata["RightHand"]["pRightHandPalm"]) / 100.0
    positions[names.index(XSENS_RACKET_SOURCE_SEGMENT)] = hand_position + hand_palm
    orientations = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (len(names), 1))
    orientations[names.index(XSENS_RACKET_SOURCE_SEGMENT)] = np.array(
        [np.sqrt(0.5), -np.sqrt(0.5), 0.0, 0.0]
    )
    with h5py.File(hdf5_path, "w") as hdf5_file:
        stream = hdf5_file.require_group("xsens-segments").create_group("body_position_xyz_m")
        stream.attrs["segment_names_body"] = repr(names)
        stream.attrs["segment_mesh_points_body_xyz_cm"] = metadata_text
        joint_stream = hdf5_file.require_group("xsens-joints").create_group(
            "body_joint_angles_eulerZXY_xyz_rad"
        )
        joint_names = [spec.source_joint for spec in XSENS_JOINT_SPECS]
        joint_stream.attrs["joint_names_body"] = repr(joint_names)
        joint_stream.attrs["joint_rotation_order_body"] = repr(
            [(name, ("x", "y", "z")) for name in joint_names]
        )
        tpose = hdf5_file.create_group("xsens-segments-tpose")
        tpose.create_dataset("body_position_Tpose_xyz_m", data=positions)
        tpose.create_dataset(
            "body_orientation_Tpose_quaternion_wijk",
            data=orientations,
        )
    return tuple(names)


def test_load_subject_proportions_and_generate_all_body_segments(tmp_path) -> None:
    hdf5_path = tmp_path / "avatar.hdf5"
    names = _write_avatar_hdf5(hdf5_path)

    proportions = load_xsens_avatar_proportions(hdf5_path)
    parts = build_xsens_avatar_meshes(proportions)
    validate_avatar_mesh_parts(parts)

    assert proportions.segment_names == names
    assert set(parts) == set(names)
    assert all(parts[name] for name in names if name != XSENS_RACKET_SOURCE_SEGMENT)

    longer_landmarks = {
        segment: {name: value.copy() for name, value in values.items()}
        for segment, values in proportions.landmarks_m.items()
    }
    longer_landmarks["RightUpperArm"]["jRightElbow"] = np.array([0.0, -0.35, 0.0])
    longer_parts = build_xsens_avatar_meshes(replace(proportions, landmarks_m=longer_landmarks))
    original_arm = next(part.mesh for part in parts["RightUpperArm"] if part.name.endswith("_shell"))
    longer_arm = next(part.mesh for part in longer_parts["RightUpperArm"] if part.name.endswith("_shell"))
    assert longer_arm.extents[1] > original_arm.extents[1] + 0.1

    for side in ("Right", "Left"):
        foot_shell = next(part.mesh for part in parts[f"{side}Foot"] if part.name.endswith("_shoe"))
        toe_shell = next(part.mesh for part in parts[f"{side}Toe"] if part.name.endswith("_toe_box"))
        assert foot_shell.extents[0] > foot_shell.extents[2]
        assert foot_shell.extents[1] > 0.07
        assert toe_shell.bounds[1, 0] > 0.06
        assert any(part.name.endswith("_outsole") for part in parts[f"{side}Foot"])
        hand_parts = parts[f"{side}Hand"]
        assert len([part for part in hand_parts if "_finger_" in part.name and "_joint_" not in part.name]) == 4
        assert len([part for part in hand_parts if "_finger_joint_" in part.name]) == 4
        fingernails = [part.mesh for part in hand_parts if "_fingernail_" in part.name]
        assert len(fingernails) == 4
        assert all(mesh.bounds[0, 2] > 0.0 for mesh in fingernails)
        dorsal_panel = next(part.mesh for part in hand_parts if part.name.endswith("_dorsal_panel"))
        palm_pad = next(part.mesh for part in hand_parts if part.name.endswith("_palm_pad"))
        assert dorsal_panel.centroid[2] > 0.0
        assert palm_pad.centroid[2] < 0.0
        thumb = next(part.mesh for part in hand_parts if part.name.endswith("_thumb"))
        thumbnail = next(part.mesh for part in hand_parts if part.name.endswith("_thumbnail"))
        assert thumb.bounds[1, 0] > 0.06
        assert thumbnail.centroid[2] > thumb.centroid[2]
        upper_leg = next(part.mesh for part in parts[f"{side}UpperLeg"] if part.name.endswith("_shell"))
        lower_leg = next(part.mesh for part in parts[f"{side}LowerLeg"] if part.name.endswith("_shell"))
        assert upper_leg.extents[1] < 0.13
        assert lower_leg.extents[1] < 0.09

    racket_position, racket_orientation = expected_racket_tpose_from_right_hand(proportions)
    np.testing.assert_allclose(racket_orientation, np.array([np.sqrt(0.5), -np.sqrt(0.5), 0.0, 0.0]))

    racket_index = proportions.segment_index(XSENS_RACKET_SOURCE_SEGMENT)
    np.testing.assert_allclose(proportions.tpose_positions_m[racket_index], racket_position)
    np.testing.assert_allclose(proportions.tpose_quaternions_wijk[racket_index], racket_orientation)

    racket_strings = next(part.mesh for part in build_tennis_racket_meshes() if part.name == "racket_strings")
    # The mesh is inverse-rolled in the recorded sword frame. The recorded
    # -90-degree frame rotation therefore places its string plane in world XY.
    assert racket_strings.extents[1] < 0.01
    assert racket_strings.extents[2] > 0.24
    string_normal_world = rotate_vector(racket_orientation, np.array([0.0, 1.0, 0.0]))
    np.testing.assert_allclose(np.abs(string_normal_world), np.array([0.0, 0.0, 1.0]), atol=1e-12)


def _part(parts, segment: str, suffix: str):
    return next(part.mesh for part in parts[segment] if part.name.endswith(suffix))


def test_visual_mesh_dimensions_follow_subject_proportions(tmp_path) -> None:
    small_path = tmp_path / "small_subject.hdf5"
    base_path = tmp_path / "base_subject.hdf5"
    large_path = tmp_path / "large_subject.hdf5"
    _write_avatar_hdf5(small_path, scale=0.8)
    _write_avatar_hdf5(base_path, scale=1.0)
    _write_avatar_hdf5(large_path, scale=1.25)

    small = build_xsens_avatar_meshes(load_xsens_avatar_proportions(small_path))
    base = build_xsens_avatar_meshes(load_xsens_avatar_proportions(base_path))
    large = build_xsens_avatar_meshes(load_xsens_avatar_proportions(large_path))

    # Link length, transverse body width, foot dimensions, hand length, and
    # head height must all respond to the corresponding calibrated landmarks.
    measurements = (
        (
            _part(small, "RightUpperArm", "_shell").extents[1],
            _part(base, "RightUpperArm", "_shell").extents[1],
            _part(large, "RightUpperArm", "_shell").extents[1],
        ),
        (
            _part(small, "RightUpperLeg", "_shell").extents[2],
            _part(base, "RightUpperLeg", "_shell").extents[2],
            _part(large, "RightUpperLeg", "_shell").extents[2],
        ),
        (
            _part(small, "Pelvis", "_shell").extents[1],
            _part(base, "Pelvis", "_shell").extents[1],
            _part(large, "Pelvis", "_shell").extents[1],
        ),
        (
            _part(small, "RightFoot", "_shoe").extents[0],
            _part(base, "RightFoot", "_shoe").extents[0],
            _part(large, "RightFoot", "_shoe").extents[0],
        ),
        (
            _part(small, "RightFoot", "_shoe").extents[1],
            _part(base, "RightFoot", "_shoe").extents[1],
            _part(large, "RightFoot", "_shoe").extents[1],
        ),
        (
            _part(small, "RightHand", "_palm").extents[1],
            _part(base, "RightHand", "_palm").extents[1],
            _part(large, "RightHand", "_palm").extents[1],
        ),
        (
            _part(small, "Head", "head_shell").extents[2],
            _part(base, "Head", "head_shell").extents[2],
            _part(large, "Head", "head_shell").extents[2],
        ),
    )
    for small_dimension, base_dimension, large_dimension in measurements:
        assert small_dimension < base_dimension * 0.85
        assert large_dimension > base_dimension * 1.15


def test_racket_is_regulation_sized_and_valid() -> None:
    racket_parts = build_tennis_racket_meshes()
    vertices = np.concatenate([part.mesh.vertices for part in racket_parts], axis=0)

    assert {part.name for part in racket_parts} == {"racket_grip", "racket_frame", "racket_strings"}
    assert 0.66 < float(vertices[:, 0].max() - vertices[:, 0].min()) < 0.71
    assert 0.25 < float(vertices[:, 2].max() - vertices[:, 2].min()) < 0.30
    grip = next(part.mesh for part in racket_parts if part.name == "racket_grip")
    assert grip.bounds[0, 0] < 0.0 < grip.bounds[1, 0]
    assert all(np.isfinite(part.mesh.vertices).all() for part in racket_parts)
