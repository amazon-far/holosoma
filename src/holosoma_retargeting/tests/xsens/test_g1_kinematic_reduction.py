"""Tests for G1 kinematic reduction."""

from __future__ import annotations

import inspect
import json

import mujoco
import numpy as np
import pytest
from holosoma_retargeting.data_utils.xsens_hdf5 import XSENS_BODY_SEGMENT_NAMES, XsensHdf5Motion
from holosoma_retargeting.kinematics import (
    KinematicMotion,
    KinematicPose,
    compute_reference_joint_positions,
    validate_kinematic_tree,
)
from holosoma_retargeting.kinematics.model import rotate_vector
from holosoma_retargeting.xsens import morphology_adaptation
from holosoma_retargeting.xsens.avatar_mesh import build_xsens_avatar_meshes
from holosoma_retargeting.xsens.g1_kinematic_reduction import (
    G1Anthropometry,
    G1XsensReductionConfig,
    _fit_collapsed_shoulder_morphology,
    build_g1_proportioned_xsens_tree,
    export_g1_proportioned_xsens_usd,
    extract_g1_anthropometry,
    g1_anthropometry_to_xsens_avatar_proportions,
)
from holosoma_retargeting.xsens.morphology_adaptation import (
    adapt_xsens_motion_to_g1,
    build_xsens_morphology_adapter,
)


@pytest.fixture(scope="module")
def anthropometry() -> G1Anthropometry:
    return extract_g1_anthropometry()


def _rigid_lengths(model) -> dict[str, float]:
    bodies = model.body_map()
    joints = compute_reference_joint_positions(model)
    return {
        "upper_arm": float(
            np.linalg.norm(
                bodies["LeftForeArm"].reference_pose.translation_m - bodies["LeftUpperArm"].reference_pose.translation_m
            )
        ),
        "thigh": float(
            np.linalg.norm(
                bodies["LeftLowerLeg"].reference_pose.translation_m
                - bodies["LeftUpperLeg"].reference_pose.translation_m
            )
        ),
        "shank": float(np.linalg.norm(joints["LeftAnkle"] - bodies["LeftLowerLeg"].reference_pose.translation_m)),
    }


def test_anthropometry_extraction_has_no_pose_input_or_state_evaluation(monkeypatch) -> None:
    assert "qpos" not in inspect.signature(extract_g1_anthropometry).parameters

    def reject_mjdata(*_args, **_kwargs):
        raise AssertionError("pose-independent extraction must not create MjData")

    monkeypatch.setattr(mujoco, "MjData", reject_mjdata)
    result = extract_g1_anthropometry()

    assert result.lengths_m["upper_arm"] > 0.0
    assert result.lengths_m["forearm"] > 0.0
    assert not np.isclose(result.lengths_m["upper_arm"], result.lengths_m["forearm"])


def test_body_only_tree_matches_optimizer_xsens_contract(anthropometry: G1Anthropometry) -> None:
    model = build_g1_proportioned_xsens_tree(
        anthropometry,
        G1XsensReductionConfig(include_tennis_racket=False),
    )

    assert len(model.bodies) == len(XSENS_BODY_SEGMENT_NAMES) == 23
    assert len(model.joints) == 22
    assert "TennisRacket" not in model.body_map()
    source_names = tuple(str(body.metadata["xsens:sourceSegmentName"]) for body in model.bodies)

    def normalized(name: str) -> str:
        return "".join(name.lower().split())

    assert tuple(map(normalized, source_names)) == tuple(map(normalized, XSENS_BODY_SEGMENT_NAMES))


def test_full_motion_adaptation_preserves_pose_data_and_g1_anchors(
    anthropometry: G1Anthropometry,
    monkeypatch,
) -> None:
    target = build_g1_proportioned_xsens_tree(
        anthropometry,
        G1XsensReductionConfig(include_tennis_racket=False),
    )
    monkeypatch.setattr(
        morphology_adaptation,
        "build_subject_xsens_reference_model",
        lambda *_args, **_kwargs: target,
    )
    reference_positions = np.stack([body.reference_pose.translation_m for body in target.bodies])
    orientations = np.stack([body.reference_pose.rotation_wxyz for body in target.bodies])
    translations = np.array([[1.0, 2.0, 0.3], [1.2, 2.1, 0.5]])
    positions = reference_positions[None, :, :] + translations[:, None, :]
    motion = XsensHdf5Motion(
        positions_m=positions,
        times_s=np.array([4.0, 4.1]),
        stream_name="body_position_xyz_m",
        segment_names=list(XSENS_BODY_SEGMENT_NAMES),
        source_indices=list(range(len(XSENS_BODY_SEGMENT_NAMES))),
        quaternions_wijk=np.repeat(orientations[None, :, :], 2, axis=0),
        orientation_stream_name="body_orientation_quaternion_wijk",
    )

    adapted = adapt_xsens_motion_to_g1(motion, hdf5_path="unused.hdf5")

    np.testing.assert_allclose(adapted.positions_m, positions, atol=5e-6)
    np.testing.assert_array_equal(adapted.orientations_wxyz, motion.quaternions_wijk)
    np.testing.assert_array_equal(adapted.times_s, motion.times_s)
    pelvis_index = XSENS_BODY_SEGMENT_NAMES.index("Pelvis")
    np.testing.assert_array_equal(
        adapted.positions_m[:, pelvis_index, :2],
        motion.positions_m[:, pelvis_index, :2],
    )


def test_default_collapses_axes_into_adapters_and_preserves_rigid_lengths(
    anthropometry: G1Anthropometry,
) -> None:
    model = build_g1_proportioned_xsens_tree(anthropometry)
    bodies = model.body_map()
    joints = model.joint_map()
    reference_joints = compute_reference_joint_positions(model)

    assert len(model.bodies) == 24
    assert len(model.joints) == 23
    assert model.bodies[-1].name == "TennisRacket"
    assert model.bodies[-1].metadata["xsens:sourceSegmentName"] == "RightHandSword"
    assert {mesh.name for mesh in model.bodies[-1].meshes} == {
        "racket_grip",
        "racket_frame",
        "racket_strings",
    }
    assert model.metadata["model:preserveJointOffsets"] is False
    assert validate_kinematic_tree(model).is_valid
    shoulder_offsets: dict[str, np.ndarray] = {}
    for side, title, sign in (("left", "Left", 1.0), ("right", "Right", -1.0)):
        shoulder_offset = (
            bodies[f"{title}UpperArm"].reference_pose.translation_m
            - bodies[f"{title}Shoulder"].reference_pose.translation_m
        )
        wrist_roll_center = bodies[f"{title}ForeArm"].reference_pose.translation_m + np.array(
            [0.0, sign * anthropometry.lengths_m["forearm"], 0.0]
        )
        wrist_adapter = bodies[f"{title}Hand"].reference_pose.translation_m - wrist_roll_center
        shoulder_offsets[side] = shoulder_offset

        expected_shoulder_length = sum(
            np.linalg.norm(edge)
            for edge in anthropometry.compound_offset_edges_m[f"{side}_shoulder"]
        )
        np.testing.assert_allclose(
            shoulder_offset,
            [0.0, sign * expected_shoulder_length, 0.0],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.linalg.norm(wrist_adapter),
            np.linalg.norm(anthropometry.compound_offsets_m[f"{side}_wrist"]),
            atol=1e-12,
        )
        assert sign * wrist_adapter[1] > 0.08
        assert np.linalg.norm(wrist_adapter[[0, 2]]) < 0.01
        np.testing.assert_allclose(
            reference_joints[f"{title}Wrist"],
            bodies[f"{title}Hand"].reference_pose.translation_m,
            atol=1e-12,
        )
        np.testing.assert_allclose(joints[f"{title}Wrist"].child_frame.translation_m, np.zeros(3))

    upper_arm_span = np.linalg.norm(
        bodies["LeftUpperArm"].reference_pose.translation_m - bodies["RightUpperArm"].reference_pose.translation_m
    )
    expected_span = anthropometry.widths_m["shoulder"] + shoulder_offsets["left"][1] - shoulder_offsets["right"][1]
    np.testing.assert_allclose(upper_arm_span, expected_span, atol=1e-12)

    assert np.linalg.norm(joints["LeftShoulder"].child_frame.translation_m) > 0.0
    np.testing.assert_allclose(joints["LeftHip"].child_frame.translation_m, np.zeros(3))
    np.testing.assert_allclose(joints["LeftAnkle"].child_frame.translation_m, np.zeros(3))
    for name, value in _rigid_lengths(model).items():
        np.testing.assert_allclose(value, anthropometry.lengths_m[name], atol=1e-12)


def test_preserved_offsets_appear_once_without_changing_rigid_lengths(
    anthropometry: G1Anthropometry,
) -> None:
    collapsed = build_g1_proportioned_xsens_tree(anthropometry)
    preserved = build_g1_proportioned_xsens_tree(
        anthropometry,
        G1XsensReductionConfig(preserve_joint_offsets=True),
    )
    collapsed_bodies = collapsed.body_map()
    preserved_bodies = preserved.body_map()
    preserved_joints = preserved.joint_map()

    assert preserved.metadata["model:preserveJointOffsets"] is True
    assert validate_kinematic_tree(preserved).is_valid
    assert not np.allclose(
        preserved_bodies["LeftShoulder"].reference_pose.translation_m,
        preserved_bodies["LeftUpperArm"].reference_pose.translation_m,
    )
    assert np.linalg.norm(preserved_joints["LeftHip"].child_frame.translation_m) > 0.0
    np.testing.assert_allclose(preserved_joints["LeftWrist"].child_frame.translation_m, np.zeros(3))
    assert np.linalg.norm(preserved_joints["LeftAnkle"].child_frame.translation_m) > 0.0
    for name, value in _rigid_lengths(preserved).items():
        np.testing.assert_allclose(value, anthropometry.lengths_m[name], atol=1e-12)

    for body_name in ("LeftUpperArm", "LeftUpperLeg", "LeftLowerLeg"):
        collapsed_mesh = collapsed_bodies[body_name].meshes[0]
        preserved_mesh = preserved_bodies[body_name].meshes[0]
        np.testing.assert_allclose(collapsed_mesh.vertices_m, preserved_mesh.vertices_m)
        np.testing.assert_array_equal(collapsed_mesh.faces, preserved_mesh.faces)

    collapsed_forearm = collapsed_bodies["LeftForeArm"].meshes[0]
    preserved_forearm = preserved_bodies["LeftForeArm"].meshes[0]
    np.testing.assert_array_equal(collapsed_forearm.faces, preserved_forearm.faces)
    np.testing.assert_allclose(preserved_forearm.vertices_m, collapsed_forearm.vertices_m)
    for bodies in (collapsed_bodies, preserved_bodies):
        for title, sign in (("Left", 1.0), ("Right", -1.0)):
            forearm = bodies[f"{title}ForeArm"]
            hand = bodies[f"{title}Hand"]
            vertices = np.vstack([mesh.vertices_m for mesh in forearm.meshes])
            visual_edge_y = vertices[:, 1].max() if sign > 0.0 else vertices[:, 1].min()
            hand_origin_y = hand.reference_pose.translation_m[1] - forearm.reference_pose.translation_m[1]
            seam_width = sign * (hand_origin_y - visual_edge_y)
            assert 0.0 <= seam_width < 0.01


@pytest.mark.parametrize("preserve_joint_offsets", [False, True])
def test_joint_collars_remain_on_segment_axes_after_visual_scaling(
    anthropometry: G1Anthropometry,
    preserve_joint_offsets: bool,
) -> None:
    model = build_g1_proportioned_xsens_tree(
        anthropometry,
        G1XsensReductionConfig(preserve_joint_offsets=preserve_joint_offsets),
    )
    bodies = model.body_map()
    joints = compute_reference_joint_positions(model)

    for side in ("Left", "Right"):
        endpoint_bodies = {
            f"{side}Shoulder": f"{side}UpperArm",
            f"{side}UpperArm": f"{side}ForeArm",
            f"{side}ForeArm": f"{side}Hand",
            f"{side}UpperLeg": f"{side}LowerLeg",
        }
        endpoints = {
            body_name: bodies[child_name].reference_pose.translation_m - bodies[body_name].reference_pose.translation_m
            for body_name, child_name in endpoint_bodies.items()
        }
        endpoints[f"{side}Shoulder"] = (
            joints[f"{side}Shoulder"] - bodies[f"{side}Shoulder"].reference_pose.translation_m
        )
        endpoints[f"{side}LowerLeg"] = joints[f"{side}Ankle"] - bodies[f"{side}LowerLeg"].reference_pose.translation_m

        for body_name, endpoint in endpoints.items():
            collars = [mesh for mesh in bodies[body_name].meshes if "joint_collar" in mesh.name]
            assert len(collars) == 1
            collar_center = np.asarray(collars[0].vertices_m).mean(axis=0)
            direction = endpoint / np.linalg.norm(endpoint)
            endpoint_delta = endpoint - collar_center
            axial_gap = float(np.dot(endpoint_delta, direction))
            transverse_error = np.linalg.norm(endpoint_delta - axial_gap * direction)

            assert 0.0 < axial_gap < 0.03
            assert transverse_error < 1e-10

            if body_name.endswith(("Shoulder", "UpperArm", "ForeArm")):
                reference = np.array([1.0, 0.0, 0.0])
                if abs(float(np.dot(direction, reference))) > 0.9:
                    reference = np.array([0.0, 1.0, 0.0])
                basis_u = np.cross(direction, reference)
                basis_u /= np.linalg.norm(basis_u)
                basis_v = np.cross(direction, basis_u)
                vertices = np.vstack([mesh.vertices_m for mesh in bodies[body_name].meshes])
                metric = "forearm" if body_name.endswith("ForeArm") else "upper_arm"
                radii = np.asarray(anthropometry.segment_radii_m[metric])
                np.testing.assert_allclose(
                    [np.ptp(vertices @ basis_u), np.ptp(vertices @ basis_v)],
                    2.0 * radii,
                    atol=1e-12,
                )


def test_preserved_compound_edges_use_region_specific_canonical_frames(
    anthropometry: G1Anthropometry,
) -> None:
    model = build_g1_proportioned_xsens_tree(
        anthropometry,
        G1XsensReductionConfig(preserve_joint_offsets=True, include_visuals=False),
    )
    bodies = model.body_map()
    joints = compute_reference_joint_positions(model)
    pelvis = bodies["Pelvis"].reference_pose.translation_m

    for side, title, sign in (("left", "Left", 1.0), ("right", "Right", -1.0)):
        shoulder_offset = (
            bodies[f"{title}UpperArm"].reference_pose.translation_m
            - bodies[f"{title}Shoulder"].reference_pose.translation_m
        )
        wrist_roll_center = bodies[f"{title}ForeArm"].reference_pose.translation_m + np.array(
            [0.0, sign * anthropometry.lengths_m["forearm"], 0.0]
        )
        wrist_adapter = bodies[f"{title}Hand"].reference_pose.translation_m - wrist_roll_center
        hip_root = pelvis + anthropometry.root_anchors_m[f"{side}_hip"]
        hip_offset = bodies[f"{title}UpperLeg"].reference_pose.translation_m - hip_root
        ankle_offset = bodies[f"{title}Foot"].reference_pose.translation_m - joints[f"{title}Ankle"]

        np.testing.assert_allclose(
            np.linalg.norm(shoulder_offset),
            np.linalg.norm(anthropometry.compound_offsets_m[f"{side}_shoulder"]),
            atol=1e-12,
        )
        assert sign * shoulder_offset[1] > 0.0
        assert shoulder_offset[2] < -0.1
        np.testing.assert_allclose(
            np.linalg.norm(wrist_adapter),
            np.linalg.norm(anthropometry.compound_offsets_m[f"{side}_wrist"]),
            atol=1e-12,
        )
        assert sign * wrist_adapter[1] > 0.08
        assert np.linalg.norm(wrist_adapter[[0, 2]]) < 0.01
        np.testing.assert_allclose(joints[f"{title}Wrist"], bodies[f"{title}Hand"].reference_pose.translation_m)
        np.testing.assert_allclose(hip_offset, anthropometry.compound_offsets_m[f"{side}_hip"], atol=1e-12)
        assert abs(float(ankle_offset[0])) < 1e-10
        assert abs(float(ankle_offset[1])) < 1e-5
        np.testing.assert_allclose(
            np.linalg.norm(ankle_offset),
            np.linalg.norm(anthropometry.compound_offsets_m[f"{side}_ankle"]),
            atol=1e-12,
        )
        assert ankle_offset[2] < 0.0


def test_raw_compound_edges_are_parent_local_and_configuration_independent(
    anthropometry: G1Anthropometry,
) -> None:
    assert len(anthropometry.compound_offset_edges_m["left_shoulder"]) == 2
    assert len(anthropometry.compound_offset_edges_m["left_wrist"]) == 2
    assert len(anthropometry.compound_offset_edges_m["left_hip"]) == 2
    assert len(anthropometry.compound_offset_edges_m["left_ankle"]) == 1
    assert len(anthropometry.compound_offset_edges_m["waist"]) == 2
    np.testing.assert_allclose(anthropometry.compound_offset_edges_m["left_wrist"][0], [0.038, 0.0, 0.0])
    np.testing.assert_allclose(anthropometry.compound_offset_edges_m["left_wrist"][1], [0.046, 0.0, 0.0])
    np.testing.assert_allclose(anthropometry.compound_offset_edges_m["left_ankle"][0], [0.0, 0.0, -0.017558])


def test_shared_pelvis_and_upper_leg_visuals_cover_g1_waist_and_hip_spans(
    anthropometry: G1Anthropometry,
) -> None:
    model = build_g1_proportioned_xsens_tree(anthropometry)
    bodies = model.body_map()
    pelvis = bodies["Pelvis"]
    assert {mesh.name for mesh in pelvis.meshes} == {"pelvis_shell", "pelvis_panel"}
    vertices = np.vstack([mesh.vertices_m for mesh in pelvis.meshes])

    assert anthropometry.region_centers_m["pelvis"][2] < -0.08
    np.testing.assert_allclose(
        np.ptp(vertices, axis=0)[:2],
        anthropometry.region_extents_m["pelvis"][:2],
        atol=1e-12,
    )
    pelvis_position = pelvis.reference_pose.translation_m
    extracted_bottom = anthropometry.region_centers_m["pelvis"][2] - 0.5 * anthropometry.region_extents_m["pelvis"][2]
    np.testing.assert_allclose(vertices[:, 2].min(), extracted_bottom, atol=1e-12)
    for title in ("Left", "Right"):
        hip = bodies[f"{title}UpperLeg"].reference_pose.translation_m - pelvis_position
        upper_leg_vertices = np.vstack([mesh.vertices_m for mesh in bodies[f"{title}UpperLeg"].meshes])
        assert float(hip[2] + upper_leg_vertices[:, 2].max()) >= float(extracted_bottom) - 1e-12
    l5 = bodies["L5"].reference_pose.translation_m - pelvis_position
    assert float(vertices[:, 2].max()) >= float(l5[2]) - 1e-12


def test_all_standard_visuals_come_from_shared_calibrated_avatar_factory(
    anthropometry: G1Anthropometry,
) -> None:
    proportions = g1_anthropometry_to_xsens_avatar_proportions(anthropometry)
    shared = build_xsens_avatar_meshes(proportions)
    model = build_g1_proportioned_xsens_tree(anthropometry)

    for body_name in proportions.segment_names:
        generated_meshes = model.body_map()[body_name].meshes
        shared_parts = shared[body_name]
        standard_meshes = [mesh for mesh in generated_meshes if not mesh.name.endswith("_shoulder_child_adapter")]
        assert [mesh.name for mesh in standard_meshes] == [part.name for part in shared_parts]
        if body_name.endswith("UpperArm"):
            assert len(generated_meshes) == len(shared_parts) + 1
        else:
            assert len(generated_meshes) == len(shared_parts)
        for mesh, part in zip(standard_meshes, shared_parts, strict=True):
            np.testing.assert_array_equal(mesh.faces, part.mesh.faces)
            assert mesh.color_rgb == part.color
            assert mesh.category == part.category


def test_spine_visual_uses_tapered_calibrated_avatar_style(
    anthropometry: G1Anthropometry,
) -> None:
    model = build_g1_proportioned_xsens_tree(anthropometry)
    bodies = model.body_map()

    for name in ("L5", "L3", "T12", "T8"):
        mesh_names = {mesh.name for mesh in bodies[name].meshes}
        assert mesh_names == {f"{name.lower()}_shell", f"{name.lower()}_rear_panel"}
        shell = next(mesh for mesh in bodies[name].meshes if mesh.name.endswith("_shell"))
        distal_z = model.joint_map()[
            {"L5": "L4L3", "L3": "L1T12", "T12": "T9T8", "T8": "T1C7"}[name]
        ].parent_frame.translation_m[2]
        assert float(shell.vertices_m[:, 2].min()) > 0.0
        assert float(shell.vertices_m[:, 2].max()) < float(distal_z)

    l5_width = np.ptp(np.vstack([mesh.vertices_m for mesh in bodies["L5"].meshes]), axis=0)[1]
    t8_width = np.ptp(np.vstack([mesh.vertices_m for mesh in bodies["T8"].meshes]), axis=0)[1]
    assert t8_width > l5_width


def test_hands_reuse_calibrated_avatar_parts_and_match_g1_envelope(
    anthropometry: G1Anthropometry,
) -> None:
    model = build_g1_proportioned_xsens_tree(anthropometry)
    target_extent = np.array(
        [
            2.0 * anthropometry.segment_radii_m["hand"][1],
            anthropometry.lengths_m["hand"],
            2.0 * anthropometry.segment_radii_m["hand"][0],
        ]
    )

    for title, sign in (("Left", 1.0), ("Right", -1.0)):
        meshes = model.body_map()[f"{title}Hand"].meshes
        names = {mesh.name for mesh in meshes}
        assert f"{title.lower()}hand_palm" in names
        assert f"{title.lower()}hand_thumb" in names
        assert sum("_finger_" in name and "joint" not in name for name in names) == 4
        vertices = np.vstack([mesh.vertices_m for mesh in meshes])
        np.testing.assert_allclose(np.ptp(vertices, axis=0), target_extent, atol=1e-12)
        if sign > 0.0:
            np.testing.assert_allclose(vertices[:, 1].min(), 0.0, atol=1e-12)
        else:
            np.testing.assert_allclose(vertices[:, 1].max(), 0.0, atol=1e-12)

        thumb = np.vstack([mesh.vertices_m for mesh in meshes if "thumb" in mesh.name])
        fingers = np.vstack(
            [mesh.vertices_m for mesh in meshes if "_finger_" in mesh.name and "joint" not in mesh.name]
        )
        assert float(thumb[:, 0].max()) > float(fingers[:, 0].max())
        assert float(thumb[:, 0].max()) > 0.0


def test_generation_is_deterministic(anthropometry: G1Anthropometry) -> None:
    first = build_g1_proportioned_xsens_tree(anthropometry)
    second = build_g1_proportioned_xsens_tree(anthropometry)

    for first_body, second_body in zip(first.bodies, second.bodies, strict=True):
        assert first_body.name == second_body.name
        np.testing.assert_array_equal(
            first_body.reference_pose.translation_m,
            second_body.reference_pose.translation_m,
        )
        for first_mesh, second_mesh in zip(first_body.meshes, second_body.meshes, strict=True):
            np.testing.assert_array_equal(first_mesh.vertices_m, second_mesh.vertices_m)
            np.testing.assert_array_equal(first_mesh.faces, second_mesh.faces)


def test_xsens_morphology_adapter_preserves_order_and_target_anchors(
    anthropometry: G1Anthropometry,
) -> None:
    model = build_g1_proportioned_xsens_tree(anthropometry)
    source_names = tuple(str(body.metadata["xsens:sourceSegmentName"]) for body in model.bodies)
    assert source_names[-1] == "RightHandSword"
    adapter = build_xsens_morphology_adapter(model, source_names)
    assert adapter.target_body_to_source_body["TennisRacket"] == "RightHandSword"
    source_positions = np.stack([body.reference_pose.translation_m for body in model.bodies])
    source_positions[1:] += np.array([10.0, -4.0, 2.0])
    angles = np.linspace(-0.4, 0.4, len(model.bodies))
    orientations = np.column_stack(
        [np.cos(0.5 * angles), np.sin(0.5 * angles), np.zeros_like(angles), np.zeros_like(angles)]
    )

    adapted = adapter.adapt_pose(KinematicPose(source_names, source_positions, orientations))

    assert adapted.body_names == source_names
    np.testing.assert_array_equal(adapted.orientations_wxyz, orientations)
    np.testing.assert_array_equal(adapted.positions_m[0], source_positions[0])
    source_indices = {name: index for index, name in enumerate(source_names)}
    body_to_source = adapter.target_body_to_source_body
    for joint in model.joints:
        parent_index = source_indices[body_to_source[joint.parent_body]]
        child_index = source_indices[body_to_source[joint.child_body]]
        parent_anchor = adapted.positions_m[parent_index] + rotate_vector(
            adapted.orientations_wxyz[parent_index],
            joint.parent_frame.translation_m,
        )
        child_anchor = adapted.positions_m[child_index] + rotate_vector(
            adapted.orientations_wxyz[child_index],
            joint.child_frame.translation_m,
        )
        np.testing.assert_allclose(parent_anchor, child_anchor, atol=5e-6)


def test_collapsed_shoulder_joint_frames_satisfy_t_and_n_pose_objectives(
    anthropometry: G1Anthropometry,
) -> None:
    model = build_g1_proportioned_xsens_tree(
        anthropometry,
        G1XsensReductionConfig(include_tennis_racket=False),
    )
    joints = model.joint_map()

    for side, title, sign in (("left", "Left", 1.0), ("right", "Right", -1.0)):
        fit = _fit_collapsed_shoulder_morphology(anthropometry, side, sign)
        joint = joints[f"{title}Shoulder"]
        np.testing.assert_allclose(joint.parent_frame.translation_m, fit.parent_anchor_m, atol=1e-12)
        np.testing.assert_allclose(joint.child_frame.translation_m, fit.child_anchor_m, atol=1e-12)
        np.testing.assert_allclose(
            fit.parent_anchor_m - fit.child_anchor_m,
            fit.tpose_target_offset_m,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            fit.parent_anchor_m - fit.npose_child_rotation @ fit.child_anchor_m,
            fit.npose_target_offset_m,
            atol=1e-12,
        )
        assert fit.tpose_error_m < 1e-12
        assert fit.npose_error_m < 1e-12


def test_collapsed_shoulder_adapter_interpolates_t_to_n_pose_over_motion(
    anthropometry: G1Anthropometry,
) -> None:
    model = build_g1_proportioned_xsens_tree(
        anthropometry,
        G1XsensReductionConfig(include_tennis_racket=False),
    )
    source_names = tuple(str(body.metadata["xsens:sourceSegmentName"]) for body in model.bodies)
    source_indices = {name: index for index, name in enumerate(source_names)}
    adapter = build_xsens_morphology_adapter(model, source_names)
    frame_count = 9
    reference_positions = np.stack([body.reference_pose.translation_m for body in model.bodies])
    positions = np.repeat(reference_positions[None, :, :], frame_count, axis=0)
    orientations = np.zeros((frame_count, len(source_names), 4), dtype=float)
    orientations[..., 0] = 1.0
    elevation = np.linspace(0.0, 0.5 * np.pi, frame_count)
    for title, sign in (("Left", 1.0), ("Right", -1.0)):
        upper_arm_index = source_indices[f"{title}UpperArm"]
        angles = -sign * elevation
        orientations[:, upper_arm_index, 0] = np.cos(0.5 * angles)
        orientations[:, upper_arm_index, 1] = np.sin(0.5 * angles)
    times_s = np.arange(frame_count, dtype=float) / 30.0

    adapted = adapter.adapt_motion(
        KinematicMotion(source_names, positions, orientations, times_s)
    )

    for side, title, sign in (("left", "Left", 1.0), ("right", "Right", -1.0)):
        shoulder_index = source_indices[f"{title}Shoulder"]
        upper_arm_index = source_indices[f"{title}UpperArm"]
        joint = model.joint_map()[f"{title}Shoulder"]
        rotated_child_anchors = np.stack(
            [
                rotate_vector(orientation, joint.child_frame.translation_m)
                for orientation in orientations[:, upper_arm_index]
            ]
        )
        expected_offsets = joint.parent_frame.translation_m - rotated_child_anchors
        actual_offsets = adapted.positions_m[:, upper_arm_index] - adapted.positions_m[:, shoulder_index]
        fit = _fit_collapsed_shoulder_morphology(anthropometry, side, sign)
        np.testing.assert_allclose(actual_offsets, expected_offsets, atol=1e-12)
        np.testing.assert_allclose(actual_offsets[0], fit.tpose_target_offset_m, atol=1e-12)
        np.testing.assert_allclose(actual_offsets[-1], fit.npose_target_offset_m, atol=1e-12)
    np.testing.assert_array_equal(adapted.orientations_wxyz, orientations)
    np.testing.assert_array_equal(adapted.times_s, times_s)


@pytest.mark.parametrize("preserve_joint_offsets", [False, True])
def test_virtual_wrist_stays_at_hand_origin_under_large_hand_rotations(
    anthropometry: G1Anthropometry,
    preserve_joint_offsets: bool,
) -> None:
    model = build_g1_proportioned_xsens_tree(
        anthropometry,
        G1XsensReductionConfig(
            preserve_joint_offsets=preserve_joint_offsets,
            include_tennis_racket=False,
        ),
    )
    source_names = tuple(str(body.metadata["xsens:sourceSegmentName"]) for body in model.bodies)
    source_indices = {name: index for index, name in enumerate(source_names)}
    adapter = build_xsens_morphology_adapter(model, source_names)
    frame_count = 8
    reference_positions = np.stack([body.reference_pose.translation_m for body in model.bodies])
    positions = np.repeat(reference_positions[None, :, :], frame_count, axis=0)
    orientations = np.zeros((frame_count, len(source_names), 4), dtype=float)
    orientations[..., 0] = 1.0
    wrist_angles = np.deg2rad(np.linspace(0.0, 140.0, frame_count))
    for title, sign in (("Left", 1.0), ("Right", -1.0)):
        hand_index = source_indices[f"{title}Hand"]
        orientations[:, hand_index, 0] = np.cos(0.5 * wrist_angles)
        orientations[:, hand_index, 1] = sign * np.sin(0.5 * wrist_angles)

    adapted = adapter.adapt_motion(
        KinematicMotion(
            source_names,
            positions,
            orientations,
            np.arange(frame_count, dtype=float) / 30.0,
        )
    )

    for title in ("Left", "Right"):
        forearm_index = source_indices[f"{title}ForeArm"]
        hand_index = source_indices[f"{title}Hand"]
        wrist = model.joint_map()[f"{title}Wrist"]
        actual_offsets = adapted.positions_m[:, hand_index] - adapted.positions_m[:, forearm_index]
        expected_offsets = np.repeat(wrist.parent_frame.translation_m[None, :], frame_count, axis=0)

        np.testing.assert_allclose(wrist.child_frame.translation_m, np.zeros(3), atol=1e-12)
        np.testing.assert_allclose(actual_offsets, expected_offsets, atol=1e-12)
        np.testing.assert_allclose(
            np.linalg.norm(actual_offsets, axis=1),
            np.linalg.norm(wrist.parent_frame.translation_m),
            atol=1e-12,
        )
    np.testing.assert_array_equal(adapted.orientations_wxyz, orientations)


def test_xsens_lowest_sole_grounding_uses_shared_avatar_outsoles(
    anthropometry: G1Anthropometry,
) -> None:
    model = build_g1_proportioned_xsens_tree(anthropometry)
    source_names = tuple(str(body.metadata["xsens:sourceSegmentName"]) for body in model.bodies)
    adapter = build_xsens_morphology_adapter(
        model,
        source_names,
        source_model=model,
        grounding="match_lowest_soles",
    )
    positions = np.stack([body.reference_pose.translation_m for body in model.bodies])
    orientations = np.stack([body.reference_pose.rotation_wxyz for body in model.bodies])

    adapted = adapter.adapt_pose(KinematicPose(source_names, positions, orientations))

    np.testing.assert_allclose(adapted.positions_m, positions, atol=5e-6)
    np.testing.assert_array_equal(adapted.orientations_wxyz, orientations)


def test_usd_export_round_trips_and_reports_both_raw_and_applied_offsets(
    tmp_path,
    anthropometry: G1Anthropometry,
) -> None:
    pytest.importorskip("pxr")
    output_path = tmp_path / "g1_proportioned_xsens.usda"
    report = export_g1_proportioned_xsens_usd(
        output_path,
        config=G1XsensReductionConfig(preserve_joint_offsets=False),
    )
    payload = json.loads(report.report_path.read_text(encoding="utf-8"))

    assert output_path.is_file()
    assert report.report_path.is_file()
    assert report.body_count == 24
    assert report.joint_count == 23
    assert report.max_length_error_m < 1e-12
    assert report.max_joint_residual_m < 5e-6
    assert payload["preserve_joint_offsets"] is False
    assert any(np.linalg.norm(value) > 0.0 for value in payload["raw_offsets_m"].values())
    assert payload["raw_offset_edge_frame"] == "parent_body_local"
    assert len(payload["raw_offset_edges_m"]["left_wrist"]) == 2
    assert np.linalg.norm(payload["root_anchors_m"]["left_hip"]) > 0.0
    expected_shoulder_path_length = sum(
        np.linalg.norm(edge) for edge in anthropometry.compound_offset_edges_m["left_shoulder"]
    )
    np.testing.assert_allclose(
        payload["collapsed_adapter_offsets_m"]["left_shoulder"],
        [0.0, expected_shoulder_path_length, 0.0],
        atol=1e-12,
    )
    assert np.linalg.norm(payload["collapsed_adapter_offsets_m"]["left_wrist"]) > 0.0
    assert np.linalg.norm(payload["collapsed_adapter_offsets_m"]["left_hip"]) > 0.0
    assert all(np.linalg.norm(value) == 0.0 for value in payload["applied_offsets_m"].values())
