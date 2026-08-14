"""Tests for root-motion policies in Xsens morphology adaptation."""

from __future__ import annotations

import numpy as np
import pytest
from holosoma_retargeting.kinematics import (
    KinematicMotion,
    KinematicPose,
    KinematicTree,
    MeshAttachment,
    RigidBodyDefinition,
    SphericalJointDefinition,
    Transform,
)
from holosoma_retargeting.xsens.morphology_adaptation import (
    XsensRootMotionConfig,
    _body_to_source_mapping,
    _clean_contact_mask,
    _contact_aware_xy_correction,
    _detect_contacts,
    _functional_leg_length_m,
    _resolved_ground_height_m,
    _surface_evaluator,
    apply_xsens_root_motion,
    build_xsens_morphology_adapter,
)

BODY_NAMES = ("Pelvis", "LeftFoot", "LeftToe", "RightFoot", "RightToe")


def _sole_mesh(name: str) -> MeshAttachment:
    return MeshAttachment(
        name=f"{name}_outsole",
        vertices_m=np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.1, 0.0]]),
        faces=np.array([[0, 1, 2]]),
    )


def _model(leg_length_m: float) -> KinematicTree:
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    bodies = (
        RigidBodyDefinition("Pelvis", Transform(np.array([0.0, 0.0, leg_length_m]), identity)),
        RigidBodyDefinition(
            "LeftFoot",
            Transform(np.array([0.0, 0.1, 0.0]), identity),
            meshes=(_sole_mesh("left_foot"),),
        ),
        RigidBodyDefinition(
            "LeftToe",
            Transform(np.array([0.2, 0.1, 0.0]), identity),
            meshes=(_sole_mesh("left_toe"),),
        ),
        RigidBodyDefinition(
            "RightFoot",
            Transform(np.array([0.0, -0.1, 0.0]), identity),
            meshes=(_sole_mesh("right_foot"),),
        ),
        RigidBodyDefinition(
            "RightToe",
            Transform(np.array([0.2, -0.1, 0.0]), identity),
            meshes=(_sole_mesh("right_toe"),),
        ),
    )
    bodies = tuple(
        RigidBodyDefinition(
            body.name,
            body.reference_pose,
            meshes=body.meshes,
            metadata={"xsens:sourceSegmentName": body.name},
        )
        for body in bodies
    )
    joints = (
        SphericalJointDefinition(
            "LeftHip",
            "Pelvis",
            "LeftFoot",
            Transform.identity(),
            Transform(np.array([0.0, -0.1, leg_length_m]), identity),
        ),
        SphericalJointDefinition(
            "LeftBallFoot",
            "LeftFoot",
            "LeftToe",
            Transform(np.array([0.2, 0.0, 0.0]), identity),
            Transform.identity(),
        ),
        SphericalJointDefinition(
            "RightHip",
            "Pelvis",
            "RightFoot",
            Transform.identity(),
            Transform(np.array([0.0, 0.1, leg_length_m]), identity),
        ),
        SphericalJointDefinition(
            "RightBallFoot",
            "RightFoot",
            "RightToe",
            Transform(np.array([0.2, 0.0, 0.0]), identity),
            Transform.identity(),
        ),
    )
    return KinematicTree(f"avatar_{leg_length_m}", "Pelvis", bodies, joints)


def _source_motion(
    model: KinematicTree,
    translations_m: np.ndarray,
    *,
    times_s: np.ndarray | None = None,
) -> KinematicMotion:
    reference = np.asarray([body.reference_pose.translation_m for body in model.bodies])
    orientations = np.asarray([body.reference_pose.rotation_wxyz for body in model.bodies])
    frame_count = len(translations_m)
    return KinematicMotion(
        BODY_NAMES,
        reference[None, :, :] + np.asarray(translations_m)[:, None, :],
        np.repeat(orientations[None, :, :], frame_count, axis=0),
        np.arange(frame_count, dtype=float) / 20.0 if times_s is None else times_s,
    )


def _raw_target(source: KinematicMotion, target_model: KinematicTree) -> KinematicMotion:
    adapter = build_xsens_morphology_adapter(target_model, BODY_NAMES, grounding="none")
    return adapter.adapt_motion(source)


def _scalar_surface_references(motion: KinematicMotion, evaluator) -> np.ndarray:
    return np.asarray(
        [
            evaluator.support_reference_m(
                KinematicPose(
                    motion.body_names,
                    motion.positions_m[frame],
                    motion.orientations_wxyz[frame],
                )
            )
            for frame in range(len(motion.times_s))
        ]
    )


def _scalar_root_motion_reference(
    source: KinematicMotion,
    raw_target: KinematicMotion,
    source_model: KinematicTree,
    target_model: KinematicTree,
    *,
    grounding: str,
    config: XsensRootMotionConfig,
) -> tuple[np.ndarray, float, float, dict[str, np.ndarray]]:
    """Retain the original per-frame algorithm as an independent test oracle."""

    sides = ("Left", "Right")
    source_mapping = _body_to_source_mapping(source_model, source.body_names, model_label="Source")
    target_mapping = _body_to_source_mapping(target_model, source.body_names, model_label="Target")
    source_all = _surface_evaluator(source_model, source.body_names, source_mapping)
    target_all = _surface_evaluator(target_model, source.body_names, target_mapping)
    source_sides = {
        side: _surface_evaluator(source_model, source.body_names, source_mapping, side=side)
        for side in sides
    }
    target_sides = {
        side: _surface_evaluator(target_model, source.body_names, target_mapping, side=side)
        for side in sides
    }
    source_references = {
        side: _scalar_surface_references(source, source_sides[side]) for side in sides
    }
    ground_height_m = _resolved_ground_height_m(
        source_references["Left"],
        source_references["Right"],
        config.ground_height_m,
    )
    source_leg_length_m = _functional_leg_length_m(source_model)
    target_leg_length_m = _functional_leg_length_m(target_model)
    scale = 1.0 if config.mode == "preserve_world" else target_leg_length_m / source_leg_length_m
    source_root_name = source_mapping[source_model.root_body]
    root_index = source.body_names.index(source_root_name)
    source_root = np.asarray(source.positions_m[:, root_index], dtype=float)
    positions = np.asarray(raw_target.positions_m, dtype=float).copy()
    initial_xy = source_root[0, :2].copy()

    for frame in range(positions.shape[0]):
        source_pose = KinematicPose(
            source.body_names,
            source.positions_m[frame],
            source.orientations_wxyz[frame],
        )
        target_pose = KinematicPose(
            raw_target.body_names,
            positions[frame],
            raw_target.orientations_wxyz[frame],
        )
        translation = np.zeros(3, dtype=float)
        desired_xy = initial_xy + scale * (source_root[frame, :2] - initial_xy)
        translation[:2] = desired_xy - positions[frame, root_index, :2]
        if grounding == "match_lowest_soles":
            source_height = source_all.minimum_height_m(source_pose)
            desired_height = ground_height_m + scale * (source_height - ground_height_m)
            translation[2] = desired_height - target_all.minimum_height_m(target_pose)
        else:
            desired_root_z = ground_height_m + scale * (source_root[frame, 2] - ground_height_m)
            translation[2] = desired_root_z - positions[frame, root_index, 2]
        positions[frame] += translation

    contacts = {side: np.zeros(positions.shape[0], dtype=bool) for side in sides}
    if config.mode == "scale_by_leg_length_contact_aware":
        contacts = {
            side: _detect_contacts(
                source_references[side],
                source.times_s,
                ground_height_m=ground_height_m,
                config=config,
            )
            for side in sides
        }
        baseline = KinematicMotion(
            raw_target.body_names,
            positions,
            raw_target.orientations_wxyz,
            raw_target.times_s,
        )
        target_references = {
            side: _scalar_surface_references(baseline, target_sides[side]) for side in sides
        }
        positions[:, :, :2] += _contact_aware_xy_correction(target_references, contacts)[:, None, :]

    return positions, ground_height_m, scale, contacts


@pytest.mark.parametrize("grounding", ["none", "match_lowest_soles"])
def test_preserve_world_reproduces_legacy_root_mapping(grounding: str) -> None:
    source_model = _model(2.0)
    target_model = _model(1.0)
    source = _source_motion(source_model, np.array([[10.0, 5.0, 0.1], [12.0, 7.0, 0.5]]))

    mapped, report = apply_xsens_root_motion(
        source,
        _raw_target(source, target_model),
        source_model=source_model,
        target_model=target_model,
        grounding=grounding,
        config=XsensRootMotionConfig(mode="preserve_world", ground_height_m=0.0),
    )

    pelvis = BODY_NAMES.index("Pelvis")
    np.testing.assert_allclose(mapped.positions_m[:, pelvis, :2], source.positions_m[:, pelvis, :2])
    if grounding == "none":
        np.testing.assert_allclose(mapped.positions_m[:, pelvis, 2], source.positions_m[:, pelvis, 2])
    else:
        left_foot = BODY_NAMES.index("LeftFoot")
        np.testing.assert_allclose(mapped.positions_m[:, left_foot, 2], source.positions_m[:, left_foot, 2])
    assert report.scale == 1.0


def test_leg_scaled_without_grounding_scales_root_xyz_about_initial_xy_and_ground() -> None:
    source_model = _model(2.0)
    target_model = _model(1.0)
    source = _source_motion(source_model, np.array([[10.0, 5.0, 0.0], [12.0, 7.0, 1.0]]))

    mapped, report = apply_xsens_root_motion(
        source,
        _raw_target(source, target_model),
        source_model=source_model,
        target_model=target_model,
        grounding="none",
        config=XsensRootMotionConfig(mode="scale_by_leg_length", ground_height_m=0.0),
    )

    pelvis = BODY_NAMES.index("Pelvis")
    np.testing.assert_allclose(mapped.positions_m[:, pelvis], [[10.0, 5.0, 1.0], [11.0, 6.0, 1.5]])
    assert report.source_leg_length_m == pytest.approx(2.0)
    assert report.target_leg_length_m == pytest.approx(1.0)
    assert report.scale == pytest.approx(0.5)


def test_leg_scaled_grounding_scales_outsole_clearance_not_root_height() -> None:
    source_model = _model(2.0)
    target_model = _model(1.0)
    source = _source_motion(source_model, np.array([[0.0, 0.0, 0.2], [2.0, 0.0, 0.8]]))

    mapped, _ = apply_xsens_root_motion(
        source,
        _raw_target(source, target_model),
        source_model=source_model,
        target_model=target_model,
        grounding="match_lowest_soles",
        config=XsensRootMotionConfig(mode="scale_by_leg_length", ground_height_m=0.0),
    )

    pelvis = BODY_NAMES.index("Pelvis")
    foot = BODY_NAMES.index("LeftFoot")
    np.testing.assert_allclose(mapped.positions_m[:, pelvis, :2], [[0.0, 0.0], [1.0, 0.0]])
    np.testing.assert_allclose(mapped.positions_m[:, foot, 2], [0.1, 0.4])
    np.testing.assert_allclose(mapped.positions_m[:, pelvis, 2], [1.1, 1.4])
    np.testing.assert_array_equal(mapped.orientations_wxyz, source.orientations_wxyz)
    np.testing.assert_array_equal(mapped.times_s, source.times_s)


def test_contact_aware_mode_locks_stationary_source_stance_and_holds_correction_in_flight() -> None:
    source_model = _model(2.0)
    target_model = _model(1.0)
    source = _source_motion(
        source_model,
        np.column_stack([np.arange(6, dtype=float), np.zeros(6), np.zeros(6)]),
    )
    left_foot = BODY_NAMES.index("LeftFoot")
    left_toe = BODY_NAMES.index("LeftToe")
    right_foot = BODY_NAMES.index("RightFoot")
    right_toe = BODY_NAMES.index("RightToe")
    source_positions = source.positions_m.copy()
    source_positions[:, [left_foot, left_toe], 0] = source_positions[0, [left_foot, left_toe], 0]
    source_positions[:4, [left_foot, left_toe], 2] = 0.0
    source_positions[4:, [left_foot, left_toe], 2] = 0.5
    source_positions[:, [right_foot, right_toe], 2] = 0.5
    source = KinematicMotion(source.body_names, source_positions, source.orientations_wxyz, source.times_s)

    mapped, report = apply_xsens_root_motion(
        source,
        _raw_target(source, target_model),
        source_model=source_model,
        target_model=target_model,
        grounding="none",
        config=XsensRootMotionConfig(
            mode="scale_by_leg_length_contact_aware",
            ground_height_m=0.0,
            contact_min_duration_s=0.1,
        ),
    )

    np.testing.assert_allclose(mapped.positions_m[:4, left_foot, 0], mapped.positions_m[0, left_foot, 0])
    np.testing.assert_allclose(np.diff(mapped.positions_m[3:, left_foot, 0]), [0.5, 0.5])
    assert report.left_contact_frames == 4
    assert report.right_contact_frames == 0


@pytest.mark.parametrize(
    ("mode", "grounding"),
    [
        ("preserve_world", "none"),
        ("preserve_world", "match_lowest_soles"),
        ("scale_by_leg_length", "none"),
        ("scale_by_leg_length", "match_lowest_soles"),
        ("scale_by_leg_length_contact_aware", "none"),
        ("scale_by_leg_length_contact_aware", "match_lowest_soles"),
    ],
)
def test_vectorized_root_motion_matches_original_scalar_algorithm(
    mode: str,
    grounding: str,
) -> None:
    rng = np.random.default_rng(1917)
    source_model = _model(1.9)
    target_model = _model(0.83)
    frame_count = 41
    translations = np.column_stack(
        (
            np.linspace(-0.4, 1.7, frame_count),
            0.2 * np.sin(np.linspace(0.0, 3.0, frame_count)),
            0.15 + 0.1 * np.cos(np.linspace(0.0, 5.0, frame_count)),
        )
    )
    times = np.cumsum(rng.uniform(0.02, 0.07, size=frame_count))
    source = _source_motion(source_model, translations, times_s=times)
    orientations = rng.normal(size=source.orientations_wxyz.shape)
    orientations *= rng.uniform(0.2, 4.0, size=orientations.shape[:-1] + (1,))
    source = KinematicMotion(source.body_names, source.positions_m, orientations, source.times_s)
    raw_target = _raw_target(source, target_model)
    config = XsensRootMotionConfig(
        mode=mode,
        contact_height_tolerance_m=100.0,
        contact_speed_threshold_m_s=100.0,
        contact_min_duration_s=0.01,
    )

    expected_positions, expected_ground, expected_scale, expected_contacts = (
        _scalar_root_motion_reference(
            source,
            raw_target,
            source_model,
            target_model,
            grounding=grounding,
            config=config,
        )
    )
    actual, report = apply_xsens_root_motion(
        source,
        raw_target,
        source_model=source_model,
        target_model=target_model,
        grounding=grounding,
        config=config,
    )

    np.testing.assert_allclose(actual.positions_m, expected_positions, rtol=2e-14, atol=2e-14)
    np.testing.assert_array_equal(actual.orientations_wxyz, raw_target.orientations_wxyz)
    np.testing.assert_array_equal(actual.times_s, raw_target.times_s)
    assert report.ground_height_m == pytest.approx(expected_ground, abs=1e-14)
    assert report.scale == pytest.approx(expected_scale, abs=1e-14)
    assert report.left_contact_frames == int(np.count_nonzero(expected_contacts["Left"]))
    assert report.right_contact_frames == int(np.count_nonzero(expected_contacts["Right"]))


def test_root_motion_validation_rejects_invalid_contact_configuration() -> None:
    source_model = _model(2.0)
    target_model = _model(1.0)
    source = _source_motion(source_model, np.zeros((2, 3)))

    with pytest.raises(ValueError, match="contact_height_tolerance_m"):
        apply_xsens_root_motion(
            source,
            _raw_target(source, target_model),
            source_model=source_model,
            target_model=target_model,
            grounding="none",
            config=XsensRootMotionConfig(contact_height_tolerance_m=-1.0),
        )


@pytest.mark.parametrize(
    ("references", "expected"),
    [
        (np.array([[0.0, 0.0, 0.0]] * 4), [True, True, True, True]),
        (
            np.array([[0.00, 0.0, 0.0], [0.03, 0.0, 0.0], [0.08, 0.0, 0.0], [0.15, 0.0, 0.0]]),
            [False, False, False, False],
        ),
        (np.array([[0.0, 0.0, 0.1]] * 4), [False, False, False, False]),
    ],
)
def test_contact_detection_combines_height_and_speed_on_irregular_timestamps(
    references: np.ndarray,
    expected: list[bool],
) -> None:
    contacts = _detect_contacts(
        references,
        np.array([0.0, 0.03, 0.08, 0.15]),
        ground_height_m=0.0,
        config=XsensRootMotionConfig(contact_min_duration_s=0.1),
    )

    np.testing.assert_array_equal(contacts, expected)


def test_contact_cleanup_fills_short_gaps_and_removes_short_runs() -> None:
    times = np.arange(7, dtype=float) * 0.03

    filled = _clean_contact_mask(
        np.array([True, True, False, True, True, False, False]),
        times,
        max_gap_s=0.067,
        min_duration_s=0.1,
    )
    removed = _clean_contact_mask(
        np.array([False, True, False, False, False, False, False]),
        times,
        max_gap_s=0.0,
        min_duration_s=0.1,
    )

    np.testing.assert_array_equal(filled[:5], True)
    np.testing.assert_array_equal(removed, False)


def test_double_support_uses_mean_least_squares_translation() -> None:
    references = {
        "Left": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        "Right": np.array([[2.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
    }
    contacts = {"Left": np.ones(2, dtype=bool), "Right": np.ones(2, dtype=bool)}

    corrections = _contact_aware_xy_correction(references, contacts)

    np.testing.assert_allclose(corrections, [[0.0, 0.0], [-1.5, 0.0]])
