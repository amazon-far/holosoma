"""Tests for kinematic morphology utilities."""

from __future__ import annotations

import numpy as np
import pytest
from holosoma_retargeting.kinematics import (
    GroundingSurface,
    KinematicMorphologyAdapter,
    KinematicMotion,
    KinematicPose,
    KinematicTree,
    LowestSurfaceGrounding,
    MeshAttachment,
    RigidBodyDefinition,
    SphericalJointDefinition,
    SurfacePoseEvaluator,
    Transform,
    compute_joint_positions,
    reference_grounding_offset_m,
    reference_root_floor_clearance_m,
)

SOURCE_NAMES = ("RootStream", "ChildStream")
MAPPING = {"Root": "RootStream", "Child": "ChildStream"}


def _model(*, child_origin_z: float = 1.0, sole_z: float | None = None, reversed_order: bool = False):
    sole_meshes = ()
    if sole_z is not None:
        sole_meshes = (
            MeshAttachment(
                "sole",
                np.array([[0.0, 0.0, sole_z], [0.1, 0.0, sole_z], [0.0, 0.1, sole_z]]),
                np.array([[0, 1, 2]]),
            ),
        )
    root = RigidBodyDefinition("Root", Transform.identity())
    child = RigidBodyDefinition("Child", Transform(np.array([0.0, 0.0, child_origin_z])), meshes=sole_meshes)
    bodies = (child, root) if reversed_order else (root, child)
    return KinematicTree(
        "two_body",
        "Root",
        bodies,
        (
            SphericalJointDefinition(
                "Joint",
                "Root",
                "Child",
                Transform(np.array([0.0, 0.0, child_origin_z * 0.5])),
                Transform(np.array([0.0, 0.0, -child_origin_z * 0.5])),
            ),
        ),
    )


def _pose(*, root_z: float = 0.0, child_z: float = 99.0) -> KinematicPose:
    return KinematicPose(
        SOURCE_NAMES,
        np.array([[0.0, 0.0, root_z], [0.0, 0.0, child_z]]),
        np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
    )


def test_adapter_preserves_source_order_root_and_orientations() -> None:
    model = _model(child_origin_z=2.0, reversed_order=True)
    adapter = KinematicMorphologyAdapter(model, SOURCE_NAMES, target_body_to_source_body=MAPPING)
    source = _pose(root_z=3.0)

    adapted = adapter.adapt_pose(source)

    assert adapted.body_names == SOURCE_NAMES
    np.testing.assert_array_equal(adapted.orientations_wxyz, source.orientations_wxyz)
    np.testing.assert_array_equal(adapted.positions_m[0], source.positions_m[0])
    np.testing.assert_allclose(adapted.positions_m[1], [0.0, 0.0, 5.0])
    poses = {
        "Root": Transform(adapted.positions_m[0], adapted.orientations_wxyz[0]),
        "Child": Transform(adapted.positions_m[1], adapted.orientations_wxyz[1]),
    }
    np.testing.assert_allclose(compute_joint_positions(model, poses)["Joint"], [0.0, 0.0, 4.0])
    assert not np.shares_memory(adapted.positions_m, source.positions_m)
    assert not np.shares_memory(adapted.orientations_wxyz, source.orientations_wxyz)


def test_authored_reference_pose_is_an_exact_identity_adaptation() -> None:
    model = _model(child_origin_z=1.25)
    source = KinematicPose(
        SOURCE_NAMES,
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.25]]),
        np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (2, 1)),
    )

    adapted = KinematicMorphologyAdapter(
        model,
        SOURCE_NAMES,
        target_body_to_source_body=MAPPING,
    ).adapt_pose(source)

    np.testing.assert_array_equal(adapted.positions_m, source.positions_m)
    np.testing.assert_array_equal(adapted.orientations_wxyz, source.orientations_wxyz)


def test_non_root_source_positions_do_not_affect_ungrounded_adaptation() -> None:
    adapter = KinematicMorphologyAdapter(_model(), SOURCE_NAMES, target_body_to_source_body=MAPPING)
    first = adapter.adapt_pose(_pose(child_z=10.0))
    second = adapter.adapt_pose(_pose(child_z=-500.0))
    np.testing.assert_array_equal(first.positions_m, second.positions_m)


def test_batch_adaptation_matches_pose_kernel_and_preserves_motion_data() -> None:
    adapter = KinematicMorphologyAdapter(_model(child_origin_z=1.5), SOURCE_NAMES, target_body_to_source_body=MAPPING)
    positions = np.array([_pose(root_z=1.0).positions_m, _pose(root_z=2.0).positions_m])
    orientations = np.repeat(_pose().orientations_wxyz[None, :, :], 2, axis=0)
    times = np.array([4.0, 4.1])
    source = KinematicMotion(SOURCE_NAMES, positions, orientations, times)

    adapted = adapter.adapt_motion(source)

    for frame in range(2):
        expected = adapter.adapt_pose(KinematicPose(SOURCE_NAMES, positions[frame], orientations[frame]))
        np.testing.assert_array_equal(adapted.positions_m[frame], expected.positions_m)
    np.testing.assert_array_equal(adapted.orientations_wxyz, orientations)
    np.testing.assert_array_equal(adapted.times_s, times)
    assert not np.shares_memory(adapted.times_s, times)


@pytest.mark.parametrize("with_grounding", [False, True])
def test_vectorized_batch_matches_scalar_pose_kernel_for_random_motion(
    with_grounding: bool,
) -> None:
    rng = np.random.default_rng(8421)
    frame_count = 127
    positions = rng.normal(size=(frame_count, len(SOURCE_NAMES), 3))
    orientations = rng.normal(size=(frame_count, len(SOURCE_NAMES), 4))
    orientations *= rng.uniform(0.2, 3.0, size=orientations.shape[:-1] + (1,))
    times = np.cumsum(rng.uniform(0.005, 0.08, size=frame_count))
    source = KinematicMotion(SOURCE_NAMES, positions, orientations, times)
    target_model = _model(child_origin_z=1.7, sole_z=-0.13)

    grounding = None
    if with_grounding:
        source_model = _model(child_origin_z=0.8, sole_z=-0.21)
        surfaces = (GroundingSurface("Child", ("sole",)),)
        grounding = LowestSurfaceGrounding(
            source_model,
            target_model,
            SOURCE_NAMES,
            source_body_to_pose_body=MAPPING,
            target_body_to_pose_body=MAPPING,
            source_surfaces=surfaces,
            target_surfaces=surfaces,
        )
    adapter = KinematicMorphologyAdapter(
        target_model,
        SOURCE_NAMES,
        target_body_to_source_body=MAPPING,
        grounding=grounding,
    )

    batch = adapter.adapt_motion(source)
    scalar_positions = np.stack(
        [
            adapter.adapt_pose(
                KinematicPose(SOURCE_NAMES, positions[frame], orientations[frame])
            ).positions_m
            for frame in range(frame_count)
        ]
    )

    np.testing.assert_allclose(batch.positions_m, scalar_positions, rtol=2e-14, atol=2e-14)
    np.testing.assert_array_equal(batch.orientations_wxyz, orientations)
    np.testing.assert_array_equal(batch.times_s, times)


def test_vectorized_batch_preserves_multilevel_kinematic_dependency_order() -> None:
    body_names = ("TipStream", "RootStream", "MiddleStream")
    mapping = {"Root": "RootStream", "Middle": "MiddleStream", "Tip": "TipStream"}
    model = KinematicTree(
        "three_body_chain",
        "Root",
        (
            RigidBodyDefinition("Tip", Transform(np.array([0.0, 0.0, 2.0]))),
            RigidBodyDefinition("Root", Transform.identity()),
            RigidBodyDefinition("Middle", Transform(np.array([0.0, 0.0, 1.0]))),
        ),
        (
            SphericalJointDefinition(
                "RootToMiddle",
                "Root",
                "Middle",
                Transform(np.array([0.1, 0.0, 0.4])),
                Transform(np.array([0.1, 0.0, -0.6])),
            ),
            SphericalJointDefinition(
                "MiddleToTip",
                "Middle",
                "Tip",
                Transform(np.array([0.0, 0.2, 0.6])),
                Transform(np.array([0.0, 0.2, -0.4])),
            ),
        ),
    )
    rng = np.random.default_rng(5540)
    frame_count = 59
    positions = rng.normal(size=(frame_count, len(body_names), 3))
    orientations = rng.normal(size=(frame_count, len(body_names), 4))
    motion = KinematicMotion(
        body_names,
        positions,
        orientations,
        np.arange(frame_count, dtype=float) * 0.02,
    )
    adapter = KinematicMorphologyAdapter(
        model,
        body_names,
        target_body_to_source_body=mapping,
    )

    batch = adapter.adapt_motion(motion)
    scalar = np.stack(
        [
            adapter.adapt_pose(
                KinematicPose(body_names, positions[frame], orientations[frame])
            ).positions_m
            for frame in range(frame_count)
        ]
    )

    np.testing.assert_allclose(batch.positions_m, scalar, rtol=2e-14, atol=2e-14)


def test_vectorized_surface_evaluation_matches_scalar_pose_methods() -> None:
    rng = np.random.default_rng(7229)
    frame_count = 83
    model = _model(child_origin_z=1.3, sole_z=-0.17)
    evaluator = SurfacePoseEvaluator(
        model,
        SOURCE_NAMES,
        MAPPING,
        (GroundingSurface("Child", ("sole",)),),
    )
    positions = rng.normal(size=(frame_count, len(SOURCE_NAMES), 3))
    orientations = rng.normal(size=(frame_count, len(SOURCE_NAMES), 4))
    orientations *= rng.uniform(0.1, 5.0, size=orientations.shape[:-1] + (1,))
    motion = KinematicMotion(
        SOURCE_NAMES,
        positions,
        orientations,
        np.arange(frame_count, dtype=float) * 0.03,
    )
    poses = [
        KinematicPose(SOURCE_NAMES, positions[frame], orientations[frame])
        for frame in range(frame_count)
    ]

    scalar_minimums = np.asarray([evaluator.minimum_height_m(pose) for pose in poses])
    scalar_references = np.asarray([evaluator.support_reference_m(pose) for pose in poses])

    np.testing.assert_allclose(
        evaluator.minimum_heights_m(motion),
        scalar_minimums,
        rtol=2e-14,
        atol=2e-14,
    )
    np.testing.assert_allclose(
        evaluator.support_references_m(motion),
        scalar_references,
        rtol=2e-14,
        atol=2e-14,
    )


def test_optional_grounding_aligns_lowest_surfaces_without_clamping_airborne_pose() -> None:
    source_model = _model(child_origin_z=1.0, sole_z=-0.2)
    target_model = _model(child_origin_z=2.0, sole_z=-0.1)
    surfaces = (GroundingSurface("Child", ("sole",)),)
    grounding = LowestSurfaceGrounding(
        source_model,
        target_model,
        SOURCE_NAMES,
        source_body_to_pose_body=MAPPING,
        target_body_to_pose_body=MAPPING,
        source_surfaces=surfaces,
        target_surfaces=surfaces,
    )
    grounded = KinematicMorphologyAdapter(
        target_model,
        SOURCE_NAMES,
        target_body_to_source_body=MAPPING,
        grounding=grounding,
    )
    ungrounded = KinematicMorphologyAdapter(target_model, SOURCE_NAMES, target_body_to_source_body=MAPPING)
    source = _pose(root_z=3.0, child_z=4.0)

    raw_target = ungrounded.adapt_pose(source)
    adapted = grounded.adapt_pose(source)

    source_minimum = 4.0 - 0.2
    target_minimum = adapted.positions_m[1, 2] - 0.1
    assert source_minimum > 0.0
    np.testing.assert_allclose(target_minimum, source_minimum)
    offsets = adapted.positions_m[:, 2] - raw_target.positions_m[:, 2]
    np.testing.assert_allclose(offsets, np.repeat(offsets[0], 2))
    np.testing.assert_array_equal(raw_target.positions_m[0], source.positions_m[0])


def test_adapter_rejects_non_bijective_mapping_and_invalid_pose_data() -> None:
    model = _model()
    with pytest.raises(ValueError, match="bijective"):
        KinematicMorphologyAdapter(
            model,
            SOURCE_NAMES,
            target_body_to_source_body={"Root": "RootStream", "Child": "RootStream"},
        )
    adapter = KinematicMorphologyAdapter(model, SOURCE_NAMES, target_body_to_source_body=MAPPING)
    with pytest.raises(ValueError, match="zero-length quaternion"):
        adapter.adapt_pose(
            KinematicPose(SOURCE_NAMES, _pose().positions_m, np.zeros((2, 4)))
        )


def test_adapter_rejects_shape_name_finiteness_and_timestamp_errors() -> None:
    adapter = KinematicMorphologyAdapter(_model(), SOURCE_NAMES, target_body_to_source_body=MAPPING)
    with pytest.raises(ValueError, match="body names/order"):
        adapter.adapt_pose(KinematicPose(tuple(reversed(SOURCE_NAMES)), _pose().positions_m, _pose().orientations_wxyz))
    with pytest.raises(ValueError, match="shapes"):
        adapter.adapt_pose(KinematicPose(SOURCE_NAMES, np.zeros((1, 3)), np.zeros((1, 4))))
    non_finite = _pose().positions_m.copy()
    non_finite[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        adapter.adapt_pose(KinematicPose(SOURCE_NAMES, non_finite, _pose().orientations_wxyz))
    with pytest.raises(ValueError, match="strictly increasing"):
        adapter.adapt_motion(
            KinematicMotion(
                SOURCE_NAMES,
                np.repeat(_pose().positions_m[None, :, :], 2, axis=0),
                np.repeat(_pose().orientations_wxyz[None, :, :], 2, axis=0),
                np.array([1.0, 1.0]),
            )
        )


def test_adapter_rejects_disconnected_target_topology() -> None:
    disconnected = KinematicTree(
        "disconnected",
        "Root",
        (
            RigidBodyDefinition("Root", Transform.identity()),
            RigidBodyDefinition("Child", Transform.identity()),
        ),
        (),
    )
    with pytest.raises(ValueError, match="no parent joint"):
        KinematicMorphologyAdapter(
            disconnected,
            SOURCE_NAMES,
            target_body_to_source_body=MAPPING,
        )


def test_grounding_rejects_missing_meshes() -> None:
    with pytest.raises(KeyError, match="has no mesh"):
        LowestSurfaceGrounding(
            _model(sole_z=-0.1),
            _model(sole_z=-0.1),
            SOURCE_NAMES,
            source_body_to_pose_body=MAPPING,
            target_body_to_pose_body=MAPPING,
            source_surfaces=(GroundingSurface("Child", ("missing",)),),
            target_surfaces=(GroundingSurface("Child", ("sole",)),),
        )


def test_reference_floor_clearance_uses_generic_mesh_surface_evaluator() -> None:
    source = _model(child_origin_z=-1.0, sole_z=0.0)
    target = _model(child_origin_z=-0.75, sole_z=0.0)

    np.testing.assert_allclose(reference_root_floor_clearance_m(source), 1.0)
    np.testing.assert_allclose(reference_root_floor_clearance_m(target), 0.75)
    np.testing.assert_allclose(reference_grounding_offset_m(source, target), -0.25)
