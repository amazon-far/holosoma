"""Tests for the Xsens Viser actor."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
from holosoma_retargeting.data_utils.xsens_hdf5 import (
    XSENS_BODY_SEGMENT_NAMES,
    XsensHdf5Motion,
)
from holosoma_retargeting.kinematics import (
    KinematicMorphologyAdapter,
    KinematicMotion,
    KinematicPose,
    KinematicTree,
    MeshAttachment,
    RigidBodyDefinition,
    SphericalJointDefinition,
    Transform,
)
from holosoma_retargeting.src.viser_utils import sample_qpos_at_time
from holosoma_retargeting.src.xsens_viser import (
    XsensMotionSampler,
    XsensUsdActor,
    validate_g1_xsens_usd,
)
from holosoma_retargeting.xsens.g1_kinematic_reduction import G1_XSENS_REDUCTION_VERSION


class _Handle:
    def __init__(self, *, position=None, wxyz=None, visible=True, vertices=None):
        self.position = np.zeros(3) if position is None else np.asarray(position, dtype=float)
        self.wxyz = np.array([1.0, 0.0, 0.0, 0.0]) if wxyz is None else np.asarray(wxyz, dtype=float)
        self.visible = visible
        self.vertices = None if vertices is None else np.asarray(vertices, dtype=float)


class _Scene:
    def __init__(self):
        self.handles: dict[str, _Handle] = {}

    def add_frame(self, name, **kwargs):
        handle = _Handle(
            position=kwargs.get("position"),
            wxyz=kwargs.get("wxyz"),
            visible=kwargs.get("visible", True),
        )
        self.handles[name] = handle
        return handle

    def add_mesh_simple(self, name, vertices, _faces=None, **kwargs):
        handle = _Handle(visible=kwargs.get("visible", True), vertices=vertices)
        self.handles[name] = handle
        return handle

    def add_point_cloud(self, name, _points=None, _colors=None, **kwargs):
        handle = _Handle(visible=kwargs.get("visible", True))
        self.handles[name] = handle
        return handle


def _model(mesh_scale: float) -> KinematicTree:
    triangle = MeshAttachment(
        name="visual",
        vertices_m=mesh_scale * np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        faces=np.array([[0, 1, 2]]),
    )
    bodies = (
        RigidBodyDefinition(
            "Pelvis",
            Transform.identity(),
            meshes=(triangle,),
            metadata={"xsens:sourceSegmentName": "Pelvis", "model:proportionedFrom": "g1_29dof"},
        ),
        RigidBodyDefinition(
            "TennisRacket",
            Transform.identity(),
            meshes=(triangle,),
            metadata={"xsens:sourceSegmentName": "RightHandSword", "model:proportionedFrom": "g1_29dof"},
        ),
    )
    return KinematicTree(
        "test",
        "Pelvis",
        bodies,
        (),
        metadata={"model:proportionedFrom": "g1_29dof"},
    )


def test_actor_direct_pose_application_is_independent_of_visual_proportions() -> None:
    subject_server = SimpleNamespace(scene=_Scene())
    g1_server = SimpleNamespace(scene=_Scene())
    subject = XsensUsdActor(subject_server, "subject.usda", model=_model(1.0))
    g1 = XsensUsdActor(g1_server, "g1.usda", model=_model(2.0))
    names = ("Pelvis", "RightHandSword")
    positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    quaternions = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])

    subject.apply_pose(names, positions, quaternions)
    g1.apply_pose(names, positions, quaternions)

    for body_name in ("Pelvis", "TennisRacket"):
        position_index = 0 if body_name == "Pelvis" else 1
        np.testing.assert_array_equal(subject.body_frames[body_name].position, positions[position_index])
        np.testing.assert_array_equal(g1.body_frames[body_name].position, positions[position_index])
        np.testing.assert_array_equal(subject.body_frames[body_name].wxyz, g1.body_frames[body_name].wxyz)
    assert not np.array_equal(subject.mesh_handles[0].vertices, g1.mesh_handles[0].vertices)


def _two_segment_model() -> KinematicTree:
    bodies = (
        RigidBodyDefinition(
            "Pelvis",
            Transform(np.array([0.0, 0.0, 1.0])),
            metadata={"xsens:sourceSegmentName": "Pelvis"},
        ),
        RigidBodyDefinition(
            "Child",
            Transform(np.array([0.0, 0.0, 2.0])),
            metadata={"xsens:sourceSegmentName": "Child"},
        ),
    )
    joint = SphericalJointDefinition(
        "Joint",
        parent_body="Pelvis",
        child_body="Child",
        parent_frame=Transform(np.array([0.0, 0.0, 0.5])),
        child_frame=Transform(np.array([0.0, 0.0, -0.5])),
    )
    return KinematicTree("two_segment", "Pelvis", bodies, (joint,))


def test_native_timestamp_sampler_results_match_batch_morphology_adaptation() -> None:
    model = _two_segment_model()
    mapping = {"Pelvis": "Pelvis", "Child": "Child"}
    adapter = KinematicMorphologyAdapter(
        model,
        ("Pelvis", "Child"),
        target_body_to_source_body=mapping,
    )
    positions = np.array(
        [
            [[0.0, 0.0, 1.0], [0.0, 0.0, 99.0]],
            [[1.0, 2.0, 3.0], [4.0, 5.0, -99.0]],
        ]
    )
    orientations = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (2, 2, 1))
    times_s = np.array([10.0, 12.0])
    motion = XsensHdf5Motion(
        positions_m=positions,
        times_s=times_s,
        stream_name="body_position_xyz_m",
        segment_names=["Pelvis", "Child"],
        source_indices=[0, 1],
        quaternions_wijk=orientations,
        orientation_stream_name="body_orientation_quaternion_wijk",
    )
    sampler = XsensMotionSampler(motion)
    batch = adapter.adapt_motion(
        KinematicMotion(("Pelvis", "Child"), positions, orientations, times_s)
    )

    for frame_index, native_time_s in enumerate(times_s):
        sample = sampler.sample(native_time_s - times_s[0])
        adapted = adapter.adapt_pose(
            KinematicPose(sampler.segment_names, sample.positions_m, sample.quaternions_wxyz)
        )
        np.testing.assert_array_equal(adapted.positions_m, batch.positions_m[frame_index])
        np.testing.assert_array_equal(adapted.orientations_wxyz, batch.orientations_wxyz[frame_index])


def test_motion_sampler_uses_timestamps_and_shortest_arc_slerp() -> None:
    motion = XsensHdf5Motion(
        positions_m=np.array([[[0.0, 0.0, 0.0]], [[2.0, 4.0, 6.0]]]),
        times_s=np.array([10.0, 12.0]),
        stream_name="body_position_xyz_m",
        segment_names=["Pelvis"],
        source_indices=[0],
        quaternions_wijk=np.array([[[1.0, 0.0, 0.0, 0.0]], [[-1.0, 0.0, 0.0, 0.0]]]),
        orientation_stream_name="body_orientation_quaternion_wijk",
    )
    sampler = XsensMotionSampler(motion)

    sample = sampler.sample(1.0)

    np.testing.assert_allclose(sample.positions_m[0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(sample.quaternions_wxyz[0], [1.0, 0.0, 0.0, 0.0])
    assert sampler.duration_s == 2.0


def test_qpos_sampler_slerps_robot_and_object_quaternions() -> None:
    qpos = np.zeros((2, 7 + 2 + 7))
    qpos[:, 3] = [1.0, -1.0]
    qpos[:, -4] = [1.0, -1.0]
    qpos[1, :3] = [2.0, 4.0, 6.0]
    qpos[1, 7:9] = [2.0, 4.0]

    sampled = sample_qpos_at_time(qpos, 0.5, fps=1.0, robot_dof=2, has_object_input=True)

    np.testing.assert_allclose(sampled[:3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(sampled[3:7], [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(sampled[7:9], [1.0, 2.0])
    np.testing.assert_allclose(sampled[-4:], [1.0, 0.0, 0.0, 0.0])


def test_g1_validation_requires_all_body_and_racket_mappings() -> None:
    bodies = tuple(
        RigidBodyDefinition(
            name.replace(" ", ""),
            Transform.identity(),
            metadata={"xsens:sourceSegmentName": name, "model:proportionedFrom": "g1_29dof"},
        )
        for name in [*XSENS_BODY_SEGMENT_NAMES, "RightHandSword"]
    )
    bodies = (*bodies[:-1], RigidBodyDefinition(
        "TennisRacket",
        Transform.identity(),
        metadata={"xsens:sourceSegmentName": "RightHandSword", "model:proportionedFrom": "g1_29dof"},
    ))
    model = KinematicTree(
        "g1",
        "Pelvis",
        bodies,
        (),
        metadata={
            "model:proportionedFrom": "g1_29dof",
            "model:generatorVersion": G1_XSENS_REDUCTION_VERSION,
        },
    )

    validate_g1_xsens_usd(model)

    stale_model = replace(
        model,
        metadata={**model.metadata, "model:generatorVersion": "stale"},
    )
    with np.testing.assert_raises_regex(ValueError, "incompatible model generator"):
        validate_g1_xsens_usd(stale_model)
