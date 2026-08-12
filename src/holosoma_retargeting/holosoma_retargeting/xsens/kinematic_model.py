"""Convert an embedded Xsens calibration into a generic kinematic tree."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from holosoma_retargeting.data_utils.xsens_hdf5 import XsensHdf5Calibration
from holosoma_retargeting.kinematics import (
    KinematicTree,
    RigidBodyDefinition,
    SphericalJointDefinition,
    Transform,
)
from holosoma_retargeting.kinematics.model import quaternion_conjugate, quaternion_multiply, rotate_vector

XSENS_RACKET_SOURCE_SEGMENT = "RightHandSword"
XSENS_RACKET_SOURCE_JOINT = "RightHandSwordOrigin"
TENNIS_RACKET_BODY = "TennisRacket"
TENNIS_RACKET_JOINT = "RightHandTennisRacketOrigin"


@dataclass(frozen=True)
class XsensJointSpec:
    source_joint: str
    parent_segment: str
    child_segment: str
    landmark: str


XSENS_JOINT_SPECS = (
    XsensJointSpec("L5S1", "Pelvis", "L5", "jL5S1"),
    XsensJointSpec("L4L3", "L5", "L3", "jL4L3"),
    XsensJointSpec("L1T12", "L3", "T12", "jL1T12"),
    XsensJointSpec("T9T8", "T12", "T8", "jT9T8"),
    XsensJointSpec("T1C7", "T8", "Neck", "jT1C7"),
    XsensJointSpec("C1Head", "Neck", "Head", "jC1Head"),
    XsensJointSpec("RightT4Shoulder", "T8", "RightShoulder", "jRightT4Shoulder"),
    XsensJointSpec("RightShoulder", "RightShoulder", "RightUpperArm", "jRightShoulder"),
    XsensJointSpec("RightElbow", "RightUpperArm", "RightForeArm", "jRightElbow"),
    XsensJointSpec("RightWrist", "RightForeArm", "RightHand", "jRightWrist"),
    XsensJointSpec("LeftT4Shoulder", "T8", "LeftShoulder", "jLeftT4Shoulder"),
    XsensJointSpec("LeftShoulder", "LeftShoulder", "LeftUpperArm", "jLeftShoulder"),
    XsensJointSpec("LeftElbow", "LeftUpperArm", "LeftForeArm", "jLeftElbow"),
    XsensJointSpec("LeftWrist", "LeftForeArm", "LeftHand", "jLeftWrist"),
    XsensJointSpec("RightHip", "Pelvis", "RightUpperLeg", "jRightHip"),
    XsensJointSpec("RightKnee", "RightUpperLeg", "RightLowerLeg", "jRightKnee"),
    XsensJointSpec("RightAnkle", "RightLowerLeg", "RightFoot", "jRightAnkle"),
    XsensJointSpec("RightBallFoot", "RightFoot", "RightToe", "jRightBallFoot"),
    XsensJointSpec("LeftHip", "Pelvis", "LeftUpperLeg", "jLeftHip"),
    XsensJointSpec("LeftKnee", "LeftUpperLeg", "LeftLowerLeg", "jLeftKnee"),
    XsensJointSpec("LeftAnkle", "LeftLowerLeg", "LeftFoot", "jLeftAnkle"),
    XsensJointSpec("LeftBallFoot", "LeftFoot", "LeftToe", "jLeftBallFoot"),
    XsensJointSpec(
        XSENS_RACKET_SOURCE_JOINT,
        "RightHand",
        XSENS_RACKET_SOURCE_SEGMENT,
        "",
    ),
)


def normalize_xsens_name(name: str) -> str:
    return "".join(character.lower() for character in name if character.isalnum())


def canonical_xsens_segment_name(source_name: str) -> str:
    if normalize_xsens_name(source_name) == normalize_xsens_name(XSENS_RACKET_SOURCE_SEGMENT):
        return TENNIS_RACKET_BODY
    return "".join(character for character in source_name if character.isalnum() or character == "_")


def canonical_xsens_joint_name(source_name: str) -> str:
    if normalize_xsens_name(source_name) == normalize_xsens_name(XSENS_RACKET_SOURCE_JOINT):
        return TENNIS_RACKET_JOINT
    return "".join(character for character in source_name if character.isalnum() or character == "_")


def calibration_fingerprint(calibration: XsensHdf5Calibration) -> str:
    """Hash the normalized model-defining calibration content."""

    digest = hashlib.sha256()
    for name in calibration.segment_names:
        digest.update(name.encode("utf-8"))
    for name in calibration.joint_names:
        digest.update(name.encode("utf-8"))
        rotation_metadata = calibration.joint_rotation_metadata.get(name)
        if rotation_metadata is not None:
            for component in rotation_metadata.components:
                digest.update(component.encode("utf-8"))
            for stream_name in rotation_metadata.available_euler_streams:
                digest.update(stream_name.encode("utf-8"))
    digest.update(np.asarray(calibration.tpose.positions_m, dtype="<f8").tobytes())
    digest.update(np.asarray(calibration.tpose.quaternions_wijk, dtype="<f8").tobytes())
    for segment_name in sorted(calibration.landmarks_m):
        digest.update(segment_name.encode("utf-8"))
        for landmark_name in sorted(calibration.landmarks_m[segment_name]):
            digest.update(landmark_name.encode("utf-8"))
            digest.update(np.asarray(calibration.landmarks_m[segment_name][landmark_name], dtype="<f8").tobytes())
    return digest.hexdigest()


def _source_index(names: tuple[str, ...], requested_name: str) -> int:
    requested = normalize_xsens_name(requested_name)
    matches = [index for index, name in enumerate(names) if normalize_xsens_name(name) == requested]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one Xsens entry named '{requested_name}', found {len(matches)}")
    return matches[0]


def _landmark(calibration: XsensHdf5Calibration, segment_name: str, landmark_name: str) -> np.ndarray:
    segment_index = _source_index(calibration.segment_names, segment_name)
    raw_segment_name = calibration.segment_names[segment_index]
    segment_landmarks = calibration.landmarks_m.get(raw_segment_name)
    if segment_landmarks is None or landmark_name not in segment_landmarks:
        raise KeyError(f"Missing calibrated landmark '{landmark_name}' on segment '{raw_segment_name}'")
    return np.asarray(segment_landmarks[landmark_name], dtype=float)


def build_xsens_kinematic_tree(
    calibration: XsensHdf5Calibration,
    *,
    include_tennis_racket: bool = True,
) -> KinematicTree:
    """Build the calibrated XSens articulation without fitting dynamic motion."""

    has_racket_segment = any(
        normalize_xsens_name(name) == normalize_xsens_name(XSENS_RACKET_SOURCE_SEGMENT)
        for name in calibration.segment_names
    )
    has_racket_joint = any(
        normalize_xsens_name(name) == normalize_xsens_name(XSENS_RACKET_SOURCE_JOINT)
        for name in calibration.joint_names
    )
    include_racket = include_tennis_racket and has_racket_segment and has_racket_joint

    bodies: list[RigidBodyDefinition] = []
    for source_index, source_name in enumerate(calibration.segment_names):
        is_racket = normalize_xsens_name(source_name) == normalize_xsens_name(XSENS_RACKET_SOURCE_SEGMENT)
        if is_racket and not include_racket:
            continue
        canonical_name = canonical_xsens_segment_name(source_name)
        body_metadata: dict[str, str | int | tuple[float, ...]] = {
            "xsens:sourceSegmentName": source_name,
            "xsens:sourceSegmentIndex": source_index,
        }
        if calibration.tpose_isb is not None:
            body_metadata["xsens:tposeIsbPositionM"] = tuple(
                float(value) for value in calibration.tpose_isb.positions_m[source_index]
            )
            body_metadata["xsens:tposeIsbOrientationWxyz"] = tuple(
                float(value) for value in calibration.tpose_isb.quaternions_wijk[source_index]
            )
        if calibration.identity_pose is not None:
            body_metadata["xsens:identityPositionM"] = tuple(
                float(value) for value in calibration.identity_pose.positions_m[source_index]
            )
            body_metadata["xsens:identityOrientationWxyz"] = tuple(
                float(value) for value in calibration.identity_pose.quaternions_wijk[source_index]
            )
        bodies.append(
            RigidBodyDefinition(
                name=canonical_name,
                reference_pose=Transform(
                    translation_m=np.asarray(calibration.tpose.positions_m[source_index], dtype=float),
                    rotation_wxyz=np.asarray(calibration.tpose.quaternions_wijk[source_index], dtype=float),
                ),
                metadata=body_metadata,
            )
        )

    joint_source_names = tuple(calibration.joint_names)
    body_map = {body.name: body for body in bodies}
    joints: list[SphericalJointDefinition] = []
    for spec in XSENS_JOINT_SPECS:
        is_racket = spec.source_joint == XSENS_RACKET_SOURCE_JOINT
        if is_racket and not include_racket:
            continue
        joint_index = _source_index(joint_source_names, spec.source_joint)
        child_index = _source_index(calibration.segment_names, spec.child_segment)
        parent_pose = body_map[canonical_xsens_segment_name(spec.parent_segment)].reference_pose
        child_pose = body_map[canonical_xsens_segment_name(spec.child_segment)].reference_pose

        if is_racket:
            world_delta = child_pose.translation_m - parent_pose.translation_m
            parent_anchor = rotate_vector(quaternion_conjugate(parent_pose.rotation_wxyz), world_delta)
            child_anchor = np.zeros(3, dtype=float)
        else:
            parent_anchor = _landmark(calibration, spec.parent_segment, spec.landmark)
            child_anchor = _landmark(calibration, spec.child_segment, spec.landmark)

        parent_to_child_rotation = quaternion_multiply(
            quaternion_conjugate(parent_pose.rotation_wxyz),
            child_pose.rotation_wxyz,
        )
        rotation_metadata = calibration.joint_rotation_metadata.get(spec.source_joint)
        metadata: dict[str, str | int | tuple[str, ...]] = {
            "xsens:sourceJointName": spec.source_joint,
            "xsens:sourceJointIndex": joint_index,
            "xsens:childSegmentIndex": child_index,
            "xsens:eulerStreams": calibration.joint_stream_names,
        }
        if rotation_metadata is not None:
            metadata["xsens:rotationComponents"] = rotation_metadata.components
        joints.append(
            SphericalJointDefinition(
                name=canonical_xsens_joint_name(spec.source_joint),
                parent_body=canonical_xsens_segment_name(spec.parent_segment),
                child_body=canonical_xsens_segment_name(spec.child_segment),
                parent_frame=Transform(parent_anchor, parent_to_child_rotation),
                child_frame=Transform(child_anchor, np.array([1.0, 0.0, 0.0, 0.0])),
                metadata=metadata,
            )
        )

    fingerprint = calibration_fingerprint(calibration)
    metadata: dict[str, str | int | tuple[str, ...]] = {
        "xsens:calibrationFingerprint": fingerprint,
        "xsens:sourceFile": calibration.source_path.name,
        "xsens:sourceStream": calibration.source_stream_name,
        "xsens:referencePose": calibration.tpose.variant,
        "xsens:jointOrder": tuple(canonical_xsens_joint_name(name) for name in calibration.joint_names),
    }
    if calibration.mvn_version is not None:
        metadata["xsens:mvnVersion"] = calibration.mvn_version
    if calibration.mvnx_version is not None:
        metadata["xsens:mvnxVersion"] = calibration.mvnx_version
    if calibration.tpose_isb is not None:
        metadata["xsens:availableReferencePoses"] = ("Tpose", "TposeISB")

    return KinematicTree(
        name="XsensAvatar",
        root_body="Pelvis",
        bodies=tuple(bodies),
        joints=tuple(joints),
        metadata=metadata,
    )
