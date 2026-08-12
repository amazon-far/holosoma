"""Procedural, subject-proportioned meshes for an Xsens segment avatar.

The mesh factory intentionally produces one rigid mesh per Xsens segment.  The
dynamic mapping from HDF5 poses to those rigid parts lives outside this module;
this module only loads the subject's static T-pose/proportion metadata and
builds geometry in each segment's local coordinate frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh  # type: ignore[import-untyped]

from holosoma_retargeting.data_utils.xsens_hdf5 import XsensHdf5Calibration, load_xsens_hdf5_calibration
from holosoma_retargeting.kinematics.model import quaternion_multiply, rotate_vector
from holosoma_retargeting.xsens.kinematic_model import XSENS_RACKET_SOURCE_SEGMENT

LIGHT_GRAY = (190, 196, 197)
MID_GRAY = (103, 112, 115)
DARK_GRAY = (39, 45, 47)
ACCENT_ORANGE = (225, 72, 35)
RACKET_FRAME = (49, 56, 59)
RACKET_GRIP = (118, 78, 48)
RACKET_STRINGS = (218, 220, 211)
NAIL_LIGHT = (238, 229, 211)


@dataclass(frozen=True)
class XsensAvatarProportions:
    """Subject-specific Xsens T-pose and local anatomical landmarks."""

    segment_names: tuple[str, ...]
    tpose_positions_m: np.ndarray
    tpose_quaternions_wijk: np.ndarray
    landmarks_m: dict[str, dict[str, np.ndarray]]

    def segment_index(self, name: str) -> int:
        return self.segment_names.index(name)


@dataclass(frozen=True)
class AvatarMeshPart:
    """A colored mesh attached to a single Xsens segment frame."""

    name: str
    mesh: trimesh.Trimesh
    color: tuple[int, int, int]
    category: str = "shell"


def avatar_proportions_from_calibration(
    calibration: XsensHdf5Calibration,
    variant: str = "Tpose",
) -> XsensAvatarProportions:
    """Adapt the shared calibration representation for procedural geometry."""

    pose = {
        "Tpose": calibration.tpose,
        "TposeISB": calibration.tpose_isb,
        "identity": calibration.identity_pose,
    }.get(variant)
    if pose is None:
        raise KeyError(f"Xsens calibration does not contain reference-pose variant '{variant}'")
    return XsensAvatarProportions(
        segment_names=calibration.segment_names,
        tpose_positions_m=pose.positions_m,
        tpose_quaternions_wijk=pose.quaternions_wijk,
        landmarks_m={segment: dict(values) for segment, values in calibration.landmarks_m.items()},
    )


def load_xsens_avatar_proportions(path: str | Path, variant: str = "Tpose") -> XsensAvatarProportions:
    """Compatibility wrapper around the shared HDF5 calibration loader."""

    return avatar_proportions_from_calibration(load_xsens_hdf5_calibration(path), variant=variant)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-10:
        raise ValueError("Cannot normalize a zero-length vector")
    return np.asarray(vector, dtype=float) / norm


def _basis_for_axis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    direction = _unit(axis)
    reference = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(direction, reference))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    cross_u = _unit(np.cross(direction, reference))
    cross_v = _unit(np.cross(direction, cross_u))
    return direction, cross_u, cross_v


def _elliptical_frustum(
    start: np.ndarray,
    end: np.ndarray,
    start_radii: tuple[float, float],
    end_radii: tuple[float, float],
    *,
    sections: int = 14,
    inset: float = 0.0,
) -> trimesh.Trimesh:
    """Create a capped tapered shell between two points in local coordinates."""

    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    axis = end - start
    length = float(np.linalg.norm(axis))
    direction, basis_u, basis_v = _basis_for_axis(axis)
    inset_m = min(max(inset, 0.0), length * 0.2)
    start = start + direction * inset_m
    end = end - direction * inset_m

    vertices: list[np.ndarray] = []
    for center, radii in ((start, start_radii), (end, end_radii)):
        vertices.extend(
            center + basis_u * np.cos(theta) * radii[0] + basis_v * np.sin(theta) * radii[1]
            for theta in np.linspace(0.0, 2.0 * np.pi, sections, endpoint=False)
        )
    vertices.extend([start, end])

    faces: list[tuple[int, int, int]] = []
    for i in range(sections):
        j = (i + 1) % sections
        faces.append((i, j, sections + j))
        faces.append((i, sections + j, sections + i))
        faces.append((2 * sections, j, i))
        faces.append((2 * sections + 1, sections + i, sections + j))
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=True)
    mesh.fix_normals()
    return mesh


def _ellipsoid(center: np.ndarray, radii: tuple[float, float, float], subdivisions: int = 2) -> trimesh.Trimesh:
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1.0)
    mesh.apply_scale(np.asarray(radii, dtype=float))
    mesh.apply_translation(np.asarray(center, dtype=float))
    return mesh


def cylinder_between(start: np.ndarray, end: np.ndarray, radius: float, sections: int = 12) -> trimesh.Trimesh:
    """Create a cylinder aligned between two points in local coordinates."""

    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    direction = end - start
    length = float(np.linalg.norm(direction))
    mesh = trimesh.creation.cylinder(radius=radius, height=length, sections=sections)
    transform = trimesh.geometry.align_vectors(np.array([0.0, 0.0, 1.0]), direction)
    if transform is None:
        transform = np.eye(4)
    transform[:3, 3] = (start + end) * 0.5
    mesh.apply_transform(transform)
    return mesh


_DISTAL_LANDMARKS = {
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


def _distal(proportions: XsensAvatarProportions, segment: str) -> np.ndarray:
    landmark_name = _DISTAL_LANDMARKS[segment]
    try:
        return proportions.landmarks_m[segment][landmark_name]
    except KeyError as exc:
        raise KeyError(f"Missing distal landmark {segment}/{landmark_name}") from exc


def _landmark(proportions: XsensAvatarProportions, segment: str, *names: str) -> np.ndarray:
    """Return the first available spelling of an Xsens mesh landmark."""

    segment_landmarks = proportions.landmarks_m[segment]
    for name in names:
        if name in segment_landmarks:
            return segment_landmarks[name]
    raise KeyError(f"Missing one of {names} in Xsens landmarks for {segment}")


def _convex_shell(points: list[np.ndarray]) -> trimesh.Trimesh:
    """Create a closed, consistently triangulated hull from anatomical points."""

    mesh = trimesh.convex.convex_hull(np.asarray(points, dtype=float))
    mesh.fix_normals()
    return mesh


def _trapezoid_prism(
    rear_x: float,
    front_x: float,
    rear_center_y: float,
    front_center_y: float,
    rear_half_width: float,
    front_half_width: float,
    bottom_z: float,
    thickness: float,
) -> trimesh.Trimesh:
    """Create a thin, flat-soled trapezoid used for shoe outsole panels."""

    footprint = np.array(
        [
            [rear_x, rear_center_y - rear_half_width],
            [rear_x, rear_center_y + rear_half_width],
            [front_x, front_center_y + front_half_width],
            [front_x, front_center_y - front_half_width],
        ]
    )
    vertices = np.vstack(
        [
            np.column_stack([footprint, np.full(4, bottom_z)]),
            np.column_stack([footprint, np.full(4, bottom_z + thickness)]),
        ]
    )
    faces = np.array(
        [
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
        ]
    )
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    mesh.fix_normals()
    return mesh


def _build_foot_parts(
    proportions: XsensAvatarProportions,
    side: str,
) -> tuple[AvatarMeshPart, AvatarMeshPart]:
    """Build an ankle-to-ball shoe shell from the recorded Xsens landmarks."""

    segment = f"{side}Foot"
    ankle = np.zeros(3)
    ball = _landmark(proportions, segment, f"j{side}BallFoot")
    heel = _landmark(proportions, segment, f"p{side}HeelCenter", f"p{side}HeelFoot")
    first = _landmark(proportions, segment, f"p{side}FirstMetatarsal")
    fifth = _landmark(proportions, segment, f"p{side}FifthMetatarsal")
    top = _landmark(proportions, segment, f"p{side}TopOfFoot")

    fore_center = (first + fifth) * 0.5
    half_width = max(0.025, abs(float(first[1] - fifth[1])) * 0.5)
    heel_half_width = half_width * 0.62
    ankle_half_width = half_width * 0.48
    top_half_width = half_width * 0.54
    sole_z = min(float(heel[2]), float(first[2]), float(fifth[2]))
    heel_upper_z = min(float(ankle[2]) - 0.012, sole_z + max(0.035, half_width * 0.85))
    ankle_upper_z = max(sole_z + 0.045, float(ankle[2]) - 0.006)
    top_z = max(float(top[2]), sole_z + max(0.045, half_width * 1.05))

    shell = _convex_shell(
        [
            np.array([heel[0], heel[1] - heel_half_width, sole_z]),
            np.array([heel[0], heel[1] + heel_half_width, sole_z]),
            np.array([heel[0], heel[1] - heel_half_width * 0.88, heel_upper_z]),
            np.array([heel[0], heel[1] + heel_half_width * 0.88, heel_upper_z]),
            np.array([ankle[0], ankle[1] - ankle_half_width, ankle_upper_z]),
            np.array([ankle[0], ankle[1] + ankle_half_width, ankle_upper_z]),
            first,
            fifth,
            np.array([ball[0], fore_center[1] - half_width * 0.82, ball[2]]),
            np.array([ball[0], fore_center[1] + half_width * 0.82, ball[2]]),
            np.array([top[0], top[1] - top_half_width, top_z]),
            np.array([top[0], top[1] + top_half_width, top_z]),
        ]
    )
    outsole = _trapezoid_prism(
        rear_x=float(heel[0]),
        front_x=max(float(first[0]), float(fifth[0]), float(ball[0])),
        rear_center_y=float(heel[1]),
        front_center_y=float(fore_center[1]),
        rear_half_width=heel_half_width,
        front_half_width=half_width,
        bottom_z=sole_z - 0.004,
        thickness=0.009,
    )
    return (
        AvatarMeshPart(f"{segment.lower()}_shoe", shell, MID_GRAY),
        AvatarMeshPart(f"{segment.lower()}_outsole", outsole, DARK_GRAY, "panel"),
    )


def _build_toe_parts(
    proportions: XsensAvatarProportions,
    side: str,
) -> tuple[AvatarMeshPart, AvatarMeshPart]:
    """Build the toe box, inheriting its width from the subject's foot."""

    segment = f"{side}Toe"
    foot_segment = f"{side}Foot"
    tip = _landmark(proportions, segment, f"p{side}Toe")
    first = _landmark(proportions, foot_segment, f"p{side}FirstMetatarsal")
    fifth = _landmark(proportions, foot_segment, f"p{side}FifthMetatarsal")
    half_width = max(0.025, abs(float(first[1] - fifth[1])) * 0.5)
    root_half_width = half_width * 0.88
    tip_half_width = half_width * 0.62
    center_y = float((first[1] + fifth[1]) * 0.5)
    sole_z = min(float(tip[2]) - 0.004, -0.012)
    root_top_z = max(0.022, sole_z + half_width * 0.9)
    tip_top_z = max(float(tip[2]) + 0.018, sole_z + half_width * 0.6)

    shell = _convex_shell(
        [
            np.array([0.0, center_y - root_half_width, sole_z]),
            np.array([0.0, center_y + root_half_width, sole_z]),
            np.array([0.0, center_y - root_half_width * 0.82, root_top_z]),
            np.array([0.0, center_y + root_half_width * 0.82, root_top_z]),
            np.array([tip[0], center_y - tip_half_width, sole_z]),
            np.array([tip[0], center_y + tip_half_width, sole_z]),
            np.array([tip[0], center_y - tip_half_width * 0.78, tip_top_z]),
            np.array([tip[0], center_y + tip_half_width * 0.78, tip_top_z]),
        ]
    )
    outsole = _trapezoid_prism(
        rear_x=0.0,
        front_x=float(tip[0]),
        rear_center_y=center_y,
        front_center_y=center_y,
        rear_half_width=root_half_width,
        front_half_width=tip_half_width,
        bottom_z=sole_z - 0.004,
        thickness=0.009,
    )
    return (
        AvatarMeshPart(f"{segment.lower()}_toe_box", shell, LIGHT_GRAY),
        AvatarMeshPart(f"{segment.lower()}_outsole", outsole, DARK_GRAY, "panel"),
    )


def _build_hand_parts(
    proportions: XsensAvatarProportions,
    side: str,
    *,
    sections: int,
) -> tuple[AvatarMeshPart, ...]:
    """Build a compact palm, four fingers, and a forward-pointing thumb."""

    segment = f"{side}Hand"
    top = _landmark(proportions, segment, f"p{side}TopOfHand")
    pinky = _landmark(proportions, segment, f"p{side}Pinky")
    palm = _landmark(proportions, segment, f"p{side}HandPalm")
    direction = 1.0 if float(top[1]) >= 0.0 else -1.0
    hand_length = abs(float(top[1]))
    knuckle_distance = min(hand_length * 0.68, abs(float(pinky[1])))
    palm_center_x = float((pinky[0] + palm[0]) * 0.5)
    palm_half_width = max(0.026, abs(float(palm[0] - pinky[0])) * 0.62)
    palm_half_thickness = max(0.012, abs(float(palm[2])) * 1.25)

    palm_start = np.array([palm_center_x, direction * 0.012, float(palm[2]) * 0.35])
    palm_end = np.array([palm_center_x, direction * knuckle_distance, float(palm[2]) * 0.65])
    palm_length = float(np.linalg.norm(palm_end - palm_start))
    palm_center = (palm_start + palm_end) * 0.5
    dorsal_surface_z = max(
        float(palm_start[2]) + palm_half_thickness,
        float(palm_end[2]) + palm_half_thickness * 0.82,
    )
    palmar_surface_z = min(
        float(palm_start[2]) - palm_half_thickness,
        float(palm_end[2]) - palm_half_thickness * 0.82,
    )
    parts: list[AvatarMeshPart] = [
        AvatarMeshPart(
            f"{segment.lower()}_palm",
            _elliptical_frustum(
                palm_start,
                palm_end,
                (palm_half_thickness, palm_half_width),
                (palm_half_thickness * 0.82, palm_half_width * 0.92),
                sections=sections,
            ),
            ACCENT_ORANGE,
        ),
        # Xsens requires fingernails/dorsum to face +Z and the palm to face
        # -Z. These thin surface markers make that convention unambiguous
        # without changing the calibrated hand frame or silhouette.
        AvatarMeshPart(
            f"{segment.lower()}_dorsal_panel",
            _ellipsoid(
                np.array([palm_center[0], palm_center[1], dorsal_surface_z + 0.0005]),
                (palm_half_width * 0.52, palm_length * 0.27, 0.0012),
                subdivisions=1,
            ),
            NAIL_LIGHT,
            "orientation_cue",
        ),
        AvatarMeshPart(
            f"{segment.lower()}_palm_pad",
            _ellipsoid(
                np.array([palm_center[0], palm_center[1], palmar_surface_z - 0.0005]),
                (palm_half_width * 0.58, palm_length * 0.31, 0.0014),
                subdivisions=1,
            ),
            DARK_GRAY,
            "orientation_cue",
        ),
    ]

    finger_x = palm_center_x + palm_half_width * np.array([-0.64, -0.22, 0.22, 0.64])
    finger_length_factors = (0.80, 0.94, 1.0, 0.89)
    finger_radius = max(0.0048, palm_half_width * 0.16)
    finger_start_y = direction * knuckle_distance * 0.92
    for index, (x, length_factor) in enumerate(zip(finger_x, finger_length_factors, strict=True)):
        finger_end_y = direction * hand_length * length_factor
        start_x = palm_center_x + (float(x) - palm_center_x) * 0.92
        end_x = palm_center_x + (float(x) - palm_center_x) * 1.14
        finger_start = np.array([start_x, finger_start_y, 0.0])
        finger_end = np.array([end_x, finger_end_y, 0.0])
        finger = _elliptical_frustum(
            finger_start,
            finger_end,
            (finger_radius * 0.88, finger_radius),
            (finger_radius * 0.70, finger_radius * 0.78),
            sections=max(8, sections - 2),
        )
        parts.append(AvatarMeshPart(f"{segment.lower()}_finger_{index + 1}", finger, ACCENT_ORANGE))
        nail_axis = finger_end - finger_start
        nail_start = finger_end - nail_axis * 0.23
        nail_end = finger_end - nail_axis * 0.05
        nail_start[2] = finger_radius * 0.76
        nail_end[2] = finger_radius * 0.76
        parts.append(
            AvatarMeshPart(
                f"{segment.lower()}_fingernail_{index + 1}",
                _elliptical_frustum(
                    nail_start,
                    nail_end,
                    (finger_radius * 0.11, finger_radius * 0.52),
                    (finger_radius * 0.08, finger_radius * 0.43),
                    sections=max(8, sections - 2),
                ),
                NAIL_LIGHT,
                "orientation_cue",
            )
        )
        joint_end = finger_start + _unit(finger_end - finger_start) * 0.007
        parts.append(
            AvatarMeshPart(
                f"{segment.lower()}_finger_joint_{index + 1}",
                _elliptical_frustum(
                    finger_start,
                    joint_end,
                    (finger_radius * 1.05, finger_radius * 1.12),
                    (finger_radius * 1.02, finger_radius * 1.08),
                    sections=max(8, sections - 2),
                ),
                DARK_GRAY,
                "panel",
            )
        )

    # In the Xsens T-pose both thumbs point character-forward (+X), while the
    # remaining fingers continue laterally along +/-Y.
    thumb_start = np.array(
        [palm_center_x + palm_half_width * 0.72, direction * knuckle_distance * 0.48, float(palm[2]) * 0.25]
    )
    thumb_end = thumb_start + np.array(
        [max(0.042, palm_half_width * 1.35), direction * palm_half_width * 0.42, 0.0]
    )
    thumb = _elliptical_frustum(
        thumb_start,
        thumb_end,
        (finger_radius * 1.12, finger_radius * 1.12),
        (finger_radius * 0.76, finger_radius * 0.76),
        sections=max(8, sections - 2),
    )
    parts.append(AvatarMeshPart(f"{segment.lower()}_thumb", thumb, ACCENT_ORANGE))
    thumb_axis = thumb_end - thumb_start
    thumbnail_start = thumb_end - thumb_axis * 0.28
    thumbnail_end = thumb_end - thumb_axis * 0.06
    thumbnail_z = float(thumb_start[2]) + finger_radius * 1.18
    thumbnail_start[2] = thumbnail_z
    thumbnail_end[2] = thumbnail_z
    parts.append(
        AvatarMeshPart(
            f"{segment.lower()}_thumbnail",
            _elliptical_frustum(
                thumbnail_start,
                thumbnail_end,
                (finger_radius * 0.11, finger_radius * 0.58),
                (finger_radius * 0.08, finger_radius * 0.46),
                sections=max(8, sections - 2),
            ),
            NAIL_LIGHT,
            "orientation_cue",
        )
    )
    return tuple(parts)


def _build_leg_parts(
    proportions: XsensAvatarProportions,
    side: str,
    kind: str,
    *,
    sections: int,
) -> tuple[AvatarMeshPart, ...]:
    """Build slim landmark-calibrated Xsens-style upper or lower leg shells."""

    segment = f"{side}{kind}Leg"
    end = _distal(proportions, segment)
    outer_sign = -1.0 if side == "Right" else 1.0

    if kind == "Upper":
        trochanter = _landmark(proportions, segment, f"p{side}GreaterTrochanter")
        lateral = _landmark(proportions, segment, f"p{side}KneeLatEpicondyle")
        medial = _landmark(proportions, segment, f"p{side}KneeMedEpicondyle")
        patella = _landmark(proportions, segment, f"p{side}Patella")
        start_lateral = max(0.045, abs(float(trochanter[1])) * 0.78)
        end_lateral = max(0.030, abs(float(lateral[1] - medial[1])) * 0.41)
        start_depth = max(0.045, abs(float(patella[0])) * 1.25)
        end_depth = max(0.032, abs(float(patella[0])) * 0.92)
        shell = _elliptical_frustum(
            np.zeros(3),
            end,
            (start_lateral, start_depth),
            (end_lateral, end_depth),
            sections=sections,
            inset=min(0.012, float(np.linalg.norm(end)) * 0.035),
        )
        accent_start = end * 0.06 + np.array([0.0, outer_sign * start_lateral * 0.93, 0.0])
        accent_end = end * 0.92 + np.array([0.0, outer_sign * end_lateral * 0.93, 0.0])
        collar_radii = (end_lateral, end_depth)
    else:
        lateral = _landmark(proportions, segment, f"p{side}LatMalleolus")
        medial = _landmark(proportions, segment, f"p{side}MedMalleolus")
        tibial = _landmark(proportions, segment, f"p{side}TibialTub")
        fibula = _landmark(proportions, segment, f"p{side}Fibula")
        ankle_lateral = max(0.023, abs(float(lateral[1] - medial[1])) * 0.39)
        knee_lateral = max(0.032, abs(float(fibula[1])) * 1.02)
        knee_depth = max(0.034, abs(float(tibial[0])) * 0.72)
        calf_lateral = knee_lateral * 1.10
        calf_depth = knee_depth * 1.12
        ankle_depth = max(0.025, ankle_lateral * 1.05)
        calf_center = end * 0.34
        shell = trimesh.util.concatenate(
            [
                _elliptical_frustum(
                    np.zeros(3),
                    calf_center,
                    (knee_lateral, knee_depth),
                    (calf_lateral, calf_depth),
                    sections=sections,
                    inset=0.005,
                ),
                _elliptical_frustum(
                    calf_center,
                    end,
                    (calf_lateral, calf_depth),
                    (ankle_lateral, ankle_depth),
                    sections=sections,
                    inset=0.005,
                ),
            ]
        )
        accent_start = end * 0.06 + np.array([0.0, outer_sign * knee_lateral * 0.93, 0.0])
        accent_end = end * 0.94 + np.array([0.0, outer_sign * ankle_lateral * 0.93, 0.0])
        collar_radii = (ankle_lateral, ankle_depth)

    return (
        AvatarMeshPart(f"{segment.lower()}_shell", shell, MID_GRAY),
        AvatarMeshPart(
            f"{segment.lower()}_outer_stripe",
            cylinder_between(accent_start, accent_end, 0.0045, sections=8),
            ACCENT_ORANGE,
            "accent",
        ),
        AvatarMeshPart(
            f"{segment.lower()}_joint_collar",
            _collar(end, np.zeros(3), collar_radii, sections=sections),
            ACCENT_ORANGE,
            "accent",
        ),
    )


def expected_racket_tpose_from_right_hand(
    proportions: XsensAvatarProportions,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive the expected racket pose for validating the recorded prop pose.

    The calibrated ``RightHandSword`` segment pose remains authoritative for
    rendering and export.  This independent anatomical construction is useful
    for detecting recordings whose tracked prop frame is inconsistent with the
    right-palm landmark and Xsens T-pose convention.
    """

    hand_index = proportions.segment_index("RightHand")
    hand_position = proportions.tpose_positions_m[hand_index]
    hand_orientation = proportions.tpose_quaternions_wijk[hand_index]
    palm_local = _landmark(proportions, "RightHand", "pRightHandPalm")
    grip_position = hand_position + rotate_vector(hand_orientation, palm_local)
    # The recorded prop frame uses a -90-degree roll about character-forward
    # (+X). The visual racket mesh applies the inverse local roll so its string
    # plane is horizontal without altering this calibrated segment frame.
    recorded_prop_roll = np.array([np.sqrt(0.5), -np.sqrt(0.5), 0.0, 0.0])
    grip_orientation = quaternion_multiply(hand_orientation, recorded_prop_roll)
    return grip_position, grip_orientation


def _collar(
    end: np.ndarray,
    start: np.ndarray,
    radii: tuple[float, float],
    *,
    sections: int,
) -> trimesh.Trimesh:
    direction = _unit(np.asarray(end) - np.asarray(start))
    length = min(0.025, float(np.linalg.norm(np.asarray(end) - np.asarray(start))) * 0.12)
    collar_end = np.asarray(end) - direction * length * 0.25
    collar_start = collar_end - direction * length
    return _elliptical_frustum(
        collar_start,
        collar_end,
        (radii[0] * 1.08, radii[1] * 1.08),
        (radii[0] * 1.08, radii[1] * 1.08),
        sections=sections,
    )


def build_xsens_avatar_meshes(
    proportions: XsensAvatarProportions,
    *,
    sections: int = 14,
) -> dict[str, tuple[AvatarMeshPart, ...]]:
    """Build rigid local meshes for every body segment in the Xsens model."""

    parts: dict[str, list[AvatarMeshPart]] = {name: [] for name in proportions.segment_names}
    landmarks = proportions.landmarks_m

    pelvis_array = np.asarray(list(landmarks["Pelvis"].values()))
    pelvis_min, pelvis_max = pelvis_array.min(axis=0), pelvis_array.max(axis=0)
    hip_width = float(np.linalg.norm(landmarks["Pelvis"]["jLeftHip"] - landmarks["Pelvis"]["jRightHip"]))
    pelvis_depth = max(float(pelvis_max[0] - pelvis_min[0]) * 0.5, hip_width * 0.34)
    pelvis_start_width = max(float(pelvis_max[1] - pelvis_min[1]) * 0.42, hip_width * 0.52)
    pelvis_end_width = max(float(pelvis_max[1] - pelvis_min[1]) * 0.35, hip_width * 0.44)
    pelvis_end = landmarks["Pelvis"]["jL5S1"] * 0.96
    pelvis_start = np.array([0.0, 0.0, min(-hip_width * 0.22, float(pelvis_min[2]) * 0.65)])
    pelvis = _elliptical_frustum(
        pelvis_start,
        pelvis_end,
        (pelvis_start_width, pelvis_depth * 0.9),
        (pelvis_end_width, pelvis_depth * 0.78),
        sections=sections,
    )
    parts["Pelvis"].append(AvatarMeshPart("pelvis_shell", pelvis, LIGHT_GRAY))
    pelvis_panel = _elliptical_frustum(
        pelvis_start + np.array([-pelvis_depth * 0.86, 0.0, 0.015]),
        pelvis_end + np.array([-pelvis_depth * 0.72, 0.0, -0.012]),
        (pelvis_start_width * 0.67, pelvis_depth * 0.22),
        (pelvis_end_width * 0.67, pelvis_depth * 0.19),
        sections=sections,
    )
    parts["Pelvis"].append(AvatarMeshPart("pelvis_panel", pelvis_panel, DARK_GRAY, "panel"))

    shoulder_width = float(
        np.linalg.norm(
            proportions.tpose_positions_m[proportions.segment_index("LeftUpperArm")]
            - proportions.tpose_positions_m[proportions.segment_index("RightUpperArm")]
        )
    )

    spine_profiles = {
        "L5": (hip_width * 0.48, hip_width * 0.32, hip_width * 0.42, hip_width * 0.29),
        "L3": (hip_width * 0.42, hip_width * 0.29, hip_width * 0.45, hip_width * 0.31),
        "T12": (hip_width * 0.45, hip_width * 0.31, shoulder_width * 0.34, hip_width * 0.36),
        "T8": (shoulder_width * 0.36, hip_width * 0.38, shoulder_width * 0.46, hip_width * 0.43),
    }
    for segment, profile in spine_profiles.items():
        end = _distal(proportions, segment)
        shell = _elliptical_frustum(
            np.zeros(3),
            end,
            (profile[0], profile[1]),
            (profile[2], profile[3]),
            sections=sections,
            inset=float(np.linalg.norm(end)) * 0.045,
        )
        color = LIGHT_GRAY if segment in {"L5", "T12", "T8"} else MID_GRAY
        parts[segment].append(AvatarMeshPart(f"{segment.lower()}_shell", shell, color))
        panel = _elliptical_frustum(
            np.asarray(end) * 0.18 + np.array([-profile[1] * 0.48, 0.0, 0.0]),
            np.asarray(end) * 0.82 + np.array([-profile[3] * 0.48, 0.0, 0.0]),
            (profile[0] * 0.74, profile[1] * 0.16),
            (profile[2] * 0.74, profile[3] * 0.16),
            sections=sections,
        )
        parts[segment].append(AvatarMeshPart(f"{segment.lower()}_rear_panel", panel, DARK_GRAY, "panel"))

    neck_end = _distal(proportions, "Neck")
    neck_length = float(np.linalg.norm(neck_end))
    neck_r = max(0.035, neck_length * 0.38)
    parts["Neck"].append(
        AvatarMeshPart(
            "neck_shell",
            _elliptical_frustum(
                np.zeros(3), neck_end, (neck_r, neck_r * 0.88), (neck_r * 0.93, neck_r * 0.82), sections=sections
            ),
            MID_GRAY,
        )
    )

    head_points = np.asarray(list(landmarks["Head"].values()))
    head_min, head_max = head_points.min(axis=0), head_points.max(axis=0)
    head_center = (head_min + head_max) * 0.5
    head_span = head_max - head_min
    # Ear-to-ear width and top-to-base height are direct surface dimensions.
    # The sparse anterior/posterior landmarks underestimate depth, so infer
    # only that axis from the calibrated head width rather than imposing a
    # fixed-size human head that would erase between-subject differences.
    head_radii = np.array(
        [max(float(head_span[0]) * 0.5, float(head_span[1]) * 0.52), head_span[1] * 0.5, head_span[2] * 0.5]
    )
    parts["Head"].append(
        AvatarMeshPart("head_shell", _ellipsoid(head_center, tuple(head_radii), subdivisions=2), LIGHT_GRAY)
    )
    head_cap = _ellipsoid(
        head_center + np.array([-head_radii[0] * 0.34, 0.0, head_radii[2] * 0.2]),
        (head_radii[0] * 0.72, head_radii[1] * 1.02, head_radii[2] * 0.82),
        subdivisions=2,
    )
    parts["Head"].append(AvatarMeshPart("head_cap", head_cap, DARK_GRAY, "panel"))

    limb_profiles = {
        "RightShoulder": (0.30, 0.24),
        "LeftShoulder": (0.30, 0.24),
        "RightUpperArm": (0.125, 0.105),
        "LeftUpperArm": (0.125, 0.105),
        "RightForeArm": (0.115, 0.09),
        "LeftForeArm": (0.115, 0.09),
    }
    taper = {
        "Shoulder": (1.0, 0.78),
        "UpperArm": (1.0, 0.76),
        "ForeArm": (1.0, 0.72),
    }
    for segment, radius_factors in limb_profiles.items():
        end = _distal(proportions, segment)
        length = float(np.linalg.norm(end))
        kind = next(key for key in taper if key in segment)
        start_scale, end_scale = taper[kind]
        radii = (max(0.018, length * radius_factors[0]), max(0.014, length * radius_factors[1]))
        start_radii = (radii[0] * start_scale, radii[1] * start_scale)
        end_radii = (radii[0] * end_scale, radii[1] * end_scale)
        inset = min(0.012, length * 0.045)
        shell = _elliptical_frustum(np.zeros(3), end, start_radii, end_radii, sections=sections, inset=inset)
        main_color = LIGHT_GRAY if kind in {"Shoulder", "UpperArm"} else MID_GRAY
        parts[segment].append(AvatarMeshPart(f"{segment.lower()}_shell", shell, main_color))
        parts[segment].append(
            AvatarMeshPart(
                f"{segment.lower()}_joint_collar",
                _collar(end, np.zeros(3), end_radii, sections=sections),
                ACCENT_ORANGE,
                "accent",
            )
        )

    for side in ("Right", "Left"):
        parts[f"{side}Hand"].extend(_build_hand_parts(proportions, side, sections=sections))
        parts[f"{side}UpperLeg"].extend(_build_leg_parts(proportions, side, "Upper", sections=sections))
        parts[f"{side}LowerLeg"].extend(_build_leg_parts(proportions, side, "Lower", sections=sections))

    # Xsens feet face +X in the T-pose. Their recorded heel, metatarsal,
    # instep, and toe landmarks give a much better shape than limb frusta.
    for side in ("Right", "Left"):
        parts[f"{side}Foot"].extend(_build_foot_parts(proportions, side))
        parts[f"{side}Toe"].extend(_build_toe_parts(proportions, side))

    return {name: tuple(segment_parts) for name, segment_parts in parts.items()}


def _ellipse_tube(
    center: np.ndarray,
    radii: tuple[float, float],
    tube_radius: float,
    *,
    sections: int = 48,
) -> trimesh.Trimesh:
    points = [
        np.asarray(center) + np.array([radii[0] * np.cos(theta), radii[1] * np.sin(theta), 0.0])
        for theta in np.linspace(0.0, 2.0 * np.pi, sections, endpoint=False)
    ]
    cylinders = [
        cylinder_between(points[i], points[(i + 1) % sections], tube_radius, sections=8) for i in range(sections)
    ]
    return trimesh.util.concatenate(cylinders)


def build_tennis_racket_meshes() -> tuple[AvatarMeshPart, ...]:
    """Build a regulation-sized racket aligned to the tracked sword frame.

    Xsens rolls the recorded ``RightHandSword`` frame -90 degrees around its
    longitudinal +X axis in the T-pose. The inverse local mesh rotation keeps
    that calibrated frame intact while making the racket's string plane
    horizontal in the resulting world-space T-pose.
    """

    handle = cylinder_between(np.array([-0.09, 0.0, 0.0]), np.array([0.09, 0.0, 0.0]), 0.018, sections=10)
    shaft = cylinder_between(np.array([0.09, 0.0, 0.0]), np.array([0.25, 0.0, 0.0]), 0.009, sections=10)
    throat_left = cylinder_between(np.array([0.16, 0.0, 0.0]), np.array([0.27, 0.075, 0.0]), 0.008, sections=8)
    throat_right = cylinder_between(np.array([0.16, 0.0, 0.0]), np.array([0.27, -0.075, 0.0]), 0.008, sections=8)
    hoop_center = np.array([0.415, 0.0, 0.0])
    hoop_radii = (0.175, 0.135)
    hoop = _ellipse_tube(hoop_center, hoop_radii, 0.009)
    frame = trimesh.util.concatenate([shaft, throat_left, throat_right, hoop])

    strings: list[trimesh.Trimesh] = []
    for x_offset in np.linspace(-0.135, 0.135, 11):
        y_extent = hoop_radii[1] * np.sqrt(max(0.0, 1.0 - (x_offset / hoop_radii[0]) ** 2)) * 0.92
        strings.append(
            cylinder_between(
                hoop_center + np.array([x_offset, -y_extent, 0.0]),
                hoop_center + np.array([x_offset, y_extent, 0.0]),
                0.0011,
                sections=6,
            )
        )
    for y_offset in np.linspace(-0.105, 0.105, 9):
        x_extent = hoop_radii[0] * np.sqrt(max(0.0, 1.0 - (y_offset / hoop_radii[1]) ** 2)) * 0.92
        strings.append(
            cylinder_between(
                hoop_center + np.array([-x_extent, y_offset, 0.0]),
                hoop_center + np.array([x_extent, y_offset, 0.0]),
                0.0011,
                sections=6,
            )
        )
    strings_mesh = trimesh.util.concatenate(strings)
    sword_frame_alignment = trimesh.transformations.rotation_matrix(np.pi / 2.0, [1.0, 0.0, 0.0])
    for mesh in (handle, frame, strings_mesh):
        mesh.apply_transform(sword_frame_alignment)

    return (
        AvatarMeshPart("racket_grip", handle, RACKET_GRIP, "racket"),
        AvatarMeshPart("racket_frame", frame, RACKET_FRAME, "racket"),
        AvatarMeshPart("racket_strings", strings_mesh, RACKET_STRINGS, "racket"),
    )


def validate_avatar_mesh_parts(parts: dict[str, tuple[AvatarMeshPart, ...]]) -> None:
    """Raise a useful error if generated geometry is unsuitable for Viser."""

    for segment_name, segment_parts in parts.items():
        if not segment_parts and segment_name != XSENS_RACKET_SOURCE_SEGMENT:
            raise ValueError(f"No mesh parts generated for Xsens segment {segment_name}")
        for part in segment_parts:
            if part.mesh.vertices.shape[0] == 0 or part.mesh.faces.shape[0] == 0:
                raise ValueError(f"Empty mesh generated for {segment_name}/{part.name}")
            if not np.isfinite(part.mesh.vertices).all():
                raise ValueError(f"Non-finite vertices generated for {segment_name}/{part.name}")
            if part.mesh.faces.ndim != 2 or part.mesh.faces.shape[1] != 3:
                raise ValueError(f"Non-triangular faces generated for {segment_name}/{part.name}")
