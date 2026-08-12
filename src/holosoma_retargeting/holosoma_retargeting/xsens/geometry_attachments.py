"""Adapters from XSens avatar geometry to backend-independent attachments."""

from __future__ import annotations

import numpy as np

from holosoma_retargeting.data_utils.xsens_hdf5 import XsensHdf5Calibration
from holosoma_retargeting.kinematics import MeshAttachment, PointSetAttachment
from holosoma_retargeting.xsens.avatar_mesh import (
    AvatarMeshPart,
    avatar_proportions_from_calibration,
    build_tennis_racket_meshes,
    build_xsens_avatar_meshes,
)
from holosoma_retargeting.xsens.kinematic_model import (
    TENNIS_RACKET_BODY,
    XSENS_RACKET_SOURCE_SEGMENT,
    canonical_xsens_segment_name,
)


def _mesh_attachment(part: AvatarMeshPart) -> MeshAttachment:
    return MeshAttachment(
        name=part.name,
        vertices_m=np.asarray(part.mesh.vertices, dtype=float),
        faces=np.asarray(part.mesh.faces, dtype=np.int64),
        color_rgb=part.color,
        category=part.category,
    )


def build_xsens_avatar_mesh_attachments(
    calibration: XsensHdf5Calibration,
) -> dict[str, tuple[MeshAttachment, ...]]:
    """Convert procedural rigid meshes into canonical local attachments."""

    parts_by_source = build_xsens_avatar_meshes(avatar_proportions_from_calibration(calibration))
    attachments = {
        canonical_xsens_segment_name(source_name): tuple(_mesh_attachment(part) for part in parts)
        for source_name, parts in parts_by_source.items()
        if source_name != XSENS_RACKET_SOURCE_SEGMENT
    }
    if XSENS_RACKET_SOURCE_SEGMENT in calibration.segment_names:
        attachments[TENNIS_RACKET_BODY] = tuple(_mesh_attachment(part) for part in build_tennis_racket_meshes())
    return attachments


def build_xsens_landmark_attachments(
    calibration: XsensHdf5Calibration,
) -> dict[str, tuple[PointSetAttachment, ...]]:
    """Convert calibrated local landmarks into canonical point sets."""

    result: dict[str, tuple[PointSetAttachment, ...]] = {}
    for source_name in calibration.segment_names:
        landmark_map = calibration.landmarks_m[source_name]
        names = tuple(landmark_map)
        points = np.asarray([landmark_map[name] for name in names], dtype=float).reshape(-1, 3)
        result[canonical_xsens_segment_name(source_name)] = (
            PointSetAttachment(
                name="AnatomicalLandmarks",
                points_m=points,
                point_names=names,
                metadata={"xsens:sourceSegmentName": source_name},
            ),
        )
    return result
