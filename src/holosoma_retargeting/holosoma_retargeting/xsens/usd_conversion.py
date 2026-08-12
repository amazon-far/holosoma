"""Application service coordinating reusable XSens, kinematics, geometry, and USD layers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path

from holosoma_retargeting.data_utils.xsens_hdf5 import load_xsens_hdf5_calibration
from holosoma_retargeting.kinematics import validate_kinematic_tree, with_body_attachments
from holosoma_retargeting.usd import create_usd_stage, validate_usd_kinematic_tree, write_kinematic_tree_to_stage
from holosoma_retargeting.xsens.geometry_attachments import (
    build_xsens_avatar_mesh_attachments,
    build_xsens_landmark_attachments,
)
from holosoma_retargeting.xsens.kinematic_model import (
    TENNIS_RACKET_BODY,
    build_xsens_kinematic_tree,
)

XSENS_USD_EXPORTER_VERSION = "1"


@dataclass(frozen=True)
class XsensUsdConversionReport:
    source_path: Path
    output_path: Path
    source_sha256: str
    calibration_fingerprint: str
    body_count: int
    joint_count: int
    includes_tennis_racket: bool
    max_joint_residual_m: float
    warnings: tuple[str, ...] = ()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_xsens_hdf5_to_usd(
    hdf5_path: str | Path,
    output_path: str | Path | None = None,
    *,
    include_visuals: bool = True,
    include_landmarks: bool = True,
    include_tennis_racket: bool = True,
) -> XsensUsdConversionReport:
    """Convert one recording's embedded calibration to one independent USD model."""

    hdf5_path = Path(hdf5_path)
    if output_path is None:
        output_path = hdf5_path.with_name(f"{hdf5_path.stem}_xsens_model.usda")
    output_path = Path(output_path)

    calibration = load_xsens_hdf5_calibration(hdf5_path)
    model = build_xsens_kinematic_tree(calibration, include_tennis_racket=include_tennis_racket)
    source_sha256 = _file_sha256(hdf5_path)
    model = replace(
        model,
        metadata={
            **model.metadata,
            "xsens:sourceFileSha256": source_sha256,
            "model:exporter": "holosoma_retargeting.xsens.usd_conversion",
            "model:exporterVersion": XSENS_USD_EXPORTER_VERSION,
        },
    )
    body_names = {body.name for body in model.bodies}
    point_sets = build_xsens_landmark_attachments(calibration) if include_landmarks else {}
    meshes = build_xsens_avatar_mesh_attachments(calibration) if include_visuals else {}
    point_sets = {name: attachments for name, attachments in point_sets.items() if name in body_names}
    meshes = {name: attachments for name, attachments in meshes.items() if name in body_names}
    model = with_body_attachments(model, point_sets=point_sets, meshes=meshes)

    validation = validate_kinematic_tree(model)
    validation.raise_if_invalid()
    stage = create_usd_stage(output_path)
    write_kinematic_tree_to_stage(stage, model)
    usd_validation = validate_usd_kinematic_tree(stage)
    usd_validation.raise_if_invalid()
    stage.GetRootLayer().Save()

    return XsensUsdConversionReport(
        source_path=hdf5_path,
        output_path=output_path,
        source_sha256=source_sha256,
        calibration_fingerprint=str(model.metadata["xsens:calibrationFingerprint"]),
        body_count=len(model.bodies),
        joint_count=len(model.joints),
        includes_tennis_racket=TENNIS_RACKET_BODY in body_names,
        max_joint_residual_m=max(validation.max_joint_residual_m, usd_validation.max_joint_residual_m),
        warnings=validation.warnings + usd_validation.warnings,
    )
