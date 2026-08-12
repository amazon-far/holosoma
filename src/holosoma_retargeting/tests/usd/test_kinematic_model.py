"""Tests for USD kinematic model conversion."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pxr")
from holosoma_retargeting.kinematics import (
    KinematicTree,
    MeshAttachment,
    PointSetAttachment,
    RigidBodyDefinition,
    SphericalJointDefinition,
    Transform,
)
from holosoma_retargeting.usd import (
    create_usd_stage,
    read_kinematic_tree_from_stage,
    validate_usd_kinematic_tree,
    write_kinematic_tree_to_stage,
)
from pxr import UsdShade


def _model() -> KinematicTree:
    child = RigidBodyDefinition(
        "TennisRacket",
        Transform(np.array([0.0, 0.0, 0.5]), np.array([1.0, 0.0, 0.0, 0.0])),
        point_sets=(PointSetAttachment("Landmarks", np.array([[0.0, 0.0, 0.0]]), ("origin",)),),
        meshes=(
            MeshAttachment(
                "triangle",
                np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]]),
                np.array([[0, 1, 2]]),
                (10, 20, 30),
                "racket",
            ),
        ),
        metadata={"xsens:sourceSegmentName": "RightHandSword"},
    )
    return KinematicTree(
        "Avatar",
        "RightHand",
        (RigidBodyDefinition("RightHand", Transform.identity()), child),
        (
            SphericalJointDefinition(
                "RightHandTennisRacketOrigin",
                "RightHand",
                "TennisRacket",
                Transform(np.array([0.0, 0.0, 0.5]), np.array([1.0, 0.0, 0.0, 0.0])),
                Transform.identity(),
                metadata={"xsens:sourceJointName": "RightHandSwordOrigin"},
            ),
        ),
        metadata={"xsens:calibrationFingerprint": "abc"},
    )


def test_usd_round_trip_preserves_tree_geometry_and_metadata(tmp_path) -> None:
    path = tmp_path / "model.usda"
    stage = create_usd_stage(path)
    write_kinematic_tree_to_stage(stage, _model())
    stage.GetRootLayer().Save()

    result = read_kinematic_tree_from_stage(stage)
    assert validate_usd_kinematic_tree(stage).is_valid
    assert [body.name for body in result.bodies] == ["RightHand", "TennisRacket"]
    assert result.joints[0].name == "RightHandTennisRacketOrigin"
    assert result.bodies[1].metadata["xsens:sourceSegmentName"] == "RightHandSword"
    assert result.bodies[1].point_sets[0].point_names == ("origin",)
    assert result.bodies[1].meshes[0].category == "racket"
    np.testing.assert_array_equal(result.bodies[1].meshes[0].faces, [[0, 1, 2]])
    mesh_prim = stage.GetPrimAtPath("/XsensAvatar/Bodies/TennisRacket/triangle")
    bound_material, relationship = UsdShade.MaterialBindingAPI(mesh_prim).ComputeBoundMaterial()
    assert relationship
    assert bound_material.GetPath() == "/XsensAvatar/Looks/racket_10_20_30"


def test_usd_replace_is_explicit_and_preserves_unrelated_prims(tmp_path) -> None:
    path = tmp_path / "model.usda"
    stage = create_usd_stage(path)
    stage.DefinePrim("/Unrelated", "Xform")
    write_kinematic_tree_to_stage(stage, _model())

    with pytest.raises(ValueError, match="already exists"):
        write_kinematic_tree_to_stage(stage, _model())
    write_kinematic_tree_to_stage(stage, _model(), replace_existing=True)

    assert stage.GetPrimAtPath("/Unrelated").IsValid()
    assert validate_usd_kinematic_tree(stage).is_valid


def test_usd_does_not_author_an_empty_looks_scope(tmp_path) -> None:
    path = tmp_path / "model_without_meshes.usda"
    model = _model()
    bodies = tuple(
        RigidBodyDefinition(body.name, body.reference_pose, body.point_sets, (), body.metadata)
        for body in model.bodies
    )
    stage = create_usd_stage(path)
    write_kinematic_tree_to_stage(
        stage,
        KinematicTree(model.name, model.root_body, bodies, model.joints, model.metadata),
    )

    assert not stage.GetPrimAtPath("/XsensAvatar/Looks").IsValid()
