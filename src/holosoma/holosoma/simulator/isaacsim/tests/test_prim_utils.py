"""Robot-link lookup against real USD, without requiring an Omni application."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

Usd = pytest.importorskip("pxr.Usd")
UsdGeom = pytest.importorskip("pxr.UsdGeom")
UsdPhysics = pytest.importorskip("pxr.UsdPhysics")

pytestmark = pytest.mark.no_sim

ROBOT = "/World/envs/env_0/Robot"
EXPRESSION = "/World/envs/env_.*/Robot"


@pytest.fixture
def prim_utils(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import the production module; only the unused Omni imports need stand-ins."""
    for name in ("omni", "omni.log", "omni.usd"):
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, ModuleType(name))
    name = "holosoma.simulator.isaacsim._prim_utils_cpu_test"
    path = Path(__file__).resolve().parents[1] / "prim_utils.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _stage() -> Any:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, ROBOT)
    return stage


def _body(stage: Any, path: str) -> Any:
    prim = UsdGeom.Xform.Define(stage, path).GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    return prim


@pytest.mark.parametrize("container", ["visuals", "collisions"])
@pytest.mark.parametrize("wrapped", [False, True], ids=["mesh", "named-frame-with-mesh"])
@pytest.mark.parametrize("nested_owner", [False, True], ids=["sibling-owner", "nested-owner"])
def test_other_body_geometry_and_named_frames_preserve_link_identity(
    prim_utils: ModuleType, container: str, wrapped: bool, nested_owner: bool
) -> None:
    stage = _stage()
    target_path = f"{ROBOT}/tool"
    _body(stage, target_path)
    owner_path = f"{target_path}/other" if nested_owner else f"{ROBOT}/other"
    _body(stage, owner_path)
    geometry_path = f"{owner_path}/{container}/tool"
    UsdGeom.Xform.Define(stage, f"{owner_path}/{container}")
    if wrapped:
        UsdGeom.Xform.Define(stage, geometry_path)
        UsdGeom.Mesh.Define(stage, f"{geometry_path}/mesh")
    else:
        UsdGeom.Mesh.Define(stage, geometry_path)

    authored = stage.GetRootLayer().ExportToString()
    for lookup in (
        prim_utils.resolve_robot_link_prim_expression,
        prim_utils.resolve_robot_link_rigid_body_prim_expression,
    ):
        if wrapped:
            # A named Xform under a different rigid owner can be a fixed sensor mount.
            # Its mesh children do not establish that it is merely an exporter wrapper.
            with pytest.raises(ValueError, match="expected one matching prim") as exc:
                lookup("tool", stage)
            assert target_path in str(exc.value) and geometry_path in str(exc.value)
        else:
            assert lookup("tool", stage) == f"{EXPRESSION}/tool"
    assert stage.GetRootLayer().ExportToString() == authored


@pytest.mark.parametrize("frame_name", ["mount", "visuals"])
def test_named_frame_with_geometry_children_remains_a_candidate(prim_utils: ModuleType, frame_name: str) -> None:
    stage = _stage()
    _body(stage, f"{ROBOT}/base")
    frame_path = f"{ROBOT}/base/{frame_name}"
    UsdGeom.Xform.Define(stage, frame_path)
    UsdGeom.Mesh.Define(stage, f"{frame_path}/mesh")
    assert prim_utils.resolve_robot_link_prim_expression(frame_name, stage) == f"{EXPRESSION}/base/{frame_name}"
    assert prim_utils.resolve_robot_link_rigid_body_prim_expression(frame_name, stage) == f"{EXPRESSION}/base"

    # Geometry children do not hide another frame with the same exact name.
    other_path = f"{ROBOT}/base/other/{frame_name}"
    UsdGeom.Xform.Define(stage, other_path)
    with pytest.raises(ValueError, match="expected one matching prim") as exc:
        prim_utils.resolve_robot_link_prim_expression(frame_name, stage)
    assert frame_path in str(exc.value) and other_path in str(exc.value)


@pytest.mark.parametrize("enabled", [True, False], ids=["enabled-body", "disabled-body"])
def test_body_authored_on_geometry_remains_an_ambiguous_candidate(prim_utils: ModuleType, enabled: bool) -> None:
    stage = _stage()
    target_path = f"{ROBOT}/tool"
    _body(stage, target_path)
    _body(stage, f"{ROBOT}/other")
    UsdGeom.Xform.Define(stage, f"{ROBOT}/other/visuals")
    other_path = f"{ROBOT}/other/visuals/tool"
    body = UsdGeom.Cube.Define(stage, other_path).GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body).CreateRigidBodyEnabledAttr(enabled)

    for lookup in (
        prim_utils.resolve_robot_link_prim_expression,
        prim_utils.resolve_robot_link_rigid_body_prim_expression,
    ):
        with pytest.raises(ValueError, match="expected one matching prim") as exc:
            lookup("tool", stage)
        assert target_path in str(exc.value) and other_path in str(exc.value)


def test_policy_synonyms_do_not_resolve_to_asset_links(prim_utils: ModuleType) -> None:
    stage = _stage()
    _body(stage, f"{ROBOT}/native_tool")
    for lookup in (
        prim_utils.resolve_robot_link_prim_expression,
        prim_utils.resolve_robot_link_rigid_body_prim_expression,
    ):
        with pytest.raises(ValueError, match="found 0"):
            lookup("policy_tool", stage)
