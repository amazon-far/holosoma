"""The YAM colour oracle reads bound geometry appearances through real USD APIs."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("pxr")

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

from holosoma.simulator.isaacsim.urdf_visual_colors import apply_urdf_visual_colors
from tests.simulators.yam_trajectory_assert import check_isaacsim_visual_colors

pytestmark = pytest.mark.isaacsim

_GEOMETRY = "/Robot/link/visuals/shape/Mesh"


def _material(stage: Usd.Stage, path: str, color: tuple[float, float, float], *, mdl: bool) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    if mdl:
        # Stock IsaacSim URDF importer: the shader has MDL source metadata and only an MDL surface output.
        shader.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
        shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    else:
        shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuse_color_constant" if mdl else "diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*color)
    )
    output = shader.CreateOutput("out" if mdl else "surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput("mdl" if mdl else "").ConnectToSource(output)
    return material


def _fixture(tmp_path: Path, *, mdl: bool = True) -> tuple[SimpleNamespace, Path, Usd.Stage, UsdShade.Material]:
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(
        '<robot name="robot"><link name="link"><visual><geometry><box size="1 1 1"/></geometry>'
        '<material name="red"><color rgba="1 0 0 1"/></material></visual></link></robot>'
    )
    usd = tmp_path / "robot.usda"
    stage = Usd.Stage.CreateNew(str(usd))
    stage.SetDefaultPrim(stage.DefinePrim("/Robot", "Xform"))
    mesh = UsdGeom.Mesh.Define(stage, _GEOMETRY)
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3, 3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 1, 3, 2])
    material = _material(stage, "/Robot/link/visuals/shape/Original", (1, 1, 1), mdl=mdl)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()
    config = SimpleNamespace(
        robot=SimpleNamespace(asset=SimpleNamespace(urdf_file=str(urdf), asset_root=str(tmp_path)))
    )
    return config, usd, stage, material


def _scene(usd: Path) -> Usd.Stage:
    stage = Usd.Stage.CreateInMemory()
    root = stage.DefinePrim("/World/envs/env_0/Robot", "Xform")
    root.GetReferences().AddReference(str(usd))
    root.SetInstanceable(True)
    assert not stage.GetCompositionErrors()
    return stage


@pytest.mark.parametrize("mdl", [False, True], ids=["preview-surface", "omnipbr-mdl"])
def test_checker_accepts_the_writers_bound_colour_after_scene_reference(tmp_path: Path, *, mdl: bool) -> None:
    config, usd, source, original = _fixture(tmp_path, mdl=mdl)
    assert apply_urdf_visual_colors(config.robot.asset.urdf_file, str(usd)) == 1
    scene = _scene(usd)
    geometry = scene.GetPrimAtPath("/World/envs/env_0" + _GEOMETRY)
    assert geometry.IsInstanceProxy()
    # Independent public-USD oracle: no writer paths or graph-discovery helpers.
    bound, _ = UsdShade.MaterialBindingAPI(geometry).ComputeBoundMaterial()
    shader, _, _ = bound.ComputeSurfaceSource("mdl" if mdl else "")
    color_name = "diffuse_color_constant" if mdl else "diffuseColor"
    assert tuple(shader.GetInput(color_name).Get()) == (1, 0, 0)
    original_shader, _, _ = original.ComputeSurfaceSource("mdl" if mdl else "")
    assert tuple(original_shader.GetInput(color_name).Get()) == (1, 1, 1)
    if mdl:
        assert not bound.ComputeSurfaceSource()[0]
    assert check_isaacsim_visual_colors(config, scene) == []


@pytest.mark.parametrize("purpose", ["", "preview", "full"], ids=["all-purpose", "preview", "full"])
def test_checker_rejects_wrong_bound_colour_even_if_unused_shader_is_correct(tmp_path: Path, purpose: str) -> None:
    config, usd, stage, original = _fixture(tmp_path)
    original_shader, _, _ = original.ComputeSurfaceSource("mdl")
    original_shader.GetInput("diffuse_color_constant").Set(Gf.Vec3f(1, 0, 0))
    wrong = _material(stage, "/Robot/Looks/Wrong", (0, 0, 1), mdl=True)
    UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(_GEOMETRY)).Bind(wrong, materialPurpose=purpose)
    stage.GetRootLayer().Save()
    failures = check_isaacsim_visual_colors(config, _scene(usd))
    assert len(failures) == 1
    assert "Wrong" in failures[0] and "(0.0, 0.0, 1.0)" in failures[0]
    assert (purpose or "allPurpose") in failures[0]


def test_checker_reads_connected_colour_instead_of_its_fallback_value(tmp_path: Path) -> None:
    config, usd, stage, material = _fixture(tmp_path)
    shader, _, _ = material.ComputeSurfaceSource("mdl")
    color = material.CreateInput("paint", Sdf.ValueTypeNames.Color3f)
    color.Set(Gf.Vec3f(1, 0, 0))
    shader.GetInput("diffuse_color_constant").ConnectToSource(color)
    stage.GetRootLayer().Save()
    scene = _scene(usd)
    assert check_isaacsim_visual_colors(config, scene) == []
    color.Set(Gf.Vec3f(0, 0, 1))
    failures = check_isaacsim_visual_colors(config, scene)
    assert len(failures) == 1 and "(0.0, 0.0, 1.0)" in failures[0]


@pytest.mark.parametrize("complete", [False, True], ids=["partial-subset", "complete-partition"])
def test_checker_checks_material_subsets_and_only_uncovered_parent_faces(tmp_path: Path, *, complete: bool) -> None:
    config, usd, stage, _ = _fixture(tmp_path)
    api = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(_GEOMETRY))
    subset = api.CreateMaterialBindSubset("red_faces", [0, 1] if complete else [0])
    material = _material(stage, "/Robot/Looks/Subset", (1, 0, 0), mdl=True)
    UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()
    scene = _scene(usd)
    failures = check_isaacsim_visual_colors(config, scene)
    if complete:
        assert failures == []  # The white parent binding is unused.
    else:
        assert len(failures) == 1 and "Original" in failures[0]  # Face 1 still uses white.
    shader, _, _ = material.ComputeSurfaceSource("mdl")
    shader.GetInput("diffuse_color_constant").Set(Gf.Vec3f(0, 0, 1))
    failures = check_isaacsim_visual_colors(config, scene)
    assert any("/Mesh/red_faces" in message and "(0.0, 0.0, 1.0)" in message for message in failures)


def test_checker_does_not_accept_an_unbound_shader_definition(tmp_path: Path) -> None:
    config, usd, stage, material = _fixture(tmp_path)
    shader, _, _ = material.ComputeSurfaceSource("mdl")
    shader.GetInput("diffuse_color_constant").Set(Gf.Vec3f(1, 0, 0))
    UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(_GEOMETRY)).UnbindAllBindings()
    stage.GetRootLayer().Save()
    failures = check_isaacsim_visual_colors(config, _scene(usd))
    assert len(failures) == 1 and "no bound material" in failures[0]


def test_checker_reports_unverifiable_texture_instead_of_accepting_fallback(tmp_path: Path) -> None:
    config, usd, stage, material = _fixture(tmp_path, mdl=False)
    shader, _, _ = material.ComputeSurfaceSource()
    shader.GetInput("diffuseColor").Set(Gf.Vec3f(1, 0, 0))
    texture = UsdShade.Shader.Define(stage, "/Robot/Looks/Texture")
    texture.CreateIdAttr("UsdUVTexture")
    shader.GetInput("diffuseColor").ConnectToSource(texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
    stage.GetRootLayer().Save()
    failures = check_isaacsim_visual_colors(config, _scene(usd))
    assert len(failures) == 1 and "cannot resolve constant RGB" in failures[0]
