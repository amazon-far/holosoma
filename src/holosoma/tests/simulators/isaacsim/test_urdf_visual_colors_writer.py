"""URDF colours stay local across geometry instances and shared material bindings."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from loguru import logger

pytestmark = pytest.mark.isaacsim

pytest.importorskip("pxr")

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade  # noqa: E402

from holosoma.simulator.isaacsim.urdf_visual_colors import apply_urdf_visual_colors  # noqa: E402


@pytest.fixture
def colour_warnings() -> Iterator[list[str]]:
    messages: list[str] = []
    sink = logger.add(messages.append, format="{message}", level="WARNING")
    try:
        yield messages
    finally:
        logger.remove(sink)


def _shader(stage: Usd.Stage, path: str) -> UsdShade.Shader:
    shader = UsdShade.Shader.Define(stage, path)
    shader.CreateIdAttr("OmniPBR")
    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 1.0, 1.0))
    return shader


def test_mixed_link_visuals_are_patched_individually(tmp_path: Path) -> None:
    urdf = tmp_path / "mixed.urdf"
    urdf.write_text(
        """<robot name="mixed">
          <link name="mixed">
            <visual><material name="grey"><color rgba="0.5 0.5 0.5 1"/></material></visual>
            <visual><material name="red"><color rgba="1 0 0 1"/></material></visual>
          </link>
        </robot>"""
    )
    usd = tmp_path / "mixed.usda"
    stage = Usd.Stage.CreateNew(str(usd))
    paths = {
        "body": "/World/Robot/mixed/visuals/body",
        "marker": "/World/Robot/mixed/visuals/marker",
        "collision": "/World/Robot/mixed/collisions/collider",
    }
    for path in paths.values():
        mesh = _mesh(stage, path + "/Geometry")
        material = UsdShade.Material.Define(stage, path + "/Material")
        shader = _shader(stage, path + "/Material/Shader")
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()

    assert apply_urdf_visual_colors(str(urdf), str(usd)) == 2

    stage = Usd.Stage.Open(str(usd))

    def color(name: str) -> tuple[float, float, float]:
        material, _ = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(paths[name] + "/Geometry")).ComputeBoundMaterial()
        shader, _, _ = material.ComputeSurfaceSource()
        value = shader.GetInput("diffuse_color_constant").Get()
        return float(value[0]), float(value[1]), float(value[2])

    assert color("body") == pytest.approx((0.5, 0.5, 0.5))
    assert color("marker") == pytest.approx((1.0, 0.0, 0.0))
    assert color("collision") == pytest.approx((1.0, 1.0, 1.0))


def test_unbound_shader_definitions_are_left_unchanged(tmp_path: Path, colour_warnings: list[str]) -> None:
    urdf = tmp_path / "unbound.urdf"
    urdf.write_text(
        '<robot><link name="source"><visual><material><color rgba="1 0 0 1"/></material></visual></link></robot>'
    )
    usd = tmp_path / "unbound.usda"
    stage = Usd.Stage.CreateNew(str(usd))
    source = "/Robot/source/visuals/shape"
    _shader(stage, source + "/Shader")
    for destination in ("/Robot/plain/visuals/shape", "/Robot/source/collisions/shape"):
        stage.DefinePrim(destination).GetReferences().AddInternalReference(source)
    stage.GetRootLayer().Save()
    saved = usd.read_bytes()

    assert apply_urdf_visual_colors(str(urdf), str(usd)) == 0
    assert usd.read_bytes() == saved
    assert not stage.GetRootLayer().dirty
    assert source in "\n".join(colour_warnings)
    assert "unbound shaders" in "\n".join(colour_warnings)


def _mesh(stage: Usd.Stage, path: str) -> UsdGeom.Mesh:
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([(-1, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayOpacityAttr([0.75])
    return mesh


def _material(stage: Usd.Stage, path: str, *, sibling_shader: bool = False) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, path)
    shader_path = path + "_Shader" if sibling_shader else path + "/Shader"
    shader = UsdShade.Shader.Define(stage, shader_path)
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1, 1, 1))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.35)
    texture = UsdShade.Shader.Define(stage, path + "/Texture")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("./roughness.png"))
    texture.CreateOutput("r", Sdf.ValueTypeNames.Float)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(texture.ConnectableAPI(), "r")
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _reference(stage: Usd.Stage, path: str, asset: Path, target: str) -> Usd.Prim:
    prim = stage.DefinePrim(path, "Xform")
    prim.GetReferences().AddReference(str(asset), target)
    prim.SetInstanceable(True)
    return prim


def _bound_state(stage: Usd.Stage, mesh_path: str, purpose: str = "") -> dict:
    material, relationship = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(mesh_path)).ComputeBoundMaterial(purpose)
    shader, _, _ = material.ComputeSurfaceSource()
    texture_output = shader.GetInput("roughness").GetAttr().GetConnections()[0]
    texture = UsdShade.Shader(stage.GetPrimAtPath(texture_output.GetPrimPath()))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(mesh_path))
    return {
        "material": str(material.GetPath()),
        "binding": str(relationship.GetPath()),
        "strength": UsdShade.MaterialBindingAPI.GetMaterialBindingStrength(relationship),
        "color": tuple(shader.GetInput("diffuseColor").Get()),
        "metallic": shader.GetInput("metallic").Get(),
        "texture": texture.GetInput("file").Get(),
        "points": mesh.GetPointsAttr().Get(),
        "faces": mesh.GetFaceVertexIndicesAttr().Get(),
        "double_sided": mesh.GetDoubleSidedAttr().Get(),
        "opacity": mesh.GetDisplayOpacityAttr().Get(),
    }


def _shared_fixture(
    tmp_path: Path, *, nested: bool, external: bool, sibling_shader: bool = False
) -> tuple[Path, Path, dict[str, str], dict[Path, bytes]]:
    (tmp_path / "roughness.png").write_bytes(b"asset-path anchoring fixture")
    leaf_path = tmp_path / "mesh.usda"
    leaf = Usd.Stage.CreateNew(str(leaf_path))
    leaf.DefinePrim("/Mesh", "Xform")
    geometry = _mesh(leaf, "/Mesh/Geometry")
    if not external:
        material = _material(leaf, "/Mesh/Looks/Material", sibling_shader=sibling_shader)
        UsdShade.MaterialBindingAPI.Apply(geometry.GetPrim()).Bind(material)
    leaf.GetRootLayer().Save()
    source, target = leaf_path, "/Mesh"
    if nested:
        source = tmp_path / "assembly.usda"
        assembly = Usd.Stage.CreateNew(str(source))
        assembly.DefinePrim("/Assembly", "Xform")
        _reference(assembly, "/Assembly/inner", leaf_path, "/Mesh")
        # This sibling has no coloured shader; its instanceability does not need to change.
        _reference(assembly, "/Assembly/spare", leaf_path, "/Mesh/Geometry")
        assembly.GetRootLayer().Save()
        target = "/Assembly"

    usd_path = tmp_path / "robot.usda"
    stage = Usd.Stage.CreateNew(str(usd_path))
    paths = {name: f"/World/Robot/mixed/visuals/{name}" for name in ("red", "blue", "uncolored")}
    paths["collision"] = "/World/Robot/mixed/collisions/mesh"
    external_material = None
    if external:
        library_path = tmp_path / "materials.usda"
        library = Usd.Stage.CreateNew(str(library_path))
        library.DefinePrim("/Looks", "Scope")
        _material(library, "/Looks/Material", sibling_shader=sibling_shader)
        library.GetRootLayer().Save()
        stage.DefinePrim("/World/Looks", "Scope").GetReferences().AddReference(str(library_path), "/Looks")
        external_material = UsdShade.Material(stage.GetPrimAtPath("/World/Looks/Material"))
    for path in paths.values():
        prim = _reference(stage, path, source, target)
        if external_material:
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                external_material, bindingStrength=UsdShade.Tokens.strongerThanDescendants
            )
    stage.GetRootLayer().Save()
    urdf_path = tmp_path / "robot.urdf"
    urdf_path.write_text(
        """<robot name="mixed"><link name="mixed">
          <visual><material name="red"><color rgba="1 0 0 1"/></material></visual>
          <visual><material name="blue"><color rgba="0 0 1 1"/></material></visual>
          <visual/>
        </link></robot>"""
    )
    sources = {p: p.read_bytes() for p in tmp_path.glob("*.usda") if p != usd_path}
    return urdf_path, usd_path, paths, sources


@pytest.mark.parametrize("nested", [False, True], ids=["flat", "nested"])
@pytest.mark.parametrize("external", [False, True], ids=["local-material", "shared-material"])
@pytest.mark.parametrize("sibling_shader", [False, True], ids=["child-shader", "sibling-shader"])
def test_colours_are_local_and_idempotent(
    tmp_path: Path, *, nested: bool, external: bool, sibling_shader: bool
) -> None:
    urdf, usd, paths, sources = _shared_fixture(
        tmp_path, nested=nested, external=external, sibling_shader=sibling_shader
    )
    suffix = "/inner/Geometry" if nested else "/Geometry"
    stage = Usd.Stage.Open(str(usd))
    before = {name: _bound_state(stage, path + suffix) for name, path in paths.items()}
    original_bindings = {
        state["binding"]: stage.GetRelationshipAtPath(state["binding"]).GetTargets() for state in before.values()
    }

    assert apply_urdf_visual_colors(str(urdf), str(usd)) == 2
    stage = Usd.Stage.Open(str(usd))
    after = {name: _bound_state(stage, path + suffix) for name, path in paths.items()}
    for name, path in paths.items():
        if name in ("red", "blue"):
            assert after[name]["color"] == ((1, 0, 0) if name == "red" else (0, 0, 1))
            assert after[name]["material"] != before[name]["material"]
            assert after[name]["strength"] == UsdShade.Tokens.strongerThanDescendants
        else:
            assert after[name] == before[name]
        assert stage.GetPrimAtPath(path).IsInstance()
        if nested:
            assert stage.GetPrimAtPath(path + "/inner").IsInstance()
            assert stage.GetPrimAtPath(path + "/spare").IsInstance()
        for key in ("metallic", "texture", "points", "faces", "double_sided", "opacity"):
            assert after[name][key] == before[name][key]
    assert {path: stage.GetRelationshipAtPath(path).GetTargets() for path in original_bindings} == original_bindings
    assert after["red"]["material"] != after["blue"]["material"]
    assert {p: p.read_bytes() for p in sources} == sources
    assert all(not Sdf.Layer.Find(str(p)).dirty for p in sources)
    saved = usd.read_bytes()
    prim_paths = [str(prim.GetPath()) for prim in stage.Traverse(Usd.TraverseInstanceProxies())]

    assert apply_urdf_visual_colors(str(urdf), str(usd)) == 2
    assert usd.read_bytes() == saved
    assert [str(prim.GetPath()) for prim in stage.Traverse(Usd.TraverseInstanceProxies())] == prim_paths
    assert {name: _bound_state(stage, path + suffix) for name, path in paths.items()} == after
    assert {p: p.read_bytes() for p in sources} == sources
    assert all(not Sdf.Layer.Find(str(p)).dirty for p in sources)


def test_material_inside_one_visual_shared_by_other_visuals_is_isolated(tmp_path: Path) -> None:
    urdf, usd, paths, sources = _shared_fixture(tmp_path, nested=False, external=False)
    stage = Usd.Stage.Open(str(usd))
    original_material = paths["red"] + "/Looks/Material"
    material = UsdShade.Material(stage.GetPrimAtPath(original_material))
    for name in ("blue", "uncolored", "collision"):
        UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(paths[name])).Bind(
            material, bindingStrength=UsdShade.Tokens.strongerThanDescendants
        )
    stage.GetRootLayer().Save()
    assert apply_urdf_visual_colors(str(urdf), str(usd)) == 2
    result = {name: _bound_state(stage, path + "/Geometry") for name, path in paths.items()}
    assert result["red"]["color"] == (1, 0, 0)
    assert result["blue"]["color"] == (0, 0, 1)
    for name in ("uncolored", "collision"):
        assert result[name]["material"] == original_material
        assert result[name]["color"] == (1, 1, 1)
        assert stage.GetPrimAtPath(paths[name]).IsInstance()
    assert len({values["material"] for values in result.values()}) == 3
    saved = usd.read_bytes()
    assert apply_urdf_visual_colors(str(urdf), str(usd)) == 2
    assert usd.read_bytes() == saved
    assert {p: p.read_bytes() for p in sources} == sources


def test_strong_binding_above_coloured_branches_is_skipped_without_changes(
    tmp_path: Path, colour_warnings: list[str]
) -> None:
    urdf, usd, paths, sources = _shared_fixture(tmp_path, nested=False, external=True)
    stage = Usd.Stage.Open(str(usd))
    material = UsdShade.Material(stage.GetPrimAtPath("/World/Looks/Material"))
    for path in paths.values():
        UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(path)).UnbindAllBindings()
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath("/World/Robot")).Bind(
        material, bindingStrength=UsdShade.Tokens.strongerThanDescendants
    )
    stage.GetRootLayer().Save()
    saved = usd.read_bytes()
    assert apply_urdf_visual_colors(str(urdf), str(usd)) == 0
    assert any("strong ancestor material binding" in message for message in colour_warnings)
    assert usd.read_bytes() == saved
    assert not stage.GetRootLayer().dirty
    assert {p: p.read_bytes() for p in sources} == sources
    assert all(stage.GetPrimAtPath(path).IsInstance() for path in paths.values())


def _root_branch(stage: Usd.Stage, path: str) -> str:
    snapshot = Sdf.Layer.CreateAnonymous()
    Sdf.CopySpec(stage.GetRootLayer(), path, snapshot, "/Branch")
    return snapshot.ExportToString()


def test_skipped_branch_stays_unchanged_while_supported_branches_paint(
    tmp_path: Path, colour_warnings: list[str]
) -> None:
    urdf, usd, paths, sources = _shared_fixture(tmp_path, nested=False, external=True)
    stage = Usd.Stage.Open(str(usd))
    blocked_root = "/World/Robot/blocked"
    blocked_visual = blocked_root + "/visuals/mesh"
    _reference(stage, blocked_visual, tmp_path / "mesh.usda", "/Mesh")
    material = UsdShade.Material(stage.GetPrimAtPath("/World/Looks/Material"))
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(blocked_root)).Bind(
        material, bindingStrength=UsdShade.Tokens.strongerThanDescendants
    )
    stage.GetRootLayer().Save()
    urdf.write_text(
        urdf.read_text().replace(
            "</robot>",
            '<link name="blocked"><visual><material name="green"><color rgba="0 1 0 1"/></material>'
            "</visual></link></robot>",
        )
    )
    before_branch = _root_branch(stage, blocked_root)
    before_state = _bound_state(stage, blocked_visual + "/Geometry")

    assert apply_urdf_visual_colors(str(urdf), str(usd)) == 2
    warning = "\n".join(colour_warnings)
    assert blocked_visual in warning and "strong ancestor material binding" in warning
    assert _root_branch(stage, blocked_root) == before_branch
    assert _bound_state(stage, blocked_visual + "/Geometry") == before_state
    assert stage.GetPrimAtPath(blocked_visual).IsInstance()
    for name, path in paths.items():
        assert _bound_state(stage, path + "/Geometry")["color"] == {"red": (1, 0, 0), "blue": (0, 0, 1)}.get(
            name, (1, 1, 1)
        )
    assert {p: p.read_bytes() for p in sources} == sources
    saved = usd.read_bytes()
    assert apply_urdf_visual_colors(str(urdf), str(usd)) == 2
    assert usd.read_bytes() == saved


@pytest.mark.parametrize("selected", [False, True], ids=["unrelated-material", "selected-material"])
def test_unresolved_material_connections_do_not_interrupt_other_colours(
    tmp_path: Path, colour_warnings: list[str], *, selected: bool
) -> None:
    urdf, usd, paths, sources = _shared_fixture(tmp_path, nested=False, external=True)
    stage = Usd.Stage.Open(str(usd))
    bad_path = "/World/Looks/Incomplete"
    material = _material(stage, bad_path)
    shader, _, _ = material.ComputeSurfaceSource()
    shader.GetInput("roughness").GetAttr().SetConnections([Sdf.Path("/MissingTexture.outputs:r")])
    if selected:
        UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(paths["red"])).Bind(material)
    stage.GetRootLayer().Save()
    before_material = _root_branch(stage, bad_path)
    before_red = _root_branch(stage, paths["red"])

    assert apply_urdf_visual_colors(str(urdf), str(usd)) == (1 if selected else 2)
    warning = "\n".join(colour_warnings)
    if selected:
        assert "unresolved connections" in warning and paths["red"] in warning
        assert _root_branch(stage, paths["red"]) == before_red
        assert stage.GetPrimAtPath(paths["red"]).IsInstance()
    else:
        assert _bound_state(stage, paths["red"] + "/Geometry")["color"] == (1, 0, 0)
    assert _root_branch(stage, bad_path) == before_material
    assert not stage.GetPrimAtPath("/MissingTexture")
    assert _bound_state(stage, paths["blue"] + "/Geometry")["color"] == (0, 0, 1)
    for name in ("uncolored", "collision"):
        assert _bound_state(stage, paths[name] + "/Geometry")["color"] == (1, 1, 1)
        assert stage.GetPrimAtPath(paths[name]).IsInstance()
    assert {p: p.read_bytes() for p in sources} == sources


def test_weak_ancestor_binding_is_isolated_without_deinstancing_inner_geometry(tmp_path: Path) -> None:
    urdf, usd, paths, sources = _shared_fixture(tmp_path, nested=True, external=True)
    stage = Usd.Stage.Open(str(usd))
    material = UsdShade.Material(stage.GetPrimAtPath("/World/Looks/Material"))
    for path in paths.values():
        stage.GetPrimAtPath(path).RemoveProperty("material:binding")
    parent_binding = UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath("/World/Robot"))
    parent_binding.Bind(material)
    stage.GetRootLayer().Save()

    assert apply_urdf_visual_colors(str(urdf), str(usd)) == 2
    assert parent_binding.GetDirectBindingRel().GetTargets() == [material.GetPath()]
    for name, path in paths.items():
        state = _bound_state(stage, path + "/inner/Geometry")
        assert state["color"] == {"red": (1, 0, 0), "blue": (0, 0, 1)}.get(name, (1, 1, 1))
        assert stage.GetPrimAtPath(path + "/inner").IsInstance()
    assert {p: p.read_bytes() for p in sources} == sources
    saved = usd.read_bytes()
    assert apply_urdf_visual_colors(str(urdf), str(usd)) == 2
    assert usd.read_bytes() == saved


def test_preview_binding_and_connected_colour_are_isolated(tmp_path: Path) -> None:
    urdf, usd, paths, sources = _shared_fixture(tmp_path, nested=False, external=True)
    stage = Usd.Stage.Open(str(usd))
    preview_material = _material(stage, "/World/Looks/Preview")
    preview_shader, _, _ = preview_material.ComputeSurfaceSource()
    texture = UsdShade.Shader(stage.GetPrimAtPath("/World/Looks/Preview/Texture"))
    texture.CreateOutput("rgb", Sdf.ValueTypeNames.Color3f)
    preview_shader.GetInput("diffuseColor").ConnectToSource(texture.ConnectableAPI(), "rgb")
    original_connections = preview_shader.GetInput("diffuseColor").GetAttr().GetConnections()
    for path in paths.values():
        api = UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(path))
        api.Bind(preview_material, materialPurpose=UsdShade.Tokens.preview)
        api.GetDirectBindingRel(UsdShade.Tokens.preview).SetCustomDataByKey("fixture", "preserved")
    stage.GetRootLayer().Save()

    assert apply_urdf_visual_colors(str(urdf), str(usd)) == 4
    for name, path in paths.items():
        expected = {"red": (1, 0, 0), "blue": (0, 0, 1)}.get(name, (1, 1, 1))
        assert _bound_state(stage, path + "/Geometry")["color"] == expected
        state = _bound_state(stage, path + "/Geometry", UsdShade.Tokens.preview)
        assert state["color"] == expected
        material = UsdShade.Material(stage.GetPrimAtPath(state["material"]))
        shader, _, _ = material.ComputeSurfaceSource()
        connections = shader.GetInput("diffuseColor").GetAttr().GetConnections()
        assert connections == ([] if name in ("red", "blue") else original_connections)
        assert stage.GetRelationshipAtPath(state["binding"]).GetCustomDataByKey("fixture") == "preserved"
    assert preview_shader.GetInput("diffuseColor").GetAttr().GetConnections() == original_connections
    saved = usd.read_bytes()
    assert apply_urdf_visual_colors(str(urdf), str(usd)) == 4
    assert usd.read_bytes() == saved
    assert {p: p.read_bytes() for p in sources} == sources


def test_material_without_supported_colour_input_causes_no_edits(tmp_path: Path) -> None:
    urdf, usd, paths, sources = _shared_fixture(tmp_path, nested=False, external=True)
    stage = Usd.Stage.Open(str(usd))
    # Replace the library's colour-bearing material at its root-layer reference site.
    replacement = UsdShade.Material.Define(stage, "/World/UncolouredMaterial")
    shader = UsdShade.Shader.Define(stage, "/World/UncolouredMaterial/Shader")
    shader.CreateIdAttr("CustomSurface")
    replacement.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    for path in paths.values():
        UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(path)).Bind(replacement)
    stage.GetRootLayer().Save()
    saved = usd.read_bytes()

    assert apply_urdf_visual_colors(str(urdf), str(usd)) == 0
    assert usd.read_bytes() == saved
    assert not stage.GetRootLayer().dirty
    assert all(stage.GetPrimAtPath(path).IsInstance() for path in paths.values())
    assert {p: p.read_bytes() for p in sources} == sources


@pytest.mark.parametrize("arc", ["material_reference", "material_inherit", "visual_reference"])
@pytest.mark.parametrize("reverse", [False, True], ids=["source-first", "consumer-first"])
def test_internal_composition_consumers_keep_independent_colours(tmp_path: Path, arc: str, *, reverse: bool) -> None:
    urdf = tmp_path / "composition.urdf"
    usd = tmp_path / "composition.usda"
    stage = Usd.Stage.CreateNew(str(usd))
    names = ("source", "derived", "plain")
    for name in reversed(names) if reverse else names:
        stage.DefinePrim("/Robot/" + name, "Xform")
    paths = {name: f"/Robot/{name}/visuals/shape" for name in names}
    paths["collision"] = "/Robot/source/collisions/shape"
    _mesh(stage, paths["source"] + "/Geometry")
    material = _material(stage, paths["source"] + "/Material")
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(paths["source"])).Bind(material)
    for name, source in (("derived", "source"), ("plain", "derived"), ("collision", "source")):
        prim = stage.DefinePrim(paths[name], "Xform")
        if arc == "visual_reference":
            prim.GetReferences().AddInternalReference(paths[source])
            prim.SetInstanceable(True)
        else:
            mesh = _mesh(stage, paths[name] + "/Geometry")
            target = paths[source] + "/Material"
            material = UsdShade.Material.Define(stage, paths[name] + "/Material")
            if arc == "material_reference":
                material.GetPrim().GetReferences().AddInternalReference(target)
            else:
                material.GetPrim().GetInherits().AddInherit(target)
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()
    before = {name: _bound_state(stage, path + "/Geometry") for name, path in paths.items()}

    for palette in ({"source": (1, 0, 0), "derived": (0, 0, 1)}, {"source": (0, 1, 0), "derived": (1, 1, 0)}):
        urdf.write_text(
            "<robot>"
            + "".join(
                f'<link name="{name}"><visual><material><color rgba="{r} {g} {b} 1"/></material></visual></link>'
                for name, (r, g, b) in palette.items()
            )
            + "</robot>"
        )
        assert apply_urdf_visual_colors(str(urdf), str(usd)) == 2
        for name, path in paths.items():
            state = _bound_state(stage, path + "/Geometry")
            if name in palette:
                assert state["color"] == palette[name]
                for key in ("metallic", "texture", "points", "faces", "double_sided", "opacity"):
                    assert state[key] == before[name][key]
            else:
                assert state == before[name]
            if arc == "visual_reference" and name != "source":
                assert stage.GetPrimAtPath(path).IsInstance()
        assert not stage.GetCompositionErrors()
        saved = usd.read_bytes()
        assert apply_urdf_visual_colors(str(urdf), str(usd)) == 2
        assert usd.read_bytes() == saved


def test_existing_root_collections_keep_distinct_material_details(tmp_path: Path) -> None:
    urdf = tmp_path / "collections.urdf"
    urdf.write_text(
        '<robot><link name="source"><visual><material><color rgba="1 0 0 1"/></material></visual></link></robot>'
    )
    usd = tmp_path / "collections.usda"
    stage = Usd.Stage.CreateNew(str(usd))
    root = stage.DefinePrim("/Robot", "Xform")
    api = UsdShade.MaterialBindingAPI.Apply(root)
    for name, metallic in (("A", 0.25), ("B", 0.75)):
        material = _material(stage, "/Robot/Looks/" + name)
        shader, _, _ = material.ComputeSurfaceSource()
        shader.GetInput("metallic").Set(metallic)
        meshes = [f"/Robot/{group}/visuals/shape/{name}" for group in ("source", "plain")]
        for path in meshes:
            _mesh(stage, path)
        collection = Usd.CollectionAPI.Apply(root, name)
        collection.CreateIncludesRel().SetTargets(meshes)
        api.Bind(collection, material, bindingName=name)
    stage.GetRootLayer().Save()
    original_targets = {rel.GetName(): rel.GetTargets() for rel in root.GetRelationships()}

    assert apply_urdf_visual_colors(str(urdf), str(usd)) == 2
    for name, metallic in (("A", 0.25), ("B", 0.75)):
        for group in ("source", "plain"):
            state = _bound_state(stage, f"/Robot/{group}/visuals/shape/{name}")
            assert state["color"] == ((1, 0, 0) if group == "source" else (1, 1, 1))
            assert state["metallic"] == metallic
    assert {name: root.GetRelationship(name).GetTargets() for name in original_targets} == original_targets
    saved = usd.read_bytes()
    assert apply_urdf_visual_colors(str(urdf), str(usd)) == 2
    assert usd.read_bytes() == saved


def test_colour_bindings_survive_referencing_a_nested_default_prim(tmp_path: Path) -> None:
    urdf, usd, paths, _ = _shared_fixture(tmp_path, nested=True, external=False)
    stage = Usd.Stage.Open(str(usd))
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World/Robot"))
    stage.GetRootLayer().Save()
    assert apply_urdf_visual_colors(str(urdf), str(usd)) == 2

    scene = Usd.Stage.CreateNew(str(tmp_path / "scene.usda"))
    robot = scene.DefinePrim("/Scene/Robot", "Xform")
    robot.GetReferences().AddReference(str(usd))
    robot.SetInstanceable(True)
    assert not scene.GetCompositionErrors()
    for name, path in paths.items():
        imported_path = path.replace("/World/Robot", "/Scene/Robot") + "/inner/Geometry"
        assert _bound_state(scene, imported_path)["color"] == {"red": (1, 0, 0), "blue": (0, 0, 1)}.get(name, (1, 1, 1))
        assert scene.GetPrimAtPath(imported_path).IsInstanceProxy()


@pytest.mark.parametrize("selected", [False, True], ids=["uncoloured-consumer", "coloured-consumer"])
def test_material_resolved_outside_default_traversal_is_discovered(tmp_path: Path, *, selected: bool) -> None:
    urdf, usd, paths, sources = _shared_fixture(tmp_path, nested=False, external=True)
    stage = Usd.Stage.Open(str(usd))
    material_path = "/World/Looks/ClassMaterial"
    material = _material(stage, material_path)
    material.GetPrim().SetSpecifier(Sdf.SpecifierClass)
    target = paths["red" if selected else "collision"]
    UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(target)).Bind(material)
    stage.GetRootLayer().Save()
    # The material is valid and resolves through shading APIs, but default traversal excludes it.
    # This is a USD-composition regression; no renderer support for class materials is assumed.
    assert not stage.GetCompositionErrors()
    assert material.GetPrim().IsAbstract()
    assert material_path not in {str(prim.GetPath()) for prim in stage.Traverse()}
    assert _bound_state(stage, target + "/Geometry")["color"] == (1, 1, 1)
    original = _root_branch(stage, material_path)

    assert apply_urdf_visual_colors(str(urdf), str(usd)) == 2
    for name, path in paths.items():
        state = _bound_state(stage, path + "/Geometry")
        assert state["color"] == {"red": (1, 0, 0), "blue": (0, 0, 1)}.get(name, (1, 1, 1))
        assert stage.GetPrimAtPath(path).IsInstance()
    assert _root_branch(stage, material_path) == original
    assert {p: p.read_bytes() for p in sources} == sources
    saved = usd.read_bytes()
    assert apply_urdf_visual_colors(str(urdf), str(usd)) == 2
    assert usd.read_bytes() == saved


@pytest.mark.parametrize(
    ("original", "imported"),
    [("arm-link", "arm_link"), ("_link", "a_link"), ("1link", "a_link")],
    ids=["dash", "leading-underscore", "leading-digit"],
)
def test_importer_normalized_link_names_receive_the_original_colour(
    tmp_path: Path, original: str, imported: str
) -> None:
    urdf = tmp_path / "names.urdf"
    urdf.write_text(
        f'<robot name="names"><link name="{original}"><visual><geometry><box size="1 1 1"/></geometry>'
        '<material name="red"><color rgba="1 0 0 1"/></material></visual></link></robot>'
    )
    usd = tmp_path / "names.usda"
    stage = Usd.Stage.CreateNew(str(usd))
    stage.SetDefaultPrim(stage.DefinePrim("/Robot", "Xform"))
    path = f"/Robot/{imported}/visuals/shape/Geometry"
    mesh = _mesh(stage, path)
    material = _material(stage, "/Robot/Looks/Material")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()
    before = _bound_state(stage, path)

    assert apply_urdf_visual_colors(str(urdf), str(usd)) == 1
    after = _bound_state(stage, path)
    assert after["color"] == (1, 0, 0)
    for key in ("metallic", "texture", "points", "faces", "double_sided", "opacity"):
        assert after[key] == before[key]


@pytest.mark.parametrize(
    ("originals", "imported"),
    [(("arm-link", "arm_link"), "arm_link"), (("_link", "1link"), "a_link")],
)
def test_normalized_link_collisions_warn_and_skip_only_the_ambiguous_link(
    tmp_path: Path, colour_warnings: list[str], originals: tuple[str, str], imported: str
) -> None:
    urdf = tmp_path / "collision.urdf"
    urdf.write_text(
        '<robot name="collision">'
        + "".join(
            f'<link name="{name}"><visual><geometry><box size="1 1 1"/></geometry>'
            '<material name="red"><color rgba="1 0 0 1"/></material></visual></link>'
            for name in (*originals, "supported")
        )
        + "</robot>"
    )
    usd = tmp_path / "collision.usda"
    stage = Usd.Stage.CreateNew(str(usd))
    stage.SetDefaultPrim(stage.DefinePrim("/Robot", "Xform"))
    material = _material(stage, "/Robot/Looks/Material")
    for name in (imported, "supported"):
        mesh = _mesh(stage, f"/Robot/{name}/visuals/shape/Geometry")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()
    path = f"/Robot/{imported}/visuals/shape/Geometry"
    before = _bound_state(stage, path)

    assert apply_urdf_visual_colors(str(urdf), str(usd)) == 1
    assert _bound_state(stage, path) == before
    assert _bound_state(stage, "/Robot/supported/visuals/shape/Geometry")["color"] == (1, 0, 0)
    warning = "\n".join(colour_warnings)
    assert "ambiguous" in warning and imported in warning
    assert all(name in warning for name in originals)


def test_prior_inline_material_reuse_reaches_both_bound_visuals(tmp_path: Path) -> None:
    urdf, usd, paths, sources = _shared_fixture(tmp_path, nested=True, external=True)
    urdf.write_text(
        '<robot name="reuse"><link name="mixed">'
        '<visual><geometry><box size="1 1 1"/></geometry>'
        '<material name="red"><color rgba="1 0 0 1"/></material></visual>'
        '<visual><geometry><box size="1 1 1"/></geometry><material name="red"/></visual>'
        "<visual/></link></robot>"
    )
    assert apply_urdf_visual_colors(str(urdf), str(usd)) == 2
    stage = Usd.Stage.Open(str(usd))
    for name, path in paths.items():
        expected = (1, 0, 0) if name in ("red", "blue") else (1, 1, 1)
        assert _bound_state(stage, path + "/inner/Geometry")["color"] == expected
        assert stage.GetPrimAtPath(path).IsInstance()
    assert {p: p.read_bytes() for p in sources} == sources


def test_optional_colour_failure_restores_cached_layer_and_keeps_asset_usable(
    tmp_path: Path, monkeypatch, colour_warnings
) -> None:
    from holosoma.simulator.isaacsim import urdf_visual_colors as colors

    usd = tmp_path / "converted.usda"
    stage = Usd.Stage.CreateNew(str(usd))
    UsdGeom.Cube.Define(stage, "/Robot/body")
    stage.GetRootLayer().Save()
    original = stage.GetRootLayer().ExportToString()
    original_bytes = usd.read_bytes()

    def fail_after_partial_edit(urdf_path, usd_path):
        cached = Usd.Stage.Open(usd_path)
        UsdGeom.Sphere.Define(cached, "/UnwantedPartialEdit")
        raise ValueError("unsupported colour metadata")

    monkeypatch.setattr(colors, "apply_urdf_visual_colors", fail_after_partial_edit)
    assert colors.try_apply_urdf_visual_colors("unused.urdf", str(usd)) == 0
    assert stage.GetRootLayer().ExportToString() == original
    assert usd.read_bytes() == original_bytes
    assert Usd.Stage.Open(str(usd)).GetPrimAtPath("/Robot/body").IsA(UsdGeom.Cube)
    assert any("optional URDF visual colours" in warning for warning in colour_warnings)
