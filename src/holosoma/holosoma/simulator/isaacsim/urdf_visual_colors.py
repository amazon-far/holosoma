"""Apply URDF visual colours to converted USD without changing referenced assets."""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from typing import Any, Tuple

from loguru import logger

# Shader colour inputs to overwrite: OmniPBR (what the importer's mesh converter emits) and UsdPreviewSurface.
_COLOR_INPUTS = ("inputs:diffuse_color_constant", "inputs:diffuseColor")
Rgb = Tuple[float, float, float]


def urdf_visual_color_groups(urdf_path: str) -> dict[str, list[Rgb | None]]:
    """Return each link's URDF visual colours in document order.

    ``None`` means that visual has no URDF material colour. A material may define its colour inline or reference a
    top-level or previously defined inline named material.
    """
    root = ET.parse(urdf_path).getroot()  # noqa: S314  # shipped asset files
    named: dict[str, str] = {}
    for material in root.findall("material"):
        color = material.find("color")
        name = material.get("name")
        if color is not None and name is not None and color.get("rgba") is not None:
            named[name] = str(color.get("rgba"))
    result: dict[str, list[Rgb | None]] = {}
    for link in root.iter("link"):
        link_name = link.get("name")
        visuals = link.findall("visual")
        if link_name is None or not visuals:
            continue
        colors: list[Rgb | None] = []
        for visual in visuals:
            material = visual.find("material")
            color = material.find("color") if material is not None else None
            rgba = None
            if material is not None:
                rgba = color.get("rgba") if color is not None else named.get(str(material.get("name")))
                if color is not None and rgba is not None and material.get("name") is not None:
                    # Later references may reuse an inline definition. Keep top-level / earlier definitions
                    # for references, while this visual's explicit colour still takes precedence above.
                    named.setdefault(str(material.get("name")), rgba)
            if rgba is None:
                colors.append(None)
            else:
                r, g, b = (float(v) for v in rgba.split()[:3])
                colors.append((r, g, b))
        result[link_name] = colors
    return result


def match_visual_colors_to_groups(colors: Sequence[Rgb | None], groups: Sequence[str]) -> dict[str, Rgb]:
    """Map ordered URDF visuals to ordered imported USD visual groups.

    IsaacLab's importer preserves one top-level group per URDF ``<visual>`` in document order. If a uniformly
    coloured visual is split into several groups, applying that one colour to every group is also unambiguous.
    A mixed-colour count mismatch is left untouched because guessing would recolour the wrong geometry.
    """
    if len(colors) == len(groups):
        return {group: color for group, color in zip(groups, colors) if color is not None}
    if colors and colors[0] is not None and all(color == colors[0] for color in colors):
        return {group: colors[0] for group in groups}
    return {}


def _make_editable(stage: Any, path: Any) -> None:
    # Deinstancing recomposes descendants: retain paths, never prim handles across these edits.
    for prefix in path.GetPrefixes():
        prim = stage.GetPrimAtPath(prefix)
        if prim and prim.IsInstance():
            prim.SetInstanceable(False)


def _material_network(stage: Any, material_path: Any) -> dict[Any, dict[str, list[Any]]]:
    # Follow material connections: importers may place shaders beside the material.
    pending = [material_path]
    network: dict[Any, dict[str, list[Any]]] = {}
    while pending:
        path = pending.pop()
        if path in network:
            continue
        prim = stage.GetPrimAtPath(path)
        if not prim:
            # Keep the unresolved endpoint in the network so a selected visual can be skipped without losing
            # sharing information about the valid nodes. Unrelated incomplete materials are otherwise harmless.
            network[path] = {}
            continue
        connections = {attr.GetName(): attr.GetConnections() for attr in prim.GetAttributes()}
        network[path] = {name: targets for name, targets in connections.items() if targets}
        pending.extend(target.GetPrimPath() for targets in network[path].values() for target in targets)
    return network


def _binding_owner(stage: Any, group_path: Any) -> Any:
    default = stage.GetDefaultPrim()
    return default.GetPath() if default and group_path.HasPrefix(default.GetPath()) else group_path.GetPrefixes()[0]


def _local_material(stage: Any, group_path: Any, material_path: Any, network: dict[Any, dict[str, list[Any]]]) -> Any:
    from pxr import UsdShade

    key = hashlib.sha256(f"{group_path}:{material_path}".encode()).hexdigest()[:16]
    local_path = _binding_owner(stage, group_path).AppendPath(f"__holosoma_urdf_colors/material_{key}")
    _make_editable(stage, local_path.GetParentPath())
    stage.DefinePrim(local_path.GetParentPath(), "Scope")
    stage.OverridePrim(local_path).GetReferences().AddInternalReference(material_path)
    mapping = {}
    for source in network:
        if source.HasPrefix(material_path):
            target = source.ReplacePrefix(material_path, local_path)
        else:
            node_key = hashlib.sha256(str(source).encode()).hexdigest()[:16]
            target = local_path.AppendChild(f"node_{node_key}")
            stage.OverridePrim(target).GetReferences().AddInternalReference(source)
        mapping[source] = target
    for source, connections in network.items():
        target = mapping[source]
        _make_editable(stage, target)
        for name, targets in connections.items():
            # Internal references preserve shading attributes and asset-path anchoring. Only connection targets
            # leaving a copied node need relocation into this visual's private network.
            stage.GetPrimAtPath(target).GetAttribute(name).SetConnections(
                [mapping[p.GetPrimPath()].AppendProperty(p.name) for p in targets]
            )
    return UsdShade.Material(stage.GetPrimAtPath(local_path))


def apply_urdf_visual_colors(urdf_path: str, usd_path: str) -> int:
    """Colour materials bound to URDF visuals; return the number of updated shaders.

    Repeated calls update supplied colours and retain earlier edits where the URDF has no colour. Reconvert the
    URDF to USD before calling to remove earlier colour edits.

    Collection bindings at the robot root select private materials, preserving instances below that root and
    uncoloured consumers, including internal references and inherits. Source assets stay unchanged. Strong ancestor
    bindings, unresolved material connections, and unbound shader definitions are left unchanged with a warning.
    """
    from pxr import Gf, Sdf, Tf, Usd, UsdGeom, UsdShade

    colors = urdf_visual_color_groups(urdf_path)
    if not colors:
        return 0
    stage = Usd.Stage.Open(Sdf.Layer.FindOrOpen(usd_path), Sdf.Layer.CreateAnonymous())
    stage.SetEditTarget(stage.GetRootLayer())
    # A private session layer hides our saved bindings while resolving the original materials on repeat calls.
    for prim in stage.Traverse():
        for rel in prim.GetRelationships():
            if rel.GetName().startswith("material:binding:collection:") and ":holosomaUrdfColor_" in rel.GetName():
                with Usd.EditContext(stage, stage.GetSessionLayer()):
                    rel.SetTargets([])
    groups: dict[str, dict[str, list[Any]]] = {}
    geometry, shaders, networks = [], [], {}
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        path = prim.GetPath()
        if prim.GetParent().GetName() == "visuals":
            link = prim.GetParent().GetParent().GetName()
            groups.setdefault(link, {}).setdefault(prim.GetName(), []).append(path)
        if prim.IsA(UsdGeom.Gprim) or prim.IsA(UsdGeom.Subset):
            geometry.append(path)
        if prim.IsA(UsdShade.Shader):
            shaders.append(path)
        if prim.IsA(UsdShade.Material):
            networks[path] = _material_network(stage, path)

    imported_names = {}
    original_names: dict[str, list[str]] = {}
    for link in colors:
        # IsaacSim 5.1's makeValidUSDIdentifier adds 'a' even for an originally valid leading underscore.
        name = Tf.MakeValidIdentifier(link)
        imported_names[link] = "a" + name if name.startswith("_") else name
        original_names.setdefault(imported_names[link], []).append(link)
    ambiguous = {name for name, originals in original_names.items() if len(originals) > 1}
    for name in sorted(ambiguous):
        logger.warning(
            f"IsaacSim: URDF links {original_names[name]} have the same imported USD name '{name}'; "
            "ambiguous visual colours left unchanged"
        )

    selected = {}
    for link, per_visual in colors.items():
        if imported_names[link] in ambiguous:
            continue
        visual_groups = groups.get(imported_names[link], {})
        matched = match_visual_colors_to_groups(per_visual, list(visual_groups))
        if not matched and any(color is not None for color in per_visual) and visual_groups:
            logger.warning(
                f"IsaacSim: URDF link '{link}' has {len(per_visual)} mixed-colour visual(s), but the converted USD "
                f"has {len(visual_groups)} visual group(s); left it as imported rather than guessing the mapping"
            )
            continue
        for visual, color in matched.items():
            selected.update({path: color for path in visual_groups[visual]})

    def selected_group(path: Any) -> Any:
        return next((group for group in selected if path.HasPrefix(group)), None)

    bindings = []
    skipped: dict[Any, str] = {}
    for path in geometry:
        group = selected_group(path)
        seen = set()
        for purpose in UsdShade.MaterialBindingAPI.GetMaterialPurposes():
            material, binding = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(path)).ComputeBoundMaterial(purpose)
            if not material or binding.GetPath() in seen:
                continue
            seen.add(binding.GetPath())
            material_path = material.GetPath()
            # Binding resolution can reach materials excluded by default traversal (e.g. class prims).
            # Discover their complete network from the actual binding too.
            if material_path not in networks:
                networks[material_path] = _material_network(stage, material_path)
            network = networks[material_path]
            if group is not None:
                unresolved = any(
                    not stage.GetPropertyAtPath(target)
                    for connections in network.values()
                    for targets in connections.values()
                    for target in targets
                )
                if unresolved:
                    skipped.setdefault(group, f"unresolved connections in material '{material_path}'")
                    continue
                if not any(
                    stage.GetPrimAtPath(node).IsA(UsdShade.Shader)
                    and any(stage.GetPrimAtPath(node).GetAttribute(name) for name in _COLOR_INPUTS)
                    for node in network
                ):
                    continue
                strength = UsdShade.MaterialBindingAPI.GetMaterialBindingStrength(binding)
                if (
                    not binding.GetPrim().GetPath().HasPrefix(group)
                    and strength == UsdShade.Tokens.strongerThanDescendants
                ):
                    skipped.setdefault(group, f"strong ancestor material binding '{binding.GetPath()}'")
                    continue
                bindings.append((group, path, material_path, binding.GetPath(), purpose))

    for group, reason in skipped.items():
        logger.warning(f"IsaacSim: URDF visual '{group}' has {reason}; left unchanged")
        selected.pop(group)

    patched_paths = set()

    def paint(paths: Any, color: Rgb) -> None:
        for path in paths:
            prim = stage.GetPrimAtPath(path)
            if not prim.IsA(UsdShade.Shader):
                continue
            attributes = [name for name in _COLOR_INPUTS if prim.GetAttribute(name)]
            if not attributes:
                continue
            _make_editable(stage, path)
            for name in attributes:
                attr = stage.GetPrimAtPath(path).GetAttribute(name)
                # An explicit URDF colour overrides a connected colour texture, but leaves other inputs intact.
                if attr.GetConnections():
                    attr.SetConnections([])
                attr.Set(Gf.Vec3f(*color))
            patched_paths.add(path)

    stage.GetSessionLayer().Clear()
    # Keep colour and binding opinions outside the visual branches: a reference to an entire visual
    # must inherit neither. Explicit collection membership also reaches geometry inside instances.
    materials = {}
    collections: dict[tuple[Any, Any, str, Any], list[Any]] = {}
    for group, path, source, binding_path, purpose in bindings:
        if group not in selected:
            continue
        key = (group, source)
        if key not in materials:
            material = _local_material(stage, group, source, networks[source])
            materials[key] = material
            paint(_material_network(stage, material.GetPath()), selected[group])
        collection_key = (group, source, purpose, binding_path)
        collections.setdefault(collection_key, []).append(path)
    binding_order: dict[Any, list[str]] = {}
    for (group, source, purpose, binding_path), paths in collections.items():
        owner = stage.GetPrimAtPath(_binding_owner(stage, group))
        name = (
            "holosomaUrdfColor_"
            + hashlib.sha256(f"{group}:{source}:{purpose}:{binding_path}".encode()).hexdigest()[:16]
        )
        collection = Usd.CollectionAPI.Apply(owner, name)
        collection.CreateExpansionRuleAttr(Usd.Tokens.explicitOnly)
        collection.CreateIncludesRel().SetTargets(paths)
        api = UsdShade.MaterialBindingAPI.Apply(owner)
        # A root binding must outrank the original descendant binding, which remains untouched.
        api.Bind(collection, materials[(group, source)], name, UsdShade.Tokens.strongerThanDescendants, purpose)
        rel = api.GetCollectionBindingRel(name, purpose)
        for metadata, value in stage.GetRelationshipAtPath(binding_path).GetAllAuthoredMetadata().items():
            if metadata != "bindMaterialAs":
                rel.SetMetadata(metadata, value)
        binding_order.setdefault(owner.GetPath(), []).append(rel.GetName())
    for owner_path, names in binding_order.items():
        owner = stage.GetPrimAtPath(owner_path)
        # Collection precedence follows property order, even when both bindings are strong.
        owner.SetPropertyOrder(names + [name for name in owner.GetPropertyOrder() if name not in names])

    # Unbound shader definitions have no identified geometry consumers; editing them could leak through
    # internal references just as editing a shared material would.
    connected_nodes = {node for network in networks.values() for node in network}
    unbound_groups = {selected_group(path) for path in shaders if path not in connected_nodes}
    for group in unbound_groups - {None}:
        logger.warning(f"IsaacSim: URDF visual '{group}' has unbound shaders; left those definitions unchanged")

    patched = len(patched_paths)
    if patched:
        stage.GetRootLayer().Save()
    logger.info(f"IsaacSim: applied URDF visual colours to {patched} shader(s) of {usd_path}")
    return patched


def try_apply_urdf_visual_colors(urdf_path: str, usd_path: str) -> int:
    """Best-effort appearance repair; failure leaves the converted articulation usable.

    Restore the cached root layer on failure too: unsaved USD edits can otherwise affect the
    subsequent spawn even though no file was saved. Conversion and spawning errors remain fatal.
    """
    layer, original = None, None
    try:
        from pxr import Sdf

        layer = Sdf.Layer.FindOrOpen(usd_path)
        if layer is None:
            raise ValueError(f"Cannot open converted USD {usd_path!r}")
        original = layer.ExportToString()
        return apply_urdf_visual_colors(urdf_path, usd_path)
    except Exception as exc:
        if layer is not None and original is not None:
            layer.ImportFromString(original)
        logger.warning(f"IsaacSim: could not apply optional URDF visual colours to {usd_path}: {exc}")
        return 0


def spawn_from_urdf_with_visual_colors(
    prim_path: str, cfg: Any, translation: Any = None, orientation: Any = None
) -> Any:
    """IsaacLab URDF spawn with a best-effort visual-colour pass between conversion and spawning.

    Mirrors ``isaaclab.sim.spawners.from_files.spawn_from_urdf`` (convert, then spawn the USD) and must be wrapped
    with ``isaaclab.sim.utils.clone`` where it is assigned, like the original.
    """
    from isaaclab.sim.converters import UrdfConverter
    from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file

    converter = UrdfConverter(cfg)
    try_apply_urdf_visual_colors(cfg.asset_path, converter.usd_path)
    return _spawn_from_usd_file(prim_path, converter.usd_path, cfg, translation, orientation)
