"""Lab 3 schema overrides after normal spawning and before scene replication."""

from __future__ import annotations

import re
import copy
from functools import partial, update_wrapper
from typing import Any

from dataclasses import asdict
from holosoma.config_types.scene import PhysicsConfig


_FAMILIES = {
    "rigid_props": ("RigidBodyFragment", "apply_rigid_body_properties"),
    "mass_props": ("MassFragment", "apply_mass_properties"),
    "collision_props": ("CollisionFragment", "apply_collision_properties"),
    "articulation_props": ("ArticulationRootFragment", "apply_articulation_root_properties"),
    "joint_drive_props": ("JointDriveFragment", "apply_joint_drive_properties"),
    "fixed_tendons_props": ("FixedTendonFragment", "apply_fixed_tendon_properties"),
    "spatial_tendons_props": ("SpatialTendonFragment", "apply_spatial_tendon_properties"),
    "mesh_collision_props": ("MeshCollisionFragment", "apply_mesh_collision_properties"),
    "physics_material": ("RigidBodyMaterialFragment", None),
}
_FLAGS = {
    "mass_props_create_if_missing",
    "articulation_props_create_if_missing",
    "joint_drive_props_create_if_missing",
    "ensure_drives_exist",
}


def physics_settings(physics: PhysicsConfig | None, backend: Any) -> dict[str, Any]:
    """Translate object/robot ``PhysicsConfig`` into Lab 3 schema fragment lists.

    The spawn wrapper applies these before cloning. Common mass/material/PhysX
    settings use the ordinary spawner; these fragments add Newton contact and
    mesh controls without changing unspecified asset properties.
    """
    if physics is None:
        return {}
    from isaaclab.sim.spawners.materials import UsdPhysicsRigidBodyMaterialCfg

    result: dict[str, Any] = {}
    if backend.options.physics_backend != "newton":
        return result
    from isaaclab_newton.sim.schemas import (
        MujocoCollisionCfg,
        NewtonCollisionCfg,
        NewtonMeshCollisionCfg,
        NewtonSDFCollisionCfg,
    )
    from isaaclab_newton.sim.spawners.materials import NewtonMaterialCfg

    mj = physics.mujoco
    if mj is not None:
        unsupported = {name for name in ("contype", "conaffinity") if getattr(mj, name) is not None} | set(mj.extra)
        if unsupported:
            raise ValueError("physics.mujoco: Newton USD importer does not expose: " + ", ".join(sorted(unsupported)))
        collision = {
            name: tuple(value) if isinstance(value, list) else value
            for name in ("condim", "priority", "solimp", "solmix", "solref")
            if (value := getattr(mj, name)) is not None
        }
        if collision:
            result.setdefault("collision_props", []).append(MujocoCollisionCfg(**collision))
        if mj.margin is not None or mj.gap is not None:
            result.setdefault("collision_props", []).append(
                NewtonCollisionCfg(contact_margin=mj.margin, contact_gap=mj.gap)
            )
        if mj.friction is not None:
            sliding, torsional, rolling = mj.friction
            result.setdefault("physics_material", []).extend(
                [
                    UsdPhysicsRigidBodyMaterialCfg(static_friction=sliding, dynamic_friction=sliding),
                    NewtonMaterialCfg(torsional_friction=torsional, rolling_friction=rolling),
                ]
            )
    if physics.newton is not None:
        values = {name: value for name, value in asdict(physics.newton).items() if value is not None}
        contacts = {name: value for name, value in values.items() if name.startswith("contact_")}
        if contacts:
            result.setdefault("physics_material", []).append(NewtonMaterialCfg(**contacts))
        if "max_hull_vertices" in values:
            result["mesh_collision_props"] = [NewtonMeshCollisionCfg(max_hull_vertices=values["max_hull_vertices"])]
        sdf = {name: value for name, value in values.items() if name.startswith(("sdf_", "hydroelastic_"))}
        if sdf:
            result.setdefault("mesh_collision_props", []).append(NewtonSDFCollisionCfg(**sdf))
    return result


def author_robot_physics(anchor: str, asset: Any, backend: Any, *, stage: Any, material_defaults: Any) -> None:
    """Author robot-wide Newton contact properties before environment cloning."""
    settings = physics_settings(asset.link_physics, backend)
    author_properties(anchor, compile_properties(settings), flags={}, stage=stage, material_defaults=material_defaults)


def _target_expression(anchor: str, pattern: str) -> str:
    """Keep every alternative within the owning asset, including grouped suffixes."""
    return re.escape(anchor) + f"(?=/|$)(?:{pattern})"


def schema_rules(value: Any, *, location: str) -> list[tuple[str, list[Any]]]:
    """A fragment/list is shorthand for all descendants; targeted rules retain order."""
    if value is None or value == []:
        return []
    entries = value if isinstance(value, list) else [value]
    targeted = any(isinstance(entry, dict) and "match" in entry for entry in entries)
    if not targeted:
        return [("(/.*)?", entries)]
    result = []
    for index, entry in enumerate(entries):
        path = f"{location}[{index}]"
        if not isinstance(entry, dict) or set(entry) != {"match", "fragments"}:
            raise ValueError(f"{path}: each target rule requires only match and fragments")
        pattern = entry["match"]
        if not isinstance(pattern, str) or (pattern and not pattern.startswith(("/", "("))):
            raise ValueError(f"{path}.match: expected a relative USD path regex suffix")
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"{path}.match: invalid regex: {exc}") from exc
        fragments = entry["fragments"]
        if not isinstance(fragments, list):
            raise ValueError(f"{path}.fragments: expected a fragment list")
        result.append((pattern, fragments))
    return result


def wrap_spawn(cfg: Any, settings: dict[str, Any], *, backend: Any, material_defaults: Any = None) -> Any:
    """Keep the existing format/rigid-body behavior, then author native overrides."""
    if not settings or not backend.is_isaaclab3:
        return cfg
    compiled = compile_properties(settings)
    if not any(fragments for rules in compiled.values() for _pattern, fragments in rules):
        return cfg
    cfg.func = update_wrapper(
        partial(
            _spawn,
            original=cfg.func,
            compiled=compiled,
            flags={name: settings[name] for name in _FLAGS & settings.keys()},
            backend=backend,
            material_defaults=material_defaults,
        ),
        _spawn,
    )
    return cfg


def compile_properties(settings: dict[str, Any]) -> dict[str, list[tuple[str, list[Any]]]]:
    """Validate schema families and compile fragments without editing a stage."""
    unknown = settings.keys() - (_FAMILIES.keys() | _FLAGS)
    if unknown:
        raise ValueError(f"physics: unknown schema settings: {', '.join(sorted(unknown))}")
    for flag in _FLAGS & settings.keys():
        if not isinstance(settings[flag], bool):
            raise ValueError(f"physics.{flag}: expected a boolean")

    from isaaclab.sim import schemas
    from isaaclab.sim.spawners import materials

    compiled = {}
    for family in _FAMILIES:
        if family not in settings:
            continue
        rules = schema_rules(settings[family], location=f"physics.{family}")
        marker, _writer = _FAMILIES[family]
        expected = getattr(materials if family == "physics_material" else schemas, marker)
        for _pattern, fragments in rules:
            if any(not isinstance(fragment, expected) for fragment in fragments):
                raise ValueError(f"physics.{family}: expected {marker} configurations")
        compiled[family] = rules
    return compiled


def _spawn(
    prim_path: str,
    cfg: Any,
    *args: Any,
    original: Any,
    compiled: dict[str, list[tuple[str, list[Any]]]],
    flags: dict[str, bool],
    backend: Any,
    material_defaults: Any,
    **kwargs: Any,
) -> Any:
    # The converter serializes the incoming cfg to form its cache key. Keep
    # its original spawn function rather than serializing this closure/partial.
    spawn_cfg = copy.copy(cfg)
    spawn_cfg.func = original
    prim = original(prim_path, spawn_cfg, *args, **kwargs)
    stage = backend.stage_utils.get_current_stage()
    author_properties(str(prim.GetPath()), compiled, flags=flags, stage=stage, material_defaults=material_defaults)
    return prim


def author_properties(
    anchor: str,
    compiled: dict[str, list[tuple[str, list[Any]]]],
    *,
    flags: dict[str, bool],
    stage: Any,
    material_defaults: Any = None,
) -> None:
    """Dispatch public SDK writers, failing when requested targets do not exist."""
    from isaaclab.sim import schemas
    from isaaclab.sim.utils import find_matching_prims
    from pxr import UsdGeom, UsdPhysics

    if not any(fragments for rules in compiled.values() for _pattern, fragments in rules):
        return
    # Editing a link's mass must not deinstance unrelated visual geometry.
    # Only remove instancing from ancestors of eligible requested targets.
    carriers = {
        "rigid_props": UsdPhysics.RigidBodyAPI,
        "mass_props": UsdPhysics.MassAPI,
        "collision_props": UsdPhysics.CollisionAPI,
        "physics_material": UsdPhysics.CollisionAPI,
        "mesh_collision_props": UsdPhysics.CollisionAPI,
        "articulation_props": UsdPhysics.ArticulationRootAPI,
    }
    instances = set()
    for family, rules in compiled.items():
        api = carriers.get(family)
        create = flags.get(f"{family}_create_if_missing", False)
        for pattern, fragments in rules:
            if not fragments:
                continue
            for prim in find_matching_prims(_target_expression(anchor, pattern), stage=stage):
                if api is not None and not create and not prim.HasAPI(api):
                    continue
                if family == "mesh_collision_props" and not prim.IsA(UsdGeom.Mesh):
                    continue
                if family == "joint_drive_props" and not (
                    prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint)
                ):
                    continue
                ancestor = prim
                while ancestor and str(ancestor.GetPath()).startswith(anchor):
                    if ancestor.IsInstance():
                        instances.add(str(ancestor.GetPath()))
                    ancestor = ancestor.GetParent()
    for path in sorted(instances):
        stage.GetPrimAtPath(path).SetInstanceable(False)

    for family, rules in compiled.items():
        _marker, writer_name = _FAMILIES[family]
        for index, (pattern, fragments) in enumerate(rules):
            if not fragments:
                continue
            expression = _target_expression(anchor, pattern)
            if family == "physics_material":
                _bind_materials(
                    anchor, expression, fragments, stage=stage, defaults=material_defaults, rule_index=index
                )
                continue
            assert writer_name is not None
            writer = getattr(schemas, writer_name)
            if family == "mesh_collision_props":
                targets = [
                    prim
                    for prim in find_matching_prims(expression, stage=stage)
                    if prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI)
                ]
                success = bool(targets)
                for prim in targets:
                    success = writer(str(prim.GetPath()), fragments, stage=stage) and success
            else:
                options: dict[str, Any] = {"stage": stage}
                if family in ("mass_props", "articulation_props", "joint_drive_props"):
                    options["create_if_missing"] = flags.get(f"{family}_create_if_missing", False)
                if family == "joint_drive_props":
                    options["ensure_drives_exist"] = flags.get("ensure_drives_exist", False)
                success = writer(expression, fragments, **options)
            if not success:
                raise ValueError(f"physics.{family}: no editable targets or failed writer for {expression!r}")


def _bind_materials(
    anchor: str,
    expression: str,
    fragments: list[Any],
    *,
    stage: Any,
    defaults: Any,
    rule_index: int,
) -> None:
    from isaaclab.sim.spawners.materials import spawn_physics_material
    from isaaclab.sim.utils import bind_physics_material, find_matching_prims
    from pxr import UsdPhysics, UsdShade

    targets = [prim for prim in find_matching_prims(expression, stage=stage) if prim.HasAPI(UsdPhysics.CollisionAPI)]
    if not targets:
        raise ValueError(f"physics.physics_material: no colliders for {expression!r}")
    for index, prim in enumerate(targets):
        material_path = f"{anchor}/holosoma_native_material_{rule_index}_{index}"
        # Preserve each collider's authored material through composition, then
        # apply portable defaults and the requested native fragment overrides.
        bound, _relationship = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose="physics")
        material = UsdShade.Material.Define(stage, material_path)
        if bound and str(bound.GetPath()) != material_path:
            material.GetPrim().GetReferences().AddInternalReference(bound.GetPath())
        # A bound material already includes portable defaults and any earlier
        # rule. Reapplying defaults here would erase a prior rule's attributes.
        if not bound and defaults is not None:
            spawn_physics_material(material_path, defaults, stage=stage)
        spawn_physics_material(material_path, fragments, stage=stage)
        bind_physics_material(str(prim.GetPath()), material_path, stage=stage)
