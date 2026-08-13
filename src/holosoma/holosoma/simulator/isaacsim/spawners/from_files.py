# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Adapted from Isaac Lab v2.0.0 (https://github.com/isaac-sim/IsaacLab)
# Contributors: https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md
#
# This file contains utilities adapted from Isaac Lab's spawner code.

"""Custom USD file spawners with source prim path support.

This module provides custom USD file spawning functionality that supports loading
specific prims from USD files.

Key Differences from IsaacLab's Built-in from_files.py:

1. **Source Path Support**: Added 'source_path' parameter to load specific prims from USD files
   instead of loading the entire file. This enables selective loading from complex USD scenes.

2. **Enhanced Schema Handling**: Implements USD schema (API) handling for properties that
   may not exist in source USD files. Uses ensure_api_and_modify() utility to automatically
   apply required APIs before setting properties.

3. **Schema "Ensure" Implementation Status**:
   - **rigid_props**: Uses ensure_api_and_modify() with UsdPhysics.RigidBodyAPI
   - **collision_props**: Uses ensure_api_and_modify() with UsdPhysics.CollisionAPI
   - **mass_props**: Uses ensure_api_and_modify() with UsdPhysics.MassAPI
   - **articulation_props**: Uses standard schemas.modify_* (may fail if API missing)
   - **fixed_tendons_props**: Uses standard schemas.modify_* (may fail if API missing)
   - **joint_drive_props**: Uses standard schemas.modify_* (may fail if API missing)
   - **deformable_props**: Uses standard schemas.modify_* (may fail if API missing)

4. **Optional Instanceable Handling**: By default, disables instanceable flag before applying
   physics properties to avoid USD instancing conflicts. Can be disabled by setting
   `disable_instanceable=False` in the configuration.

Background:
IsaacLab's original from_files.py uses schemas.modify_*() functions which assume the required
USD APIs already exist in the source files. This fails for prims that only have collision APIs
but no rigid body APIs. Our enhanced version ensures the required APIs exist before modification.

The "ensure" functionality is currently implemented for the most commonly used properties
(rigid body, collision, mass). Other properties still use the original IsaacLab approach
and may fail silently if the required APIs don't exist in the source USD files.

NOTE: This is copied and adapted from IsaacLab with patches for 'source_path' and enhanced
schema handling. This is a temporary work-around until IsaacLab has these capabilities.
"""

from __future__ import annotations

import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
import omni.kit.commands
import omni.log
from isaaclab.sim import schemas
from isaaclab.sim.utils import bind_physics_material, bind_visual_material, clone, select_usd_variants
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade

from holosoma.simulator.isaacsim.spawners import from_files_cfg
from holosoma.simulator.isaacsim.spawners.schema_utils import ensure_api_and_modify
from holosoma.simulator.isaacsim.prim_utils import set_instanceable

import carb


def create_prim(target_path, usd_path, source_path=None, translation=None, orientation=None, scale=None):
    """Create a prim at the target path by referencing a USD file.

    Parameters
    ----------
    target_path : str
        The prim path where the asset should be created.
    usd_path : str
        Path to the USD file to reference.
    source_path : str, optional
        Optional prim path within the USD file to load specifically.
    translation : array-like, optional
        Translation [x, y, z] for the prim's local translate op.
    orientation : array-like, optional
        Orientation quaternion [w, x, y, z] for the prim's local orient op.
    scale : array-like, optional
        Scale [x, y, z] for the prim's local scale op.

    Returns
    -------
    Usd.Prim
        The created (or existing) prim.

    Raises
    ------
    FileNotFoundError
        If the USD file is not found at the specified path.

    Notes
    -----
    The local pose is authored directly as USD xformOps via ``UsdGeom.Xformable`` instead of
    ``isaacsim.core.prims.XFormPrim``, whose ``__init__`` decomposes the world matrix and rejects
    negative-scale (left-handed) frames. Each op is authored only when its value is provided; an
    unspecified component keeps the referenced asset's value.
    """
    # Check if prim exists
    stage = stage_utils.get_current_stage()
    prim = stage.GetPrimAtPath(target_path)
    if not prim.IsValid():
        prim = stage.DefinePrim(target_path, "Xform")
    if not prim:
        return None

    # Add reference with logging
    if source_path:
        carb.log_info(f"Loading {source_path} from {usd_path}")
        success = prim.GetReferences().AddReference(assetPath=usd_path, primPath=source_path)
    else:
        carb.log_info(f"Loading from {usd_path}")
        success = prim.GetReferences().AddReference(usd_path)

    if not success:
        raise FileNotFoundError(f"USD file not found: {usd_path}")

    if translation is not None or orientation is not None or scale is not None:
        xformable = UsdGeom.Xformable(prim)
        # Author a fresh op set (translate, orient, scale) at double precision so the local transform
        # is unambiguous and the op attribute types match the Gf*d values being set.
        double = UsdGeom.XformOp.PrecisionDouble
        xformable.ClearXformOpOrder()
        if translation is not None:
            xformable.AddTranslateOp(precision=double).Set(
                Gf.Vec3d(float(translation[0]), float(translation[1]), float(translation[2]))
            )
        if orientation is not None:
            # orientation is (w, x, y, z); GfQuatd takes (real, imaginary-vec).
            w, x, y, z = (float(c) for c in orientation)
            xformable.AddOrientOp(precision=double).Set(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
        if scale is not None:
            xformable.AddScaleOp(precision=double).Set(Gf.Vec3d(float(scale[0]), float(scale[1]), float(scale[2])))

    return prim


def _has_independent_motion(root_prim: "Usd.Prim") -> bool:
    """Whether ``root_prim``'s subtree has structure that lets its bodies move independently.

    Independent motion comes from a non-fixed joint (every typed joint subclasses
    ``UsdPhysics.Joint``) or an enabled articulation root — i.e. the asset is an articulation, not
    one rigid body. A fixed joint is a weld and does not count.
    """
    for prim in Usd.PrimRange(root_prim):
        if prim.IsA(UsdPhysics.Joint) and not prim.IsA(UsdPhysics.FixedJoint):
            return True
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            attr = prim.GetAttribute("physxArticulation:articulationEnabled")
            if not attr or not attr.IsValid() or attr.Get() is None or bool(attr.Get()):
                return True
    return False


def _collapse_to_single_rigid_body(stage: "Usd.Stage", prim_path: str) -> None:
    """Ensure the subtree at ``prim_path`` has exactly one prim with ``UsdPhysics.RigidBodyAPI``.

    IsaacLab resolves an asset's rigid body by counting prims with ``RigidBodyAPI`` applied
    (``HasAPI``, ignoring ``rigidBodyEnabled``) and requires exactly one. An asset authoring
    ``RigidBodyAPI`` on several sibling colliders of one object is collapsed to one ``RigidBodyAPI``
    on the asset root, keeping the former bodies as plain collider descendants.

    - 0 or 1 rigid-body prim: no-op.
    - >1 with no independent-motion structure (see :func:`_has_independent_motion`): collapse. The
      merged bodies must share one kinematic state; raise if they disagree.
    - >1 with independent-motion structure: raise (it is an articulation; load it as one).

    Descendant ``RemoveAPI`` is authored over the asset's reference/payload arc, which drops the
    ``HasAPI`` count; clearing ``rigidBodyEnabled`` would not.
    """
    root_prim = stage.GetPrimAtPath(prim_path)
    body_prims = [p for p in Usd.PrimRange(root_prim) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    # At most one rigid body already: nothing to collapse (the single-body case is stamped by
    # ensure_api_and_modify).
    if len(body_prims) <= 1:
        return

    if _has_independent_motion(root_prim):
        raise ValueError(
            f"Asset at '{prim_path}' has {len(body_prims)} rigid bodies AND joint/articulation "
            f"structure, so it is an articulation, not a single rigid body. Load it as an "
            f"articulation; the rigid-object path cannot represent independently-moving bodies."
        )

    # Kinematic state must be unanimous across the bodies being merged into one.
    def _is_kinematic(prim: "Usd.Prim") -> bool:
        attr = UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr()
        return bool(attr.Get()) if attr and attr.HasAuthoredValue() else False

    kinematic_flags = {_is_kinematic(p) for p in body_prims}
    if len(kinematic_flags) > 1:
        raise ValueError(
            f"Asset at '{prim_path}' has {len(body_prims)} rigid bodies with mixed "
            f"kinematicEnabled values {kinematic_flags}; they cannot be merged into one rigid body. "
            f"Author a consistent kinematic state, or split them into separate scene-file bodies."
        )
    kinematic = next(iter(kinematic_flags))

    # Author one RigidBodyAPI on the root with the agreed kinematic state, then RemoveAPI (USD +
    # PhysX) from every descendant so the colliders fall under the single root body.
    root_rb = UsdPhysics.RigidBodyAPI.Apply(root_prim)
    root_rb.CreateRigidBodyEnabledAttr(True)
    root_rb.CreateKinematicEnabledAttr(kinematic)
    for prim in body_prims:
        if prim.GetPath() == root_prim.GetPath():
            continue
        prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
        if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
            prim.RemoveAPI(PhysxSchema.PhysxRigidBodyAPI)

    remaining = [p for p in Usd.PrimRange(root_prim) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(remaining) != 1:
        raise RuntimeError(
            f"Failed to collapse '{prim_path}' to a single rigid body: {len(remaining)} remain "
            f"({[str(p.GetPath()) for p in remaining]}). The asset may be instanceable (so the "
            f"descendant edits no-op); ensure disable_instanceable is set on the spawn cfg."
        )


def _assert_asset_composed(stage: "Usd.Stage", prim_path: str, usd_path: str, source_path: str) -> None:
    """Fail loud if the referenced asset composed no renderable geometry or has dangling material bindings.

    The load-bearing guard against silent-invisible assets: referencing the wrong prim (an interior
    gprim, a prim whose materials live in a now-out-of-scope sibling/ancestor) makes ``AddReference``
    SUCCEED while the spawned prim ends up with no geometry or with material bindings that point
    outside the referenced subtree. PhysX still stamps, the object still spawns — invisible, with no
    error. This converts both failure modes into an actionable raise.

    Checks the composed subtree under ``prim_path``:
      1. GEOMETRY: at least one ``UsdGeom.Gprim`` descendant whose geometry actually resolved. For a
         ``Mesh`` that means a non-empty ``points`` array (a bare-Mesh reference yields a prim whose
         points did not compose); analytic gprims (``Cube``/``Sphere``/``Capsule``/...) carry their
         shape in attributes, so their mere presence counts.
      2. MATERIALS: every ``material:binding`` relationship that is authored must resolve to a prim
         INSIDE the composed stage — a binding whose target fell outside the referenced subtree
         (the sibling-materials failure) is a dangling binding and fails here.
    """
    root = stage.GetPrimAtPath(prim_path)
    where = f"referenced '{prim_path}' from '{usd_path}' (source '{source_path}')"

    has_geometry = False
    dangling: list[str] = []
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdGeom.Gprim):
            if prim.IsA(UsdGeom.Mesh):
                # EarliestTime so time-sampled-only points (no default opinion) still count.
                pts = UsdGeom.Mesh(prim).GetPointsAttr().Get(Usd.TimeCode.EarliestTime())
                if pts is not None and len(pts) > 0:
                    has_geometry = True
            else:
                has_geometry = True  # analytic gprim (Cube/Sphere/...) — shape is in attributes
        # A bound material whose target fell outside the referenced subtree is dangling. USD DROPS
        # such targets during composition (they never surface as invalid paths), so the signature is
        # "authored targets, zero composed targets"; a composed target resolving to no prim is the
        # same failure spelled differently, checked too.
        binding_rel = UsdShade.MaterialBindingAPI(prim).GetDirectBindingRel()
        if binding_rel:
            targets = binding_rel.GetTargets()
            if targets:
                dangling.extend(f"{prim.GetPath()} -> {t}" for t in targets if not stage.GetPrimAtPath(t).IsValid())
            elif binding_rel.HasAuthoredTargets():
                dangling.append(f"{prim.GetPath()} -> (target dropped by composition)")

    if not has_geometry:
        raise RuntimeError(
            f"Asset {where} composed NO renderable geometry; the source prim is likely not the asset "
            f"enclosure (e.g. a bare geometry/physics leaf). Set a defaultPrim on the USD pointing at "
            f"the Xform/Scope that encloses the geometry + materials."
        )
    if dangling:
        raise RuntimeError(
            f"Asset {where} has material bindings that resolve OUTSIDE the referenced subtree "
            f"(dangling): {dangling}. The materials live outside the source prim; reference an "
            f"ancestor that encloses both geometry and materials, or set a defaultPrim."
        )


@clone
def spawn_from_usd(
    prim_path: str,
    cfg: from_files_cfg.CustomUsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Spawn an asset from a USD file and override settings with the given config.

    In the case of a USD file, the asset is spawned at the default prim specified in the USD file.
    If a default prim is not specified, then the asset is spawned at the root prim.

    In case a prim already exists at the given prim path, then the function does not create a new prim
    or throw an error that the prim already exists. Instead, it just takes the existing prim and overrides
    the settings with the given config.

    .. note::
        This function is decorated with :func:`clone` that resolves prim path into list of paths
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.

    Parameters
    ----------
    prim_path : str
        The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
        then the asset is spawned at all the matching prim paths.
    cfg : from_files_cfg.CustomUsdFileCfg
        The configuration instance.
    translation : tuple[float, float, float], optional
        The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which
        case the translation specified in the USD file is used.
    orientation : tuple[float, float, float, float], optional
        The orientation in (w, x, y, z) to apply to the prim w.r.t. its parent prim. Defaults to None,
        in which case the orientation specified in the USD file is used.

    Returns
    -------
    Usd.Prim
        The prim of the spawned asset.

    Raises
    ------
    FileNotFoundError
        If the USD file does not exist at the given path.
    """
    return _spawn_from_usd_file(prim_path, cfg.usd_path, cfg, cfg.source_path, translation, orientation)


def _spawn_from_usd_file(
    prim_path: str,
    usd_path: str,
    cfg: from_files_cfg.CustomUsdFileCfg,
    source_path: str | None = None,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Spawn an asset from a USD file and override settings with the given config.

    In case a prim already exists at the given prim path, then the function does not create a new prim
    or throw an error that the prim already exists. Instead, it just takes the existing prim and overrides
    the settings with the given config.

    Parameters
    ----------
    prim_path : str
        The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
        then the asset is spawned at all the matching prim paths.
    usd_path : str
        The path to the USD file to spawn the asset from.
    cfg : from_files_cfg.CustomUsdFileCfg
        The configuration instance.
    source_path : str, optional
        The prim path within the USD file to load. Defaults to "/" (root).
    translation : tuple[float, float, float], optional
        The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which
        case the translation specified in the generated USD file is used.
    orientation : tuple[float, float, float, float], optional
        The orientation in (w, x, y, z) to apply to the prim w.r.t. its parent prim. Defaults to None,
        in which case the orientation specified in the generated USD file is used.

    Returns
    -------
    Usd.Prim
        The prim of the spawned asset.

    Raises
    ------
    FileNotFoundError
        If the USD file does not exist at the given path.
    """
    # check file path exists
    stage: Usd.Stage = stage_utils.get_current_stage()
    if not stage.ResolveIdentifierToEditTarget(usd_path):
        raise FileNotFoundError(f"USD file not found at path: '{usd_path}'.")
    # spawn asset if it doesn't exist.
    if not prim_utils.is_prim_path_valid(prim_path):
        # add prim as reference to stage, replaces built-in create_prim to support source_path
        prim = create_prim(
            prim_path,
            usd_path=usd_path,
            source_path=source_path,
            translation=translation,
            orientation=orientation,
            scale=cfg.scale,
        )
    else:
        omni.log.warn(f"A prim already exists at prim path: '{prim_path}'.")
        prim = prim_utils.get_prim_at_path(prim_path)

    # modify variants
    if hasattr(cfg, "variants") and cfg.variants is not None:
        select_usd_variants(prim_path, cfg.variants)

    # Optionally disable instanceable before modifying physics properties to avoid instancing issues
    if cfg.disable_instanceable:
        set_instanceable(stage, prim_path, False)

    # modify rigid body properties
    if cfg.rigid_props is not None:
        # IsaacLab's RigidObject/RigidObjectCollection require EXACTLY ONE prim with RigidBodyAPI
        # applied in the prim subtree (it counts by HasAPI, ignoring rigidBodyEnabled). Normalize a
        # multi-rigid-body asset to a single compound body BEFORE stamping props, so the props land
        # on that one body. No-op for the common single-body asset.
        _collapse_to_single_rigid_body(stage, prim_path)
        ensure_api_and_modify(
            prim_path, cfg.rigid_props, UsdPhysics.RigidBodyAPI, schemas.modify_rigid_body_properties, stage
        )

    # modify collision properties
    if cfg.collision_props is not None:
        ensure_api_and_modify(
            prim_path, cfg.collision_props, UsdPhysics.CollisionAPI, schemas.modify_collision_properties, stage
        )

    # modify mass properties
    if cfg.mass_props is not None:
        ensure_api_and_modify(prim_path, cfg.mass_props, UsdPhysics.MassAPI, schemas.modify_mass_properties, stage)

    # modify articulation root properties
    if cfg.articulation_props is not None:
        schemas.modify_articulation_root_properties(prim_path, cfg.articulation_props)
    # modify tendon properties
    if cfg.fixed_tendons_props is not None:
        schemas.modify_fixed_tendon_properties(prim_path, cfg.fixed_tendons_props)
    # define drive API on the joints
    # note: these are only for setting low-level simulation properties. all others should be set or are
    #  and overridden by the articulation/actuator properties.
    if cfg.joint_drive_props is not None:
        schemas.modify_joint_drive_properties(prim_path, cfg.joint_drive_props)

    # modify deformable body properties
    if cfg.deformable_props is not None:
        schemas.modify_deformable_body_properties(prim_path, cfg.deformable_props)

    # apply visual material
    if cfg.visual_material is not None:
        if not cfg.visual_material_path.startswith("/"):
            material_path = f"{prim_path}/{cfg.visual_material_path}"
        else:
            material_path = cfg.visual_material_path
        # create material
        cfg.visual_material.func(material_path, cfg.visual_material)
        # apply material
        bind_visual_material(prim_path, material_path)

    # apply physics material — spawn it under the prim and bind to its collider(s) here, at spawn,
    # so PhysX parses the friction/restitution when it first initializes the body (the same point
    # mass/collision props are applied). bind_physics_material is decorated with apply_nested, so a
    # single call on prim_path binds every descendant collider.
    if cfg.physics_material is not None:
        physics_material_path = f"{prim_path}/physicsMaterial"
        cfg.physics_material.func(physics_material_path, cfg.physics_material)
        bind_physics_material(prim_path, physics_material_path)

    # Fail loud if the reference composed no geometry / has dangling material bindings, instead of
    # spawning a silently-invisible object (see _assert_asset_composed). Default-on; opt out via the
    # cfg flag for a legitimately geometry-less or escape-hatched asset.
    if cfg.validate_composition:
        _assert_asset_composed(stage, prim_path, usd_path, source_path)

    # return the prim
    return prim_utils.get_prim_at_path(prim_path)
