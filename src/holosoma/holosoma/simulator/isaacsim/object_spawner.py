"""Unified IsaacSim rigid-object spawner (USD + URDF, 1->1 + 1->N).

Both USD and URDF rigid objects use the same spawn/configuration pipeline. The only
*irreducible* difference between the formats is how the asset reaches USD: a URDF is converted
by the URDF importer (which authors mass/collision/rigid-body physics during conversion and
takes converter-only knobs like ``fix_base``/``joint_drive``), while a USD is referenced
directly. Everything that is NOT converter-specific — the rigid-body props, mass props,
collision props, and contact-sensor activation derived from ``PhysicsConfig`` — is built ONCE
in :func:`_shared_spawn_kwargs` and applied to BOTH formats, so the same ``PhysicsConfig``
produces the same physical body regardless of which format the tri-format selector picks.

- :func:`select_spawn_cfg` — the format strategy: shared props + the per-format extras.
- :func:`build_rigid_object` — the shared tail every object (USD/URDF, standalone or
  scene-file body, free or kinematic) flows through.
- :func:`expand_scene_file` — USD-only 1->N: expand a multi-body USD file into one
  actor per rigid-body prim, reusing :func:`build_rigid_object`.

PhysX keeps its existing pose-settable kinematic representation for ``fixed``.
Newton uses an actual zero-DOF FixedJoint to world and a separate actor handle:
the SDK's articulation view cannot bind standalone fixed bodies. Authored
kinematic scene bodies retain their free joints unless ``fixed=True`` overrides
their motion. Neither representation changes the source asset.

Friction/restitution (``physics.isaacsim``) is carried as a ``physics_material`` on the
``CustomUsdFileCfg`` (see :func:`physics_material_cfg`); the custom USD spawner spawns the
material and binds it to the body's collider at spawn time, so PhysX applies it when the body is
first initialized — the same point mass/collision props are applied. The URDF importer authors no
physics material, so the URDF branch reaches USD via :func:`_convert_urdf_to_usd` and takes the
same path, keeping the two formats physically identical. When ``physics.isaacsim`` is unset, no
material is attached and the asset keeps its authored material.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from functools import partial
from typing import TYPE_CHECKING, Any

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from loguru import logger
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from holosoma.config_types.scene import PhysicsConfig, RigidObjectConfig, SceneFileConfig
from holosoma.simulator.isaacsim.converters import (
    physics_to_collision_props,
    physics_to_mass_props,
    physics_to_rigid_body_props,
)
from holosoma.simulator.isaacsim.prim_naming import distinguishing_names
from holosoma.simulator.isaacsim.prim_utils import compute_world_transform, get_pose, is_rigid_body_enabled
from holosoma.simulator.isaacsim.spawners.from_files_cfg import CustomUsdFileCfg
from holosoma.simulator.isaacsim.spawners.from_files import spawn_from_usd
from holosoma.simulator.shared.asset_format import select_asset_format
from holosoma.utils.path import resolve_asset_path

if TYPE_CHECKING:
    from isaaclab.sim.spawners import SpawnerCfg
    from holosoma.simulator.isaacsim.backends.base import IsaacSimBackendAdapter


def _shared_spawn_kwargs(physics_cfg: PhysicsConfig, fixed: bool) -> dict[str, Any]:
    """Format-independent spawn-cfg fields derived from ``PhysicsConfig``.

    These live on the common ``RigidObjectSpawnerCfg`` base of both ``UrdfFileCfg`` and
    ``UsdFileCfg`` (``rigid_props``/``mass_props``/``collision_props``/
    ``activate_contact_sensors``), so applying them to BOTH formats is what keeps a URDF and a
    USD built from the same ``PhysicsConfig`` physically identical. ``mass_props`` is ``None``
    when no mass/density override is set, which IsaacLab treats as "leave the asset's authored
    mass alone" — so a URDF without an override still uses its ``<inertial>``, a USD its
    authored mass, but an explicit ``physics.mass`` now lands on EITHER format. ``fixed`` is
    threaded into ``rigid_props.kinematic_enabled`` (the static realization), kept off
    ``PhysicsConfig`` so ``RigidObjectConfig.fixed`` stays the single source of truth.
    """
    return dict(
        rigid_props=physics_to_rigid_body_props(physics_cfg, fixed),
        mass_props=physics_to_mass_props(physics_cfg),
        collision_props=physics_to_collision_props(physics_cfg),
        activate_contact_sensors=True,
    )


def select_spawn_cfg(
    fmt: str,
    resolved_path: str,
    *,
    fixed: bool,
    physics: PhysicsConfig | None,
    source_prim_path: str | None,
    backend: IsaacSimBackendAdapter,
    kinematic: bool = False,
) -> "SpawnerCfg":
    """Build the format-specific ``spawn`` cfg.

    Parameters
    ----------
    fmt : str
        Selected asset format, ``"usd"`` or ``"urdf"`` (from ``select_asset_format``, which has
        already validated it against the backend's supported formats).
    resolved_path : str
        Absolute, already-resolved asset path.
    fixed : bool
        World-fixed if True; otherwise dynamic, unless ``kinematic`` preserves
        an authored pose-driven Newton body.
    physics : PhysicsConfig or None
        Physics overrides. ``fixed`` (the static realization) is threaded separately into the
        rigid-props builder, not stored on ``PhysicsConfig``.
    source_prim_path : str or None
        USD only — a specific prim path inside the file for a 1->N body, or None for a
        whole-file standalone object. Ignored for URDF (no sub-prim concept).
    """
    # `fixed` is the single source of truth for static-vs-free; thread it into the rigid-props
    # builder (-> kinematic_enabled) for both formats rather than storing it on PhysicsConfig.
    physics_cfg = physics or PhysicsConfig()
    shared = _shared_spawn_kwargs(physics_cfg, fixed or kinematic)
    physics_material = physics_material_cfg(physics_cfg)

    if fmt == "urdf":
        # URDF adds converter-only knobs on top of the shared props: the URDF->USD importer
        # authors the geometry/joints, fix_base MUST stay False (see module docstring), and the
        # articulation/joint-drive cfgs are converter inputs with no USD analogue.
        urdf_cfg = sim_utils.UrdfFileCfg(
            asset_path=resolved_path,
            # Convert the free asset; the spawn overlay supplies its world attachment.
            fix_base=False,
            replace_cylinders_with_capsules=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
            ),
            **shared,
        )
        if physics_material is None and not (fixed and backend.uses_newton):
            # No friction/restitution to apply — spawn the URDF directly (the converter authors
            # the geometry/joints; the shared rigid/mass/collision props land at spawn).
            return _native_spawn(urdf_cfg, physics_cfg, backend)
        # Friction/restitution IS set. The URDF importer authors no physics material, so reach USD
        # via the same converter and let the USD spawner bind the material at spawn (below).
        resolved_path = _convert_urdf_to_usd(urdf_cfg)
        source_prim_path = None

    # USD (or a URDF now converted to USD). Reference the file (or a sub-prim) and stamp the shared
    # rigid/mass/collision props plus the physics material at spawn, so PhysX applies friction when
    # it first initializes the body (no post-init writes). disable_instanceable=True is required:
    # physics edits on descendants (collapse, RemoveAPI, stamp) no-op on instance proxies if the
    # asset is instanceable.
    cfg = CustomUsdFileCfg(
        func=partial(spawn_from_usd, backend=backend),
        usd_path=resolved_path,
        source_path=source_prim_path,
        physics_material=physics_material,
        disable_instanceable=True,
        **shared,
    )
    cfg = _native_spawn(cfg, physics_cfg, backend)
    if fixed and backend.uses_newton:
        from .backends.fixed_object import spawn_fixed

        cfg.func = partial(spawn_fixed, original=cfg.func)
    return cfg


def _native_spawn(cfg: Any, physics: PhysicsConfig, backend: IsaacSimBackendAdapter) -> Any:
    if backend.is_isaacsim6:
        from holosoma.simulator.isaacsim.backends.native_properties import physics_settings, wrap_spawn

        return wrap_spawn(
            cfg,
            physics_settings(physics, backend),
            backend=backend,
            material_defaults=physics_material_cfg(physics),
        )
    return cfg


def physics_material_cfg(physics_cfg: PhysicsConfig) -> "sim_utils.RigidBodyMaterialCfg | None":
    """The IsaacSim friction/restitution material for ``physics_cfg``, or None if unset.

    Maps the ``isaacsim`` sub-config to a ``RigidBodyMaterialCfg`` the USD spawner binds to the
    body's collider at spawn time. ``None`` leaves the asset's authored material untouched.
    """
    mat = physics_cfg.isaacsim
    if mat is None:
        return None
    return sim_utils.RigidBodyMaterialCfg(
        static_friction=mat.static_friction,
        dynamic_friction=mat.dynamic_friction,
        restitution=mat.restitution,
        friction_combine_mode=mat.friction_combine_mode,
        restitution_combine_mode=mat.restitution_combine_mode,
    )


def _convert_urdf_to_usd(urdf_cfg: "sim_utils.UrdfFileCfg") -> str:
    """Run the IsaacLab URDF->USD converter and return the generated USD path (URDF material path).

    Used ONLY when ``physics.isaacsim`` (friction/restitution) is set on a URDF object: the URDF
    importer authors no physics material, so we must reach USD ourselves to bake+bind the material
    the same way the USD branch does. ``UrdfConverter`` does the exact conversion the spawner would
    have done internally (same cfg, so same geometry/joints/fix_base) — we just intercept its USD
    output instead of letting the spawner reference it directly. Raises loud on converter failure.
    """
    from isaaclab.sim import converters

    loader = converters.UrdfConverter(urdf_cfg)
    usd_path: str = loader.usd_path
    if not usd_path or not os.path.exists(usd_path):
        raise RuntimeError(f"URDF->USD conversion produced no USD for '{urdf_cfg.asset_path}'.")
    return usd_path


def _default_prim_path(stage: "Usd.Stage", usd_path: str) -> str:
    """The stage's default prim path for whole-file authoring (never the pseudo-root ``/``).

    A USD's ``defaultPrim`` is the body the file represents; the pseudo-root ``/`` cannot carry
    the ``instanceable``/physics metadata the material author writes. Fail loud if the asset
    declares no default prim — better than silently mis-authoring onto ``/``.
    """
    default_prim = stage.GetDefaultPrim()
    if default_prim and default_prim.IsValid():
        return str(default_prim.GetPath())
    raise RuntimeError(
        f"USD '{usd_path}' has no defaultPrim, so the physics-material target prim is ambiguous. "
        f"Set a defaultPrim on the asset, or pass an explicit source prim path."
    )


def _assert_asset_exists(resolved_path: str, what: str) -> None:
    """Fail loud and early if a resolved local asset path does not exist.

    ``resolve_asset_path`` returns a path without checking existence, so without this a
    missing file surfaces as an opaque IsaacLab/USD-stage error deep in spawn.
    """
    if not os.path.exists(resolved_path):
        raise ValueError(f"{what} asset not found at path: '{resolved_path}'.")


def _structural_static(prim: "Usd.Prim") -> bool:
    """The file-structural static default for a body prim (overridable by ``resolve_fixed``).

    Recognizes kinematic/disabled rigid bodies and joints attaching a body to
    world. The rigid-body API can sit on a descendant; the world joint can be
    a sibling. Authored kinematic motion is preserved separately by the caller.
    """
    for descendant in Usd.PrimRange(prim):
        if not descendant.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        body = UsdPhysics.RigidBodyAPI(descendant)
        if body.GetKinematicEnabledAttr().Get() or not body.GetRigidBodyEnabledAttr().Get():
            return True
    return _world_attachment(prim)


def _world_attachment(prim: "Usd.Prim") -> bool:
    """Whether an authored fixed joint anchors this body/subtree to world."""
    # A fixed-to-world joint can be a sibling of the body it anchors.
    path = prim.GetPath()
    for joint_prim in prim.GetStage().Traverse():
        if joint_prim.IsA(UsdPhysics.FixedJoint):
            joint = UsdPhysics.FixedJoint(joint_prim)
            first, second = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
            if (not first and any(target.HasPrefix(path) for target in second)) or (
                not second and any(target.HasPrefix(path) for target in first)
            ):
                return True
    return False


def _structural_kinematic(prim: "Usd.Prim") -> bool:
    """Whether the source explicitly authors a pose-driven, enabled rigid body."""
    return any(
        p.HasAPI(UsdPhysics.RigidBodyAPI)
        and UsdPhysics.RigidBodyAPI(p).GetRigidBodyEnabledAttr().Get()
        and UsdPhysics.RigidBodyAPI(p).GetKinematicEnabledAttr().Get()
        for p in Usd.PrimRange(prim)
    ) and not _world_attachment(prim)


def _enabled_rigid_body_prims(stage: "Usd.Stage") -> list["Usd.Prim"]:
    """Every prim in the stage that is an enabled rigid body."""
    return [p for p in stage.Traverse() if is_rigid_body_enabled(p)]


def _referenceable_ancestor(stage: "Usd.Stage", prim_path: str) -> str:
    """Nearest ancestor of ``prim_path`` carrying an authored reference or payload arc.

    A rigid-body prim may live deep inside a payload; that path is only reachable through composition
    and is not a valid ``AddReference`` ``primPath``. The referenceable unit is the ancestor carrying
    the arc. Returns ``prim_path`` unchanged if none is found.
    """
    path = Sdf.Path(prim_path)
    while not path.IsRootPrimPath():
        prim = stage.GetPrimAtPath(str(path))
        if prim.IsValid():
            for spec in prim.GetPrimStack():
                # Any authored arc form counts: a plain `references = @..@` authors explicitItems;
                # prepend/append author their own lists.
                ref, pay = spec.referenceList, spec.payloadList
                arcs = (
                    ref.prependedItems,
                    ref.explicitItems,
                    ref.appendedItems,
                    pay.prependedItems,
                    pay.explicitItems,
                    pay.appendedItems,
                )
                if any(len(a) for a in arcs):
                    return str(path)
        path = path.GetParentPath()
    return prim_path


def _geometry_owner_prims(stage: "Usd.Stage") -> list["Usd.Prim"]:
    """Prims that DIRECTLY wrap renderable/collidable geometry (have a direct ``UsdGeom.Gprim`` child).

    This is the candidate-body universe for an UNAUTHORED scene file under EXPLICIT
    include/exclude patterns (the opt-in multi-body path). The parent of a geom (e.g. an Xform
    ``leg_1`` holding a ``Cube``) IS a body; the geom leaf itself is not (it has no Gprim child);
    and a pure grouping prim whose children are all Xforms (e.g. a ``desk`` aggregating leg
    subassemblies) is not — so include/exclude match on body-level names, never geom leaves or
    bare groups. We deliberately do NOT try to infer a body split from this set on our own — that
    decomposition is exactly what authoring ``RigidBodyAPI`` (or a URDF/MJCF body tree) provides;
    here the user has named the granularity explicitly via the patterns.
    """
    return [p for p in stage.Traverse() if any(child.IsA(UsdGeom.Gprim) for child in p.GetChildren())]


def _discover_scene_body_prim_paths(
    stage: "Usd.Stage",
    scene_file_name: str,
    scene_file: SceneFileConfig,
    usd_path: str,
    *,
    include_static: bool = False,
) -> list[str]:
    """Decide which prims of a scene file become rigid bodies (the 1->N decomposition).

    Three cases (the include/exclude SUBSET filter is applied by the caller, AFTER this — this
    function only fixes the candidate universe):

    - AUTHORED — rigid-body prims declare the file's decomposition. Newton
      includes disabled rigid bodies, which describe static collision geometry;
      other backends retain the enabled-body selection.
    - UNAUTHORED + DEFAULT patterns (``include=['*']``, no excludes): the file expresses no body
      split AND the user requested no subset, so collapse to ONE body on the ``defaultPrim`` (the
      USD's "what this file is"), warning that nothing was authored. Geometry is NOT inspected to
      guess a multi-body split — a ``desk`` whose legs are child geoms loads as ONE rigid desk.
    - UNAUTHORED + EXPLICIT patterns: the user has opted into a multi-body split, so the candidate
      universe is the geometry-owning prims; the caller's include/exclude then selects among them.
      ``RigidBodyAPI`` is stamped on each selected prim at spawn (``ensure_api_and_modify``).

    Returns prim paths (unsorted; not yet include/exclude-filtered).
    """
    authored = (
        [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        if include_static
        else _enabled_rigid_body_prims(stage)
    )
    if authored:
        # Rigid-body prims may live inside payloads; remap each to its referenceable ancestor
        # (see _referenceable_ancestor) and deduplicate so multiple leaves under one ancestor
        # (e.g. Actor_N/Geom/body_a and body_b -> Actor_N) collapse to a single entry.
        return list(dict.fromkeys(_referenceable_ancestor(stage, str(p.GetPath())) for p in authored))

    patterns_are_default = scene_file.include_patterns == ["*"] and not scene_file.exclude_patterns
    if patterns_are_default:
        default_prim = _default_prim_path(stage, usd_path)
        logger.warning(
            f"Scene file '{scene_file_name}' ({usd_path}) authors no enabled RigidBodyAPI prim; "
            f"loading the WHOLE file as ONE rigid body '{default_prim}'. To split it into multiple "
            f"bodies, author RigidBodyAPI on each body (or load the URDF/MJCF form), or set "
            f"include_patterns to the body prim paths you want promoted."
        )
        return [default_prim]

    owners = _geometry_owner_prims(stage)
    if not owners:
        raise ValueError(
            f"Scene file '{scene_file_name}' ({usd_path}) authors no rigid bodies and has no "
            f"geometry-owning prims to promote under include_patterns={scene_file.include_patterns}; "
            f"nothing to load. Author RigidBodyAPI on the bodies, or load the URDF/MJCF form."
        )
    return [str(p.GetPath()) for p in owners]


def resolve_asset_root(usd_path: str, obj_name: str) -> str | None:
    """Pick the prim path a 1->1 USD ``RigidObjectConfig`` references: the asset enclosure root.

    A USD reference composes only the subtree rooted at the chosen prim, re-rooted onto the spawned
    prim: sibling prims and ancestor opinions are not pulled in, and any relationship (notably
    ``material:binding``) whose target falls outside the subtree breaks. The reference target must
    enclose the asset's geometry and its materials, so the asset root is referenced rather than an
    interior physics-bearing prim. Physics placement is irrelevant to enclosure: the spawn tail
    (``_collapse_to_single_rigid_body`` and ``ensure_api_and_modify``, stamping at the root)
    normalizes physics wherever the asset authored it. A rigid body on a Mesh leaf, in a payload, or
    under a Scope all compose correctly without per-shape special-casing.

    Resolution order:
      1. ``defaultPrim`` if set: the body the file represents.
      2. else the unique active top-level prim under the pseudo-root.
      3. else (no defaultPrim and zero or many top-level prims): fail with candidates, since the
         asset root is ambiguous; the author must set a defaultPrim on the USD.
    """
    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise ValueError(f"Rigid object '{obj_name}' USD could not be opened: '{usd_path}'.")
    default_prim = stage.GetDefaultPrim()
    if default_prim and default_prim.IsValid():
        return str(default_prim.GetPath())
    tops = [p for p in stage.GetPseudoRoot().GetChildren() if p.IsActive()]
    if len(tops) == 1:
        return str(tops[0].GetPath())
    raise ValueError(
        f"Rigid object '{obj_name}' ({usd_path}) has no defaultPrim and "
        f"{len(tops)} active top-level prims {sorted(str(p.GetPath()) for p in tops)}; the asset "
        f"root is ambiguous. Set a defaultPrim on the USD."
    )


def build_rigid_object(
    name: str,
    fmt: str,
    resolved_path: str,
    pos: Sequence[float],
    rot_wxyz: Sequence[float],
    *,
    fixed: bool,
    physics: PhysicsConfig | None = None,
    source_prim_path: str | None = None,
    backend: IsaacSimBackendAdapter,
    kinematic: bool = False,
) -> Any:
    """Build the actor handle both formats and both shapes flow through.

    The leaf of ``prim_path`` MUST equal ``name`` (the ObjectRegistry key) so per-env
    placement and state read/write stay aligned. ``rot_wxyz`` uses the Holosoma config convention.
    """
    _assert_asset_exists(resolved_path, f"Object '{name}' ({fmt})")
    spawn = select_spawn_cfg(
        fmt,
        resolved_path,
        fixed=fixed,
        physics=physics,
        source_prim_path=source_prim_path,
        backend=backend,
        kinematic=kinematic,
    )
    cfg = RigidObjectCfg(
        prim_path=f"{backend.env_regex_ns}/{name}",
        spawn=spawn,
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(pos), rot=backend.config_quat_to_native(rot_wxyz)),
    )
    if fixed and backend.uses_newton:
        from .backends.fixed_object import FixedObject

        cfg.class_type = FixedObject
        return FixedObject(cfg)
    return RigidObject(cfg)


def build_standalone_rigid_object(
    name: str,
    obj: RigidObjectConfig,
    supported_formats: list[str],
    scene_asset_root: str | None,
    *,
    backend: IsaacSimBackendAdapter,
) -> RigidObject:
    """Spawn one standalone ``RigidObjectConfig`` (1->1), USD or URDF, via the shared tail.

    For USD, the referenced prim is the asset enclosure root chosen by :func:`resolve_asset_root`
    (defaultPrim or unique top-level prim). Referencing the enclosure rather than an interior
    physics prim composes geometry and materials together; the spawn tail stamps a single rigid
    body at the root. URDF has no sub-prim concept, so it always references the whole converted
    asset.
    """
    fmt, asset_rel = select_asset_format(obj, supported_formats)
    resolved = resolve_asset_path(asset_rel, obj.asset_root or scene_asset_root)
    source_prim_path = resolve_asset_root(resolved, name) if fmt == "usd" else None
    return build_rigid_object(
        name,
        fmt,
        resolved,
        obj.position,
        obj.orientation,
        fixed=obj.fixed,
        physics=obj.physics,
        source_prim_path=source_prim_path,
        backend=backend,
    )


def expand_scene_file(
    scene_file_name: str,
    scene_file: SceneFileConfig,
    supported_formats: list[str],
    scene_asset_root: str | None,
    *,
    backend: IsaacSimBackendAdapter,
) -> tuple[dict[str, RigidObject], set[str]]:
    """Expand a multi-body USD scene file into per-body ``RigidObject`` instances (1->N).

    Each body prim becomes its own ``RigidObject`` named ``{scene_file_name}_{body}``, reusing
    :func:`build_rigid_object` so a scene-file body and a standalone object are structurally
    identical. Bodies load in name-sorted order for backend-independent registration.

    Which prims are bodies (the 1->N decomposition) is decided by
    :func:`_discover_scene_body_prim_paths`: an authored file uses its enabled ``RigidBodyAPI``
    prims; an unauthored file collapses to one body (defaultPrim) unless include_patterns opt
    into a multi-body split. Each body's ``body`` name is the distinguishing tail of its prim
    path (:func:`~holosoma.simulator.isaacsim.prim_naming.distinguishing_names`) — the bare leaf
    for a flat file, ``parent_leaf`` where leaves collide across parents.

    Three shared, per-body rules (identical on every backend) are then applied, all keyed on the
    ``body`` name:
    - ``scene_file.should_include`` — include/exclude SUBSET filter (which bodies load);
    - ``scene_file.resolve_fixed`` — free vs static, defaulting to the prim's authored
      ``physics:kinematicEnabled``;
    - ``scene_file.resolve_physics`` — per-body mass/damping/material override.

    Each body's world pose is the file's configured world pose composed with the prim's authored
    local transform, preserving inter-body relative poses.

    IsaacSim 1->N is USD-only: a converted multi-body URDF is one articulation, not N
    rigid-body siblings, so a non-USD scene file fails loud.

    Returns ``(objects_by_name, static_names)``.
    """
    fmt, asset_rel = select_asset_format(scene_file, supported_formats)
    if fmt != "usd":
        raise ValueError(
            f"IsaacSim scene_files support only USD 1->N expansion (got '{fmt}' for "
            f"scene file '{scene_file_name}'). Provide a usd_path."
        )
    usd_path = resolve_asset_path(asset_rel, scene_file.asset_root or scene_asset_root)
    _assert_asset_exists(usd_path, f"Scene file '{scene_file_name}'")
    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise ValueError(f"Scene file '{scene_file_name}' USD could not be opened: '{usd_path}'.")

    # File world transform (outer): place the whole file at its configured pose.
    fx, fy, fz = scene_file.position
    fw, fxq, fyq, fzq = scene_file.orientation  # wxyz
    file_tf = Gf.Transform()
    file_tf.SetTranslation(Gf.Vec3d(fx, fy, fz))
    file_tf.SetRotation(Gf.Rotation(Gf.Quatd(fw, fxq, fyq, fzq)))
    file_mtx = file_tf.GetMatrix()

    # Decide which prims are bodies, then derive a collision-free `body` name per prim (leaf for a
    # flat file; `parent_leaf` where leaves collide across parents — see distinguishing_names).
    body_prim_paths = _discover_scene_body_prim_paths(
        stage,
        scene_file_name,
        scene_file,
        usd_path,
        include_static=backend.uses_newton,
    )
    name_by_path = distinguishing_names(body_prim_paths)

    # Apply the include/exclude SUBSET filter (the only place membership is decided), keyed on the
    # body name exactly like every other backend; then sort by body name for stable order.
    selected = sorted(
        (path for path, body in name_by_path.items() if scene_file.should_include(body)),
        key=lambda path: name_by_path[path],
    )
    if not selected:
        raise ValueError(
            f"Scene file '{scene_file_name}' ({usd_path}): include/exclude patterns selected no "
            f"bodies (include={scene_file.include_patterns}, exclude={scene_file.exclude_patterns}; "
            f"candidates={sorted(name_by_path.values())})."
        )

    # RigidBodyAPI cannot nest (PhysX rejects a rigid body inside another). Authored files don't
    # nest by construction, but an unauthored file selected by include_patterns could pick an
    # ancestor and its descendant (e.g. a `desk` with a direct geom AND its `leg_1`). Catch that
    # loudly here rather than silently spawning an invalid nested-body hierarchy.
    for outer in selected:
        for inner in selected:
            if inner != outer and inner.startswith(outer + "/"):
                raise ValueError(
                    f"Scene file '{scene_file_name}' ({usd_path}): selected bodies nest — "
                    f"'{name_by_path[inner]}' ({inner}) is inside '{name_by_path[outer]}' ({outer}). "
                    f"A rigid body cannot contain another; narrow include_patterns/exclude_patterns "
                    f"to one level, or author RigidBodyAPI only on the intended bodies."
                )

    objects: dict[str, RigidObject] = {}
    static_names: set[str] = set()
    for prim_path in selected:
        body = name_by_path[prim_path]
        actor_name = f"{scene_file_name}_{body}"
        prim = stage.GetPrimAtPath(prim_path)

        structural_static = _structural_static(prim)
        if backend.uses_newton:
            # USD collider-only geometry is native static geometry.
            structural_static |= not any(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in Usd.PrimRange(prim)) and any(
                p.HasAPI(UsdPhysics.CollisionAPI) for p in Usd.PrimRange(prim)
            )
        is_static = scene_file.resolve_fixed(body, structural_default=structural_static)
        # An authored kinematic body is independently pose-driven, not welded.
        # Resolving both defaults distinguishes an explicit fixed override from
        # an unchanged structural default without another configuration option.
        kinematic = (
            backend.uses_newton
            and _structural_kinematic(prim)
            and not scene_file.resolve_fixed(body, structural_default=False)
            and scene_file.resolve_fixed(body, structural_default=True)
        )

        # Per-body physics override (mass/damping/material) from object_configs, applied the
        # same way every backend does. None => the body keeps the file's authored physics.
        # Motion classification is independent of the per-body physics overrides.
        body_physics = scene_file.resolve_physics(body)

        # Composed world pose = file world transform . prim local transform.
        local_mtx = compute_world_transform(stage, prim_path)
        composed = Gf.Transform()
        composed.SetMatrix(local_mtx * file_mtx)
        pos, quat_wxyz = get_pose(composed)

        # Per-body world-frame position offset added on top of the composed world placement.
        # None => unchanged.
        offset = scene_file.resolve_position_offset(body)
        px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
        if offset is not None:
            px, py, pz = px + offset[0], py + offset[1], pz + offset[2]

        objects[actor_name] = build_rigid_object(
            actor_name,
            "usd",
            usd_path,
            [px, py, pz],
            list(quat_wxyz),
            fixed=is_static and not kinematic,
            kinematic=kinematic,
            physics=body_physics,
            source_prim_path=prim_path,
            backend=backend,
        )
        if is_static:
            static_names.add(actor_name)

    return objects, static_names
