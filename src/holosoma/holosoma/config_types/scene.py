"""Scene-asset configuration types.

Declarative description of what is spawned into a scene — standalone rigid
objects, scene files (USD/URDF), and their physics.
"""

from __future__ import annotations

import fnmatch
from dataclasses import field

import tyro
from pydantic import ConfigDict, model_validator
from pydantic.dataclasses import dataclass
from typing_extensions import Annotated

# Reject unknown fields on the physics configs so a typo'd or misplaced field (e.g. a damping
# field that belongs on ``physx``) fails loud at construction instead of being ignored.
_FORBID_EXTRA = ConfigDict(extra="forbid")


@dataclass(frozen=True, config=_FORBID_EXTRA)
class PhysXPhysicsConfig:
    """Rigid-body solver knobs that PhysX interprets identically under BOTH Isaac backends.

    IsaacGym and IsaacSim run the same PhysX solver and read these the same way, so they live
    in ONE shared sub-config referenced by both.
    """

    linear_damping: float = 0.1
    """Linear velocity damping coefficient. Defaults to 0.1."""

    angular_damping: float = 0.1
    """Angular velocity damping coefficient. Defaults to 0.1."""

    max_linear_velocity: float = 1000.0
    """Maximum linear velocity limit. Defaults to 1000.0."""

    max_angular_velocity: float = 1000.0
    """Maximum angular velocity limit. Defaults to 1000.0."""


@dataclass(frozen=True, config=_FORBID_EXTRA)
class IsaacGymPhysicsConfig:
    """Rigid-shape material properties for the IsaacGym simulator.

    Provides 1:1 mapping to IsaacGym RigidShapeProperties for physics simulation. Friction is
    IsaacGym-specific (a single sliding coefficient, vs IsaacSim's static/dynamic split).
    """

    friction: float = 1.0
    """Static friction coefficient. Defaults to 1.0."""

    rolling_friction: float = 0.0
    """Rolling resistance coefficient. Defaults to 0.0."""

    torsion_friction: float = 0.0
    """Torsion resistance coefficient. Defaults to 0.0."""

    restitution: float = 0.0
    """Bounce coefficient. Defaults to 0.0."""

    compliance: float = 0.0
    """Shape compliance. Defaults to 0.0."""


@dataclass(frozen=True, config=_FORBID_EXTRA)
class IsaacSimPhysicsConfig:
    """IsaacSim rigid-body physics: material (friction/restitution) and collision offsets.

    Friction/restitution map 1:1 to IsaacLab ``RigidBodyMaterialCfg`` (friction is a static/dynamic
    pair with combine modes, vs IsaacGym's single coefficient). The collision offsets map to
    ``CollisionPropertiesCfg`` (``physxCollision:contactOffset``/``restOffset``). All are applied at
    spawn, so PhysX reads them when the body is first initialized.
    """

    static_friction: float = 1.0
    """Static friction coefficient. Defaults to 1.0."""

    dynamic_friction: float = 1.0
    """Dynamic friction coefficient. Defaults to 1.0."""

    restitution: float = 0.0
    """Bounce coefficient. Defaults to 0.0."""

    friction_combine_mode: str = "multiply"
    """Friction combination mode. Options: "multiply", "max", "min", "average". Defaults to "multiply"."""

    restitution_combine_mode: str = "multiply"
    """Restitution combination mode. Options: "multiply", "max", "min", "average". Defaults to "multiply"."""

    contact_offset: float | None = None
    """Collision contact offset in meters: distance at which PhysX starts generating contacts.
    ``None`` (default) keeps the asset's authored value. Required: >= ``rest_offset``."""

    rest_offset: float | None = None
    """Collision rest offset in meters: surface separation at rest. ``None`` (default) keeps the
    asset's authored value. Required: <= ``contact_offset``."""

    torsional_patch_radius: float | None = None
    """Radius (m) of the contact patch PhysX uses to apply torsional friction: the rotational
    friction resisting twist about the contact normal (a flat object spinning in place on a
    surface). 0 (or ``None``, the default) disables torsional friction; a non-zero radius enables it,
    scaled by the material's friction coefficients. Set it to roughly the contact-patch size."""

    @model_validator(mode="after")
    def _validate_offsets(self) -> IsaacSimPhysicsConfig:
        """PhysX requires contact_offset >= rest_offset; checked only when both are set."""
        if self.contact_offset is not None and self.rest_offset is not None and self.contact_offset < self.rest_offset:
            raise ValueError(
                f"IsaacSimPhysicsConfig.contact_offset ({self.contact_offset}) must be >= rest_offset "
                f"({self.rest_offset}); PhysX rejects contact_offset < rest_offset."
            )
        if self.torsional_patch_radius is not None and self.torsional_patch_radius < 0:
            raise ValueError(
                f"IsaacSimPhysicsConfig.torsional_patch_radius ({self.torsional_patch_radius}) must be >= 0."
            )
        return self


@dataclass(frozen=True, config=_FORBID_EXTRA)
class MujocoPhysicsConfig:
    """MuJoCo geom-level physics, set on an ``MjSpec`` before compile.

    These are geom attributes (friction/solref/solimp/condim) that apply to any body's geoms,
    whether a free scene object or a robot link, so this config is shared across both entity kinds.
    ``None`` on any field keeps the asset's value.
    """

    friction: list[float] | None = None
    """Geom friction ``[sliding, torsional, rolling]``. ``None`` keeps the asset's value."""

    solref: list[float] | None = None
    """Constraint solver reference ``[timeconst, dampratio]``. ``None`` keeps the asset's value."""

    solimp: list[float] | None = None
    """Constraint solver impedance ``[dmin, dmax, width, midpoint, power]``. ``None`` keeps it."""

    condim: int | None = None
    """Contact dimensionality (1/3/4/6). ``None`` keeps the asset's value."""


@dataclass(frozen=True, config=_FORBID_EXTRA)
class PhysicsConfig:
    """Unified physics configuration shared across simulators.

    Only the fields every backend interprets identically — ``mass`` (kg) and ``density``
    (kg/m^3) — live in the core. Everything that a given engine interprets in its own way is
    pushed into a backend-specific sub-config, so a field never silently means different things
    on different backends:

    - :class:`PhysXPhysicsConfig` (``physx``): solver damping / velocity limits, shared by BOTH
      Isaac backends (same PhysX semantics).
    - :class:`IsaacGymPhysicsConfig` (``isaacgym``): IsaacGym rigid-shape friction.
    - :class:`IsaacSimPhysicsConfig` (``isaacsim``): IsaacSim rigid-body material.
    - :class:`MujocoPhysicsConfig` (``mujoco``): MuJoCo geom friction/solver props.

    Friction is not a shared core field. Each engine models it differently (IsaacGym: a single
    ``friction``; IsaacSim: ``static_friction``/``dynamic_friction``; MuJoCo: ``friction[0]`` of a
    3-vector), so it lives in the per-backend sub-configs and there is no cross-check that the three
    agree. For cross-backend friction parity, set all three (``isaacgym.friction``,
    ``isaacsim.static_friction``/``dynamic_friction``, ``mujoco.friction``) to matching values;
    setting only one leaves the other backends at their defaults.
    """

    mass: float | None = None
    """Direct mass override in kg (highest priority). Defaults to None."""

    density: float | None = None
    """Density in kg/m³ for mass calculation (medium priority). Defaults to None."""

    physx: PhysXPhysicsConfig | None = None
    """PhysX solver knobs (damping / velocity limits) shared by IsaacGym + IsaacSim. Defaults to None."""

    isaacgym: IsaacGymPhysicsConfig | None = None
    """IsaacGym-specific physics configuration. Defaults to None."""

    isaacsim: IsaacSimPhysicsConfig | None = None
    """IsaacSim-specific physics configuration. Defaults to None."""

    mujoco: MujocoPhysicsConfig | None = None
    """MuJoCo-specific physics configuration. Defaults to None."""


@dataclass(frozen=True)
class ObjectPatternConfig:
    """Per-object override for bodies in a scene file, matched by name pattern."""

    physics: PhysicsConfig | None = None
    """Physics configuration to apply to matching objects. Defaults to None."""

    fixed: bool | None = None
    """Override whether matching bodies are static. ``None`` (default) keeps the file's
    own structure (free joint / movable base / dynamic body => free; jointless /
    fixed-to-world / kinematic => static); ``True``/``False`` forces static/free."""

    position_offset: list[float] | None = None
    """Per-body world-frame position offset ``[dx, dy, dz]`` added to the body's authored placement.
    ``None`` (default) keeps the authored placement."""


@dataclass(frozen=True)
class SceneFileConfig:
    """A multi-body scene file: ONE file expands to N registered bodies (1->N).

    Keyed in :attr:`SceneConfig.scene_files` by a file-scope namespace; each top-level body
    in the file becomes its own registered actor named ``{key}_{body}`` so two files (or a
    file and a standalone rigid object) never collide. A body is registered free (INDIVIDUAL)
    or static (SCENE) according to the
    FILE's own structure (a free joint / movable base / dynamic rigid body => free;
    jointless / fixed-to-world / kinematic => static), unless an ``object_configs`` entry
    overrides it with ``fixed``. Inter-body relative poses authored in the file are
    preserved; the whole file is placed at ``position``/``orientation`` (world pose).
    Tri-format like RigidObjectConfig.

    Two distinct, complementary controls:

    - WHICH bodies load: ``include_patterns`` / ``exclude_patterns`` (see
      :meth:`should_include`).
    - HOW the loaded bodies behave: ``object_configs`` (see :meth:`resolve_fixed` /
      :meth:`resolve_physics`). A body that matches NO ``object_configs``
      pattern still loads, keeping the file's own structural ``fixed`` and its authored physics.
    """

    # --- asset format (tri-format sibling fields, like RigidObjectConfig) ---
    usd_path: str | None = None
    """Path to USD scene file. Defaults to None."""

    urdf_path: str | None = None
    """Path to URDF scene file. Defaults to None."""

    xml_path: str | None = None
    """Path to MJCF/XML scene file. Defaults to None."""

    position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """Position offset [x, y, z]. Defaults to [0.0, 0.0, 0.0]."""

    orientation: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])  # [w,x,y,z]
    """Orientation quaternion [w, x, y, z]. Defaults to [1.0, 0.0, 0.0, 0.0]."""

    include_patterns: list[str] = field(default_factory=lambda: ["*"])
    """Glob patterns (against bare body names) for WHICH bodies load. Default ``["*"]`` =
    all. A body must match at least one to load (see :meth:`should_include`). For an
    unauthored USD, listing body prim paths here also opts into a multi-body split."""

    exclude_patterns: list[str] = field(default_factory=list)
    """Glob patterns for bodies to DROP; takes precedence over ``include_patterns``.
    Defaults to empty list (drop nothing). See :meth:`should_include`."""

    # Suppress from the tyro CLI: a populated dict-of-dataclasses default DOES expose per-key
    # flags, but these keys are fnmatch globs (e.g. "*bottle_Actor_0001") that don't make usable
    # flag names. object_configs is set in presets, not on the CLI.
    object_configs: Annotated[dict[str, ObjectPatternConfig] | None, tyro.conf.Suppress] = None
    """Per-body OVERRIDES (``fixed`` / ``physics`` / ``position_offset``) by name pattern. Defaults
    to None. Set in scene presets (config_values); not overridable on the CLI."""

    asset_root: str | None = None
    """Root directory for resolving relative paths. Defaults to None."""

    def should_include(self, body_name: str) -> bool:
        """Whether a body of this file should load, per include/exclude patterns.

        The single subset-selection rule, shared by every backend (MuJoCo / IsaacGym /
        IsaacSim) so a scene file's body set is identical regardless of which backend
        loads it. Semantics:

        1. ``exclude_patterns`` wins: a body matching ANY exclude pattern is dropped.
        2. ``include_patterns`` (default ``["*"]`` => everything) then gates the rest: a
           body must match at least one include pattern to load.

        Patterns are ``fnmatch`` globs against the bare body name; a leading ``*/`` is
        stripped (:meth:`_normalize_pattern`, shared with the ``object_configs`` resolvers)
        so file-path-style patterns (``*/free_box``) match the same everywhere.
        """

        def _matches_any(patterns: list[str]) -> bool:
            return any(fnmatch.fnmatch(body_name, self._normalize_pattern(p)) for p in patterns)

        if self.exclude_patterns and _matches_any(self.exclude_patterns):
            return False
        # Empty include list would match nothing; the field defaults to ["*"], and an
        # explicit [] is treated as "no include filter" (load everything not excluded).
        if not self.include_patterns:
            return True
        return _matches_any(self.include_patterns)

    @staticmethod
    def _normalize_pattern(pattern: str) -> str:
        """Strip a leading ``*/`` so file-path-style patterns match the bare body name.

        The single normalization for include/exclude filtering AND ``object_configs`` matching,
        so one pattern string means the same thing in both places.
        """
        return pattern[2:] if pattern.startswith("*/") else pattern

    def resolve_fixed(self, body_name: str, structural_default: bool) -> bool:
        """Whether ``body_name`` is static, applying any per-object config override.

        Defaults to ``structural_default`` (what the file's own structure says — freejoint
        / movable base / dynamic body => free; jointless / fixed / kinematic => static).
        An ``object_configs`` entry whose pattern matches the body and sets ``fixed`` to a
        non-None value overrides it. Shared by every backend so the rule is identical.
        """
        for pattern, obj_config in (self.object_configs or {}).items():
            if obj_config.fixed is not None and fnmatch.fnmatch(body_name, self._normalize_pattern(pattern)):
                return obj_config.fixed
        return structural_default

    def resolve_physics(self, body_name: str) -> PhysicsConfig | None:
        """Per-body physics override from ``object_configs``, or ``None`` if none matches.

        The first ``object_configs`` entry whose pattern matches ``body_name`` and carries a
        non-None ``physics`` wins. Shared by every backend so a scene file's per-body physics
        (mass/damping/material) is resolved identically whether it loads under IsaacGym,
        IsaacSim, or MuJoCo — counterpart to :meth:`resolve_fixed` for the ``fixed`` field.
        """
        for pattern, obj_config in (self.object_configs or {}).items():
            if obj_config.physics is not None and fnmatch.fnmatch(body_name, self._normalize_pattern(pattern)):
                return obj_config.physics
        return None

    def resolve_position_offset(self, body_name: str) -> list[float] | None:
        """Per-body world-frame position offset from ``object_configs``, or ``None`` if none matches.

        First ``object_configs`` entry whose pattern matches ``body_name`` with a non-None
        ``position_offset`` wins; the offset is added to the body's composed world placement. Shared
        by every backend; counterpart to :meth:`resolve_fixed` / :meth:`resolve_physics`.
        """
        for pattern, obj_config in (self.object_configs or {}).items():
            if obj_config.position_offset is not None and fnmatch.fnmatch(body_name, self._normalize_pattern(pattern)):
                return obj_config.position_offset
        return None


@dataclass(frozen=True)
class RigidObjectConfig:
    """Configuration for an individual rigid object (a free body / manipuland).

    Keyed in :attr:`SceneConfig.rigid_objects` by its actor name (the ObjectRegistry key).
    """

    # --- asset format (tri-format sibling fields) ---
    # what format is used depends on the simulator preference
    urdf_file: str | None = None
    usd_file: str | None = None
    xml_file: str | None = None

    asset_root: str | None = None
    """Optional per-asset root; falls back to SceneConfig.asset_root when None."""

    # --- placement & physics ---
    position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """Position [x, y, z] of the object. Defaults to [0.0, 0.0, 0.0]."""

    orientation: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])  # [w,x,y,z]
    """Orientation quaternion [w, x, y, z] of the object. Defaults to [1.0, 0.0, 0.0, 0.0]."""

    fixed: bool = False
    """If True the object is static — attached to the world with no joint, so it holds
    its pose and never moves; registered as ObjectType.SCENE with a read-only pose. If
    False (default) it is a free body (ObjectType.INDIVIDUAL) with a free joint."""

    linear_velocity: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """Initial world-frame linear velocity [vx, vy, vz] of the free body. Defaults to [0, 0, 0].
    A non-zero value on a ``fixed`` object is rejected by ``validate_velocity_requires_free``."""

    angular_velocity: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """Initial angular velocity [wx, wy, wz] of the free body. Defaults to [0, 0, 0].
    A non-zero value on a ``fixed`` object is rejected by ``validate_velocity_requires_free``."""

    physics: PhysicsConfig | None = None
    """Physics configuration for the object. Defaults to None."""

    @model_validator(mode="after")
    def validate_velocity_requires_free(self) -> RigidObjectConfig:
        """A static (``fixed=True``) object is welded to the world and cannot move, so an
        initial velocity on it can never take effect."""
        if self.fixed and (any(self.linear_velocity) or any(self.angular_velocity)):
            raise ValueError(
                f"A fixed=True (static) RigidObjectConfig sets a non-zero "
                f"initial velocity (linear_velocity={self.linear_velocity}, "
                f"angular_velocity={self.angular_velocity}). A static object is welded to the "
                f"world and cannot move; set fixed=False for a moving body, or leave the "
                f"velocities at zero."
            )
        return self


@dataclass(frozen=True)
class SceneConfig:
    """Composition of scene assets for the simulator."""

    replicate_physics: bool = True
    """Whether to reuse physics properties across duplicated assets (IsaacSim's env cloner only;
    the other backends have no equivalent knob and ignore it)."""

    asset_root: str | None = None
    """Optional root directory for relative asset paths."""

    scene_files: dict[str, SceneFileConfig] = field(default_factory=dict)
    """Scene files (USD/URDF) to load, keyed by file-scope namespace; registered bodies
    are ``{key}_{body}``. The key must not collide with any rigid_objects key."""

    rigid_objects: dict[str, RigidObjectConfig] = field(default_factory=dict)
    """Standalone rigid objects to instantiate, keyed by actor name (the ObjectRegistry key)."""

    env_spacing: float = 20.0
    """Distance between parallel environments in the grid layout."""

    @model_validator(mode="after")
    def validate_unique_names(self) -> SceneConfig:
        """Names are ObjectRegistry keys and must be unique across the whole scene.

        Keys are unique within each dict by construction; the remaining hazard is a key
        shared between ``rigid_objects`` and ``scene_files`` (their namespaces are one
        space), which would let a file body (``{file}_{body}``) silently shadow a
        standalone object, so reject overlaps up front (the registry is the final backstop).
        """
        dupes = set(self.rigid_objects) & set(self.scene_files)
        if dupes:
            raise ValueError(
                f"Duplicate names across scene rigid_objects/scene_files: {sorted(dupes)}. "
                f"Each rigid_objects name and each scene_files name must be unique."
            )
        return self
