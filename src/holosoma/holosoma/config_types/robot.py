from __future__ import annotations

from collections import Counter
from dataclasses import field

from pydantic import ConfigDict, model_validator
from pydantic.dataclasses import dataclass

from holosoma.config_types.scene import PhysicsConfig
from holosoma.config_types.value_types import UnitQuaternionWXYZ
from holosoma.utils.name_tags import expand_joint_value_map, validate_tag_table
from holosoma.utils.robot_properties import resolve_dof_properties

# Unknown fields are a configuration error, not something to drop silently: a removed or misspelled
# key would otherwise change behaviour without a trace.
_FORBID_EXTRA = ConfigDict(extra="forbid")


@dataclass(frozen=True, config=_FORBID_EXTRA)
class RobotBridgeConfig:
    """Bridge-specific configuration for robot SDK communication.

    Currently supports sim2sim (holosoma/run_sim.py) only.
    """

    sdk_type: str = "unitree"
    """SDK type for robot communication ('unitree', 'unitree_mp', 'booster', 'ros2').

    'unitree_mp' runs the Unitree SDK's C++/CycloneDDS binding in a spawned child process so it never
    shares the simulator's address space. Select it when this process also loads rclpy (a ROS2 sensor
    egress): the Unitree SDK's bundled CycloneDDS heap-corrupts if it coexists with rclpy's CycloneDDS
    in one process. Otherwise 'unitree' (in-process) is fine."""

    motor_type: str = "serial"
    """Motor communication type ('serial', etc.)."""

    controlled_dof_names: tuple[str, ...] = ()
    """Canonical DOF names in bridge motor order, without duplicates. Required by transports with
    fixed unnamed motor slots, such as Unitree and Booster, even for whole-robot control. A nonempty
    tuple takes precedence over ``excluded_dof_names``. Preserve this order when reordering
    ``RobotConfig.dof_names``."""

    excluded_dof_names: tuple[str, ...] = ()
    """DOFs to omit when ``controlled_dof_names`` is empty; remaining DOFs follow canonical order.
    Both empty selects all DOFs for bridges whose transport accepts canonical order. Unitree and Booster
    use fixed hardware motor IDs and require an explicit controlled tuple instead."""

    sdk_robot_type: str | None = None
    """robot_type presented to the SDK bridge (its supported-type gate + motor-vector sizing).
    ``None`` (default) = use ``asset.robot_type`` unchanged.

    Set this when the asset's ``robot_type`` is not itself an SDK motor model — e.g. a robot whose
    asset id differs from the SDK's fixed-size motor model it should be driven as. The controlled
    DOF subset must then match that model's motor count."""


@dataclass(frozen=True, config=_FORBID_EXTRA)
class RobotInitState:
    """Where the robot spawns and how its joints start."""

    position: list[float]
    """Base position [x, y, z] in metres, before per-env origins are added."""

    orientation: UnitQuaternionWXYZ = field(
        default_factory=lambda: [1.0, 0.0, 0.0, 0.0], metadata={"validate_default": True}
    )
    """Base unit quaternion ``[w, x, y, z]``. Norm deviations up to 1e-4 are normalized. Default identity."""

    lin_vel: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """Initial base linear velocity [m/s], world frame."""

    ang_vel: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """Initial base angular velocity [rad/s], world frame."""

    default_joint_positions: dict[str, float] = field(default_factory=dict)
    """Default joint positions (radians for hinge/revolute joints, metres for slide/prismatic joints) keyed
    by an exact ``dof_names`` entry or a ``#joint_tag`` of the owning ``RobotConfig``. Unspecified
    joints start at zero; overlapping source keys are rejected. See :meth:`RobotConfig.default_dof_positions`."""

    @property
    def orientation_xyzw(self) -> list[float]:
        """The orientation in the runtime tensor convention ``[x, y, z, w]``."""
        w, x, y, z = self.orientation
        return [x, y, z, w]

    def root_state_list(self) -> list[float]:
        """The 13-wide runtime root-state row ``[position(3), quat_xyzw(4), lin_vel(3), ang_vel(3)]``."""
        return [*self.position, *self.orientation_xyzw, *self.lin_vel, *self.ang_vel]

    @model_validator(mode="after")
    def _validate_shapes(self) -> RobotInitState:
        if len(self.position) != 3:
            raise ValueError(f"RobotInitState.position must have 3 entries, got {self.position}")
        if len(self.lin_vel) != 3 or len(self.ang_vel) != 3:
            raise ValueError(
                f"RobotInitState.lin_vel/ang_vel must have 3 entries each, got {self.lin_vel} / {self.ang_vel}"
            )
        return self


@dataclass(frozen=True, config=_FORBID_EXTRA)
class RobotAssetConfig:
    asset_root: str
    collapse_fixed_joints: bool
    replace_cylinder_with_capsule: bool
    flip_visual_attachments: bool
    thickness: float
    urdf_file: str
    usd_file: str | None
    xml_file: str
    robot_type: str
    enable_self_collisions: bool
    default_dof_drive_mode: int
    fix_base_link: bool
    """Weld the robot's root to the world, for example for a fixed-base arm, instead of giving it a free joint."""
    disable_gravity: bool | None = None

    link_physics: PhysicsConfig | None = None
    """Shared physics applied to all robot links (``body_names='.*'``) via the same ``PhysicsConfig``
    scene objects use. Holds the robot's surface/material/per-rigid-body physics: ``density`` and the
    PhysX solver knobs (``physx.linear_damping``/``angular_damping``/``max_linear_velocity``/
    ``max_angular_velocity``), plus the friction/restitution/collision-offset channels objects get
    (``isaacgym``/``isaacsim``/``mujoco``). Each backend reads its own sub-config and ignores the rest.
    ``None`` (default) keeps the asset's authored physics.

    ``mass`` is rejected here (see :meth:`_validate_link_physics`): one mass applied to every link is
    not meaningful for an articulation. Per-link physics (the only place per-link ``mass`` would
    apply) is not exposed here yet: it cannot be made cross-backend cleanly (a per-link
    collision-offset/damping override is whole-subtree, not per-link, on IsaacSim), and a one-backend
    knob would violate the shared-config invariant."""

    @model_validator(mode="after")
    def _validate_link_physics(self) -> RobotAssetConfig:
        """``link_physics`` applies to all links (``body_names='.*'``), so it must not carry ``mass``.

        A single ``mass`` applied to every link would set each link to the same mass (and on
        IsaacGym/IsaacSim the apply path writes it to all bodies), which is not what an articulation
        wants. This is a structural check, not a backend-applicability one."""
        if self.link_physics is not None and self.link_physics.mass is not None:
            raise ValueError(
                "RobotAssetConfig.link_physics carries mass, but it applies to all links "
                "(body_names='.*'), which would give every link the same mass. Per-link mass is not "
                "supported (no cross-backend apply); leave mass unset."
            )
        return self


@dataclass(frozen=True, config=_FORBID_EXTRA)
class RobotConfig:
    """Robot articulation, initial state, and physical properties shared by the simulators.

    ``dof_names`` defines action and per-DOF value order; ``body_names`` defines body-observation order.
    Names match the asset exactly; tags select joints or bodies in task terms.
    Sparse name/tag property maps override asset values only for the joints they cover.
    """

    dof_names: list[str]
    """Every robot DOF is a NAMED 1-DOF joint (hinge/revolute or slide/prismatic). This order is the
    action order, independent of asset enumeration. Physical properties are keyed by name/tag."""
    body_names: list[str]
    """Canonical robot bodies in observation order. Extra asset helper bodies may be omitted."""
    init_state: RobotInitState

    asset: RobotAssetConfig

    dof_pos_limits: dict[str, tuple[float, float]] | None = None
    """Name/tag -> (lower, upper) position bounds (rad / m). Unspecified joints keep asset limits."""
    dof_vel_limits: dict[str, float] | None = None
    """Name/tag -> velocity limit (rad/s or m/s). Unspecified joints keep asset limits.
    MuJoCo reports these values (otherwise infinity), but does not enforce joint velocity limits."""
    dof_effort_limits: dict[str, float] | None = None
    """Name/tag -> effort limit (N m for hinges, N for slides). Unspecified joints keep asset limits."""
    dof_armature: dict[str, float] | None = None
    """Name/tag -> armature. Unspecified joints keep asset values."""
    dof_joint_friction: dict[str, float] | None = None
    """Name/tag -> joint friction. Unspecified joints keep asset values. MuJoCo uses friction loss
    (N m for hinges, N for slides); PhysX uses a dimensionless coefficient."""

    bridge: RobotBridgeConfig = field(default_factory=RobotBridgeConfig)
    """Bridge SDK configuration for this robot."""

    body_tags: dict[str, list[str]] = field(default_factory=dict)
    """Named sets of robot bodies, declared and referenced with a leading ``#``
    (``{"#feet": ["left_ankle_roll_link", "right_ankle_roll_link"]}``). Entries are exact ``body_names``
    entries or other ``#tags``. ``#all`` (every body) is defined automatically unless given. Used by
    tag-aware term parameters, e.g. a termination term's ``bodies=["#root", "#hip_links"]`` or the
    mass-randomization ``link_bodies``; see ``holosoma.utils.name_tags``."""

    joint_tags: dict[str, list[str]] = field(default_factory=dict)
    """Named sets of robot DOFs, same syntax as ``body_tags`` over ``dof_names``, ``#all`` included
    automatically. Used by action/controller parameters, ``default_joint_positions`` keys, or
    algorithm configs such as the symmetry augmentation's
    ``#mirror_left`` / ``#mirror_right`` pairs."""

    def default_dof_positions(self) -> dict[str, float]:
        """Return a fresh, validated map of exact DOF names to default joint positions.

        Expands tags in ``dof_names`` order without modifying the config. Unspecified joints get zero;
        overlapping assignments and non-finite values are rejected (rad for hinges, m for slides).
        Use ``list(result.values())`` for a vector.
        """
        supplied = expand_joint_value_map(
            self.init_state.default_joint_positions,
            self.joint_tags,
            self.dof_names,
            what="RobotConfig.init_state.default_joint_positions",
            required=False,
        )
        return {name: supplied.get(name, 0.0) for name in self.dof_names}

    @model_validator(mode="after")
    def _validate_articulation(self) -> RobotConfig:
        """Cross-field consistency: every per-DOF table is indexed by ``dof_names``."""
        _require_unique(self.dof_names, "dof_names")
        if not self.body_names:
            raise ValueError("RobotConfig.body_names must not be empty.")
        _require_unique(self.body_names, "body_names")
        _require_unique(list(self.bridge.controlled_dof_names), "bridge.controlled_dof_names")
        bridge_names = set(self.bridge.controlled_dof_names) | set(self.bridge.excluded_dof_names)
        unknown_bridge = sorted(bridge_names - set(self.dof_names))
        if unknown_bridge:
            raise ValueError(f"RobotConfig.bridge names {unknown_bridge} are not in dof_names.")

        # '#all' is always available; a declared '#all' wins.
        object.__setattr__(self, "body_tags", {"#all": list(self.body_names), **self.body_tags})
        if self.dof_names:
            object.__setattr__(self, "joint_tags", {"#all": list(self.dof_names), **self.joint_tags})
        validate_tag_table(self.body_tags, self.body_names, kind="body")
        validate_tag_table(self.joint_tags, self.dof_names, kind="joint")
        # Keyed-by-DOF tables resolve against the validated tag table.
        resolve_dof_properties(self)
        return self


def _require_unique(names: list[str], field_name: str) -> None:
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        raise ValueError(f"RobotConfig.{field_name} has duplicate entries: {duplicates}")
