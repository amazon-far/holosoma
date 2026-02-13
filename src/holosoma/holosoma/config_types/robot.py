from __future__ import annotations

from dataclasses import field

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class RobotBridgeConfig:
    """Bridge-specific configuration for robot SDK communication.

    Currently supports sim2sim (holosoma/run_sim.py) only.
    """

    sdk_type: str = "unitree"
    """SDK type for robot communication ('unitree', 'booster', 'ros2')."""

    motor_type: str = "serial"
    """Motor communication type ('serial', etc.)."""

    num_sdk_motors: int | None = None
    """Number of SDK motor slots. None = same as simulator DOFs."""

    joint2motor: tuple[int, ...] | None = None
    """Maps simulator joint index to SDK motor index (length: num_dofs). None = identity."""


@dataclass(frozen=True)
class RobotInitState:
    pos: list[float]
    rot: list[float]
    lin_vel: list[float]
    ang_vel: list[float]
    default_joint_angles: dict[str, float]


@dataclass(frozen=True)
class RobotControlConfig:
    control_type: str
    stiffness: dict[str, float]
    damping: dict[str, float]
    action_scale: float
    action_clip_value: float
    clip_actions: bool
    clip_torques: bool
    action_scales_by_effort_limit_over_p_gain: bool = False


@dataclass(frozen=True)
class RobotAssetConfig:
    asset_root: str
    collapse_fixed_joints: bool
    replace_cylinder_with_capsule: bool
    flip_visual_attachments: bool
    armature: float
    thickness: float
    max_angular_velocity: float
    max_linear_velocity: float
    angular_damping: float
    linear_damping: float
    urdf_file: str
    usd_file: str | None
    xml_file: str
    robot_type: str
    enable_self_collisions: bool
    default_dof_drive_mode: int
    fix_base_link: bool
    mesh_root: str | None = None
    density: float | None = None
    disable_gravity: bool | None = None


@dataclass(frozen=True)
class RobotForceControlConfig:
    apply_force_link: list[str] | None = None
    left_hand_link: str | None = None
    right_hand_link: str | None = None


@dataclass(frozen=True)
class ObjectConfig:
    object_urdf_path: str | None = None


@dataclass(frozen=True)
class RobotConfig:
    num_bodies: int
    dof_obs_size: int
    algo_obs_dim_dict: dict[str, int]
    actions_dim: int
    policy_obs_dim: int
    critic_obs_dim: int
    contact_pairs_multiplier: int
    key_bodies: list[str]
    num_feet: int
    foot_body_name: str
    """Name/pattern of the real foot link(s) used for contacts and kinematics."""
    foot_height_name: str
    """Name/pattern of auxiliary 'fake' foot link(s) used only to compute foot height/clearance"""
    knee_name: str
    torso_name: str
    dof_names: list[str]
    upper_dof_names: list[str]
    upper_left_arm_dof_names: list[str]
    upper_right_arm_dof_names: list[str]
    lower_dof_names: list[str]
    has_torso: bool
    has_upper_body_dof: bool
    left_ankle_dof_names: list[str]
    right_ankle_dof_names: list[str]
    knee_dof_names: list[str]
    hips_dof_names: list[str]
    dof_pos_lower_limit_list: list[float]
    dof_pos_upper_limit_list: list[float]
    dof_vel_limit_list: list[float]
    dof_effort_limit_list: list[float]
    dof_armature_list: list[float]
    dof_joint_friction_list: list[float]
    body_names: list[str]
    terminate_after_contacts_on: list[str]
    penalize_contacts_on: list[str]
    init_state: RobotInitState
    randomize_link_body_names: list[str]

    control: RobotControlConfig
    asset: RobotAssetConfig

    # TODO(jchen): talk to SAM, merge this into scene config
    object: ObjectConfig = field(default_factory=ObjectConfig)

    bridge: RobotBridgeConfig = field(default_factory=RobotBridgeConfig)
    """Bridge SDK configuration for this robot."""

    waist_dof_names: list[str] | None = None
    waist_yaw_dof_name: str | None = None
    waist_roll_dof_name: str | None = None
    waist_pitch_dof_name: str | None = None

    arm_dof_names: list[str] | None = None
    left_arm_dof_names: list[str] | None = None
    right_arm_dof_names: list[str] | None = None

    symmetry_joint_names: dict[str, str] | None = None
    flip_sign_joint_names: list[str] | None = None

    apply_dof_armature_in_isaacgym: bool = True
    knee_joint_min_threshold: float = 0.2
    lidar_height_offset: float = 0.5

    soft_dof_pos_limit: float = 0.95
    termination_close_to_dof_pos_limit: float = 0.98
