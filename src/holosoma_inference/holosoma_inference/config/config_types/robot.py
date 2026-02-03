"""Robot configuration types for holosoma_inference."""

from __future__ import annotations

import typing
from typing import Any

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class RobotConfig:
    """Robot hardware and control configuration.

    Defines all robot-specific parameters including kinematics,
    control gains, limits, and SDK configuration.

    Examples
    --------
    G1 29-DOF robot configuration:
    >>> robot_config = RobotConfig(
    ...     robot_type="g1_29dof",
    ...     robot="g1",
    ...     num_motors=29,
    ...     num_joints=29,
    ...     default_dof_angles=(...),  # 29 values
    ...     motor_kp=(...),  # 29 values
    ...     # ... other required fields
    ... )
    """

    # =========================================================================
    # Identity (REQUIRED - no defaults)
    # =========================================================================

    robot_type: str
    """Robot identifier (e.g., 'g1_29dof', 't1_29dof')."""

    robot: str
    """Robot family name (e.g., 'g1', 't1')."""

    # =========================================================================
    # Default Positions (REQUIRED - immutable tuples, no defaults)
    # =========================================================================

    default_dof_angles: tuple[float, ...]
    """Default joint angles in radians (length: num_joints).

    These are the target positions when the robot is in its default standing pose.
    """

    default_motor_angles: tuple[float, ...]
    """Default motor angles in radians (length: num_motors).

    Typically identical to default_dof_angles unless there's a motor-to-joint mapping.
    """

    # =========================================================================
    # Limits (REQUIRED - no defaults)
    # =========================================================================

    joint_pos_min: tuple[float, ...]
    """Lower position limits for joints in radians (length: num_joints)."""

    joint_pos_max: tuple[float, ...]
    """Upper position limits for joints in radians (length: num_joints)."""

    joint_vel_limit: tuple[float, ...]
    """Velocity limits for joints in rad/s (length: num_joints)."""

    motor_effort_limit: tuple[float, ...]
    """Torque/effort limits for motors in Nm (length: num_motors)."""

    # =========================================================================
    # Mappings (REQUIRED - no defaults)
    # =========================================================================

    motor2joint: tuple[int, ...]
    """Motor index to joint index mapping (length: num_motors).

    Maps motor indices to their corresponding joint indices.
    For most robots, this is an identity mapping [0, 1, 2, ..., n-1].
    """

    joint2motor: tuple[int, ...]
    """Joint index to motor index mapping (length: num_joints).

    Inverse of motor2joint mapping.
    """

    dof_names: tuple[str, ...]
    """Joint names in order (length: num_joints).

    Example: ('left_hip_pitch_joint', 'left_hip_roll_joint', ...)
    """

    # =========================================================================
    # Deprecation candidates with no defaults
    # =========================================================================

    dof_names_upper_body: tuple[str, ...]
    """Upper body joint names (length: num_upper_body_joints).

    Subset of dof_names containing only upper body joints.
    """

    dof_names_lower_body: tuple[str, ...]
    """Lower body joint names (length: num_joints - num_upper_body_joints).

    Subset of dof_names containing only lower body joints (legs + waist).
    """

    # =========================================================================
    # Control Gains (OPTIONAL - can be loaded from ONNX metadata)
    # =========================================================================

    motor_kp: tuple[float, ...] | None = None
    """Proportional gains for PD control (length: num_motors).

    These gains are applied in the motor-space PD controller:
    tau = kp * (q_target - q_current) + kd * (dq_target - dq_current)

    If None, values will be loaded from ONNX model metadata during inference.
    If provided, these values override the ONNX metadata.
    """

    motor_kd: tuple[float, ...] | None = None
    """Derivative gains for PD control (length: num_motors).

    These gains are applied in the motor-space PD controller.

    If None, values will be loaded from ONNX model metadata during inference.
    If provided, these values override the ONNX metadata.
    """

    # =========================================================================
    # WBT Stiff Startup Configuration (OPTIONAL - for WBT policies)
    # =========================================================================

    stiff_startup_pos: tuple[float, ...] | None = None
    """Stiff startup joint positions for WBT policy (length: num_joints).

    Target positions used during stiff hold mode before policy activation.
    Only used by WholeBodyTracking policy.
    """

    stiff_startup_kp: tuple[float, ...] | None = None
    """Stiff startup position gains for WBT policy (length: num_joints).

    Proportional gains used during stiff hold mode before policy activation.
    Only used by WholeBodyTracking policy.
    """

    stiff_startup_kd: tuple[float, ...] | None = None
    """Stiff startup velocity gains for WBT policy (length: num_joints).

    Derivative gains used during stiff hold mode before policy activation.
    Only used by WholeBodyTracking policy.
    """

    # =========================================================================
    # SDK Configuration (OPTIONAL - with defaults)
    # =========================================================================

    sdk_type: typing.Literal["unitree", "booster", "ros2"] = "unitree"
    """SDK type for robot communication."""

    motor_type: typing.Literal["serial", "parallel"] = "serial"
    """Motor communication type."""

    message_type: typing.Literal["HG", "GO2"] = "HG"
    """Message protocol type."""

    # =========================================================================
    # Dimensions (OPTIONAL - with defaults)
    # =========================================================================

    num_motors: int = 29
    """Number of motors in the robot."""

    num_joints: int = 29
    """Number of joints in the robot."""

    # =========================================================================
    # Link Names (OPTIONAL - with defaults)
    # =========================================================================

    torso_link_name: str = "torso_link"
    """Name of the torso/base link in the robot model."""

    left_hand_link_name: str | None = None
    """Name of the left hand/end-effector link (if applicable)."""

    right_hand_link_name: str | None = None
    """Name of the right hand/end-effector link (if applicable)."""

    # =========================================================================
    # SDK-Specific Configuration (optional)
    # =========================================================================

    unitree_legged_const: dict[str, Any] | None = None
    """Unitree SDK-specific constants.

    Contains protocol-specific values like HIGHLEVEL, LOWLEVEL, etc.
    Only used when sdk_type='unitree'.
    """

    weak_motor_joint_index: dict[str, int] | None = None
    """Mapping of joint names to their weak motor indices.

    Used for robots with weak motors that need special handling.
    """

    motion: dict[str, list[str]] | None = None
    """Motion reference configuration.

    Specifies body names used for motion tracking/reference.
    Example: {'body_name_ref': ['torso_link']}
    """

    # =========================================================================
    # Deprecation candidates
    # (I'm moving for parity with hydra, but I'd like to remove in the future)
    # =========================================================================

    dof_names_parallel_mech: tuple[str, ...] = ()
    """Parallel mechanism joint names (optional, booster specific)."""

    use_sensor: bool = False
    """Whether to use sensor feedback instead of joint states."""

    num_upper_body_joints: int = 14
    """Number of upper body degrees of freedom."""
