"""Default robot configurations for holosoma_inference.

This module provides pre-configured robot hardware and control parameters
for different robot types.
"""

from __future__ import annotations

from holosoma_inference.compat import entry_points
from holosoma_inference.config.config_types.robot import RobotConfig

# =============================================================================
# G1 Robot Config
# =============================================================================

# fmt: off

# G1 29-DOF per-joint action scales for BeyondMimic-style scaling (0.25 * effort / p_gain).
# TODO: this is legacy for onnx that do not have action scale vector in metadata
G1_29DOF_PER_JOINT_ACTION_SCALE = (
    0.547546465219,
    0.350661466378,
    0.547546465219,
    0.350661466378,
    0.438577313919,
    0.438577313919,
    0.547546465219,
    0.350661466378,
    0.547546465219,
    0.350661466378,
    0.438577313919,
    0.438577313919,
    0.547546465219,
    0.438577313919,
    0.438577313919,
    0.438577313919,
    0.438577313919,
    0.438577313919,
    0.438577313919,
    0.438577313919,
    0.074500870329,
    0.074500870329,
    0.438577313919,
    0.438577313919,
    0.438577313919,
    0.438577313919,
    0.438577313919,
    0.074500870329,
    0.074500870329,
)

g1_29dof = RobotConfig(
    # Identity
    robot_type="g1_29dof",
    robot="g1",

    # SDK Configuration
    sdk_type="unitree",
    motor_type="serial",
    message_type="HG",
    use_sensor=False,

    # Dimensions
    num_motors=29,
    num_joints=29,
    num_upper_body_joints=14,

    # Default Positions
    default_dof_angles=(
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,  # left leg
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,  # right leg
        0.0, 0.0, 0.0,  # waist
        0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,  # left arm
        0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,  # right arm
    ),
    default_motor_angles=(
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,  # left leg
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,  # right leg
        0.0, 0.0, 0.0,  # waist
        0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,  # left arm
        0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,  # right arm
    ),

    # Mappings
    motor2joint=tuple(range(29)),  # Identity mapping
    joint2motor=tuple(range(29)),  # Identity mapping
    dof_names=(
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
        "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
        "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    ),
    dof_names_upper_body=(
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
        "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
        "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    ),
    dof_names_lower_body=(
        "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    ),

    # Link Names
    torso_link_name="torso_link",
    left_hand_link_name="left_rubber_hand",
    right_hand_link_name="right_rubber_hand",

    # Unitree-Specific Constants
    unitree_legged_const={
        "HIGHLEVEL": 238,
        "LOWLEVEL": 255,
        "TRIGERLEVEL": 240,
        "PosStopF": 2146000000.0,
        "VelStopF": 16000.0,
        "MODE_MACHINE": 5,
        "MODE_PR": 0,
    },
    weak_motor_joint_index={
        "left_hip_yaw_joint": 0, "left_hip_roll_joint": 1, "left_hip_pitch_joint": 2,
        "left_knee_joint": 3, "left_ankle_pitch_joint": 4, "left_ankle_roll_joint": 5,
        "right_hip_yaw_joint": 6, "right_hip_roll_joint": 7, "right_hip_pitch_joint": 8,
        "right_knee_joint": 9, "right_ankle_pitch_joint": 10, "right_ankle_roll_joint": 11,
        "waist_yaw_joint": 12, "waist_roll_joint": 13, "waist_pitch_joint": 14,
        "left_shoulder_pitch_joint": 15, "left_shoulder_roll_joint": 16,
        "left_shoulder_yaw_joint": 17, "left_elbow_joint": 18,
        "left_wrist_roll_joint": 19, "left_wrist_pitch_joint": 20, "left_wrist_yaw_joint": 21,
        "right_shoulder_pitch_joint": 22, "right_shoulder_roll_joint": 23,
        "right_shoulder_yaw_joint": 24, "right_elbow_joint": 25,
        "right_wrist_roll_joint": 26, "right_wrist_pitch_joint": 27, "right_wrist_yaw_joint": 28,
    },
    motion={"body_name_ref": ["torso_link"]},
    default_per_joint_action_scale=G1_29DOF_PER_JOINT_ACTION_SCALE,
)


# =============================================================================
# T1 Robot Config
# =============================================================================

t1_29dof = RobotConfig(
    # Identity
    robot_type="t1_29dof",
    robot="t1",

    # SDK Configuration
    sdk_type="booster",  # T1 uses booster SDK
    motor_type="serial",
    message_type="HG",  # Using default
    use_sensor=False,

    # Dimensions
    num_motors=29,
    num_joints=29,
    num_upper_body_joints=16,  # T1 has 16 upper body joints (includes head)

    # Default Positions
    default_dof_angles=(
        0.0, 0.0,  # head (yaw, pitch)
        0.2, -1.35, 0.0, -0.5, 0.0, 0.0, 0.0,  # left arm
        0.2, 1.35, 0.0, 0.5, 0.0, 0.0, 0.0,  # right arm
        0.0,  # waist
        -0.2, 0.0, 0.0, 0.4, -0.25, 0.0,  # left leg
        -0.2, 0.0, 0.0, 0.4, -0.25, 0.0,  # right leg
    ),
    default_motor_angles=(
        0.0, 0.0,  # head
        0.2, -1.35, 0.0, -0.5, 0.0, 0.0, 0.0,  # left arm
        0.2, 1.35, 0.0, 0.5, 0.0, 0.0, 0.0,  # right arm
        0.0,  # waist
        -0.2, 0.0, 0.0, 0.4, -0.25, 0.0,  # left leg
        -0.2, 0.0, 0.0, 0.4, -0.25, 0.0,  # right leg
    ),

    # Mappings
    motor2joint=tuple(range(29)),  # Identity mapping
    joint2motor=tuple(range(29)),  # Identity mapping
    dof_names=(
        "AAHead_yaw", "Head_pitch",
        "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
        "Left_Wrist_Pitch", "Left_Wrist_Yaw", "Left_Hand_Roll",
        "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
        "Right_Wrist_Pitch", "Right_Wrist_Yaw", "Right_Hand_Roll",
        "Waist",
        "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw",
        "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
        "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw",
        "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
    ),
    dof_names_upper_body=(
        "AAHead_yaw", "Head_pitch",
        "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
        "Left_Wrist_Pitch", "Left_Wrist_Yaw", "Left_Hand_Roll",
        "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
        "Right_Wrist_Pitch", "Right_Wrist_Yaw", "Right_Hand_Roll",
    ),
    dof_names_lower_body=(
        "Waist",
        "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw",
        "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
        "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw",
        "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
    ),

    # Link Names
    torso_link_name="Trunk",
    left_hand_link_name=None,
    right_hand_link_name=None,

    # Set unitree-specific values to `None`
    unitree_legged_const=None,
    weak_motor_joint_index=None,
    motion=None,
)


# =============================================================================
# H1 Robot Config
# =============================================================================

h1_19dof = RobotConfig(
    # Identity
    robot_type="h1",
    robot="h1",

    # SDK Configuration
    # H1 uses GO2 protocol. The C++ binding exposes 19 compact motor slots;
    # unitree_interface.py converts between 19-compact and 20-physical arrays
    # so that the rest of the codebase works with physical motor IDs (matching
    # Beyondmimic's dof_idx and the real DDS message layout).
    sdk_type="unitree",
    motor_type="serial",
    message_type="GO2",
    use_sensor=False,

    # Dimensions
    # 20 physical motor slots (ID 0..19), ID 9 unused on real H1 hardware.
    num_motors=20,
    num_joints=19,
    num_upper_body_joints=9,

    # Default Positions (joint order, matching dof_names)
    default_dof_angles=(
        0.0, 0.0, -0.4, 0.8, -0.4,  # left leg
        0.0, 0.0, -0.4, 0.8, -0.4,  # right leg
        0.0,  # torso
        0.0, 0.0, 0.0, 0.0,  # left arm
        0.0, 0.0, 0.0, 0.0,  # right arm
    ),
    # Default positions in physical motor order (20 slots, ID 9 = 0.0 dummy)
    default_motor_angles=(
        0.0, -0.4, 0.8, 0.0, -0.4, 0.8,  # physical motors 0..5
        0.0, 0.0, 0.0,                    # physical motors 6..8
        0.0,                               # physical motor 9 (unused)
        -0.4, -0.4,                       # physical motors 10..11
        0.0, 0.0, 0.0, 0.0,               # physical motors 12..15
        0.0, 0.0, 0.0, 0.0,               # physical motors 16..19
    ),

    # Mappings (joint order ↔ physical motor ID, matching Beyondmimic dof_idx)
    # motor2joint: physical motor ID → joint index (-1 = unused ID 9)
    motor2joint=(6, 7, 8, 1, 2, 3, 10, 0, 5, -1, 4, 9, 15, 16, 17, 18, 11, 12, 13, 14),
    # joint2motor: joint index → physical motor ID
    joint2motor=(7, 3, 4, 5, 10, 8, 0, 1, 2, 11, 6, 16, 17, 18, 19, 12, 13, 14, 15),
    dof_names=(
        "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint",
        "left_knee_joint", "left_ankle_joint",
        "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint",
        "right_knee_joint", "right_ankle_joint",
        "torso_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint", "left_elbow_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint", "right_elbow_joint",
    ),
    dof_names_upper_body=(
        "torso_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint", "left_elbow_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint", "right_elbow_joint",
    ),
    dof_names_lower_body=(
        "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint",
        "left_knee_joint", "left_ankle_joint",
        "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint",
        "right_knee_joint", "right_ankle_joint",
    ),

    # Control Gains (physical motor order, 20 slots, ID 9 = 0.0 dummy)
    motor_kp=(
        200.0, 200.0, 300.0, 200.0, 200.0, 300.0,
        300.0, 200.0, 200.0,
        0.0,                                # motor 9 (unused)
        40.0, 40.0,
        100.0, 100.0, 100.0, 100.0,
        100.0, 100.0, 100.0, 100.0,
    ),
    motor_kd=(
        5.0, 5.0, 6.0, 5.0, 5.0, 6.0,
        6.0, 5.0, 5.0,
        0.0,                                # motor 9 (unused)
        2.0, 2.0,
        2.0, 2.0, 2.0, 2.0,
        2.0, 2.0, 2.0, 2.0,
    ),

    # Link Names
    torso_link_name="torso_link",
    left_hand_link_name=None,
    right_hand_link_name=None,

    # Unitree-Specific Constants
    unitree_legged_const={
        "HIGHLEVEL": 238,
        "LOWLEVEL": 255,
        "TRIGERLEVEL": 240,
        "PosStopF": 2146000000.0,
        "VelStopF": 16000.0,
        "MODE_MACHINE": 5,
        "MODE_PR": 0,
    },
    # weak_motor_joint_index: joint name → physical motor ID
    weak_motor_joint_index={
        "left_hip_yaw_joint": 7,
        "left_hip_roll_joint": 3,
        "left_hip_pitch_joint": 4,
        "left_knee_joint": 5,
        "left_ankle_joint": 10,
        "right_hip_yaw_joint": 8,
        "right_hip_roll_joint": 0,
        "right_hip_pitch_joint": 1,
        "right_knee_joint": 2,
        "right_ankle_joint": 11,
        "torso_joint": 6,
        "left_shoulder_pitch_joint": 16,
        "left_shoulder_roll_joint": 17,
        "left_shoulder_yaw_joint": 18,
        "left_elbow_joint": 19,
        "right_shoulder_pitch_joint": 12,
        "right_shoulder_roll_joint": 13,
        "right_shoulder_yaw_joint": 14,
        "right_elbow_joint": 15,
    },
    motion={"body_name_ref": ["torso_link"]},
)

h1_19dof_real_go = RobotConfig(
    # H1 real-robot defaults (matching Beyondmimic h1.yaml FixStand posture).
    # Uses GO2 protocol with 20-element physical motor arrays.
    robot_type="h1",
    robot="h1",
    sdk_type="unitree",
    motor_type="serial",
    message_type="GO2",
    use_sensor=False,
    num_motors=20,
    num_joints=19,
    num_upper_body_joints=9,
    # DOF-order defaults (joint order)
    default_dof_angles=(
        0.0, 0.0, -0.28, 0.79, -0.52,  # left leg
        0.0, 0.0, -0.28, 0.79, -0.52,  # right leg
        0.0,  # torso
        0.2, 0.0, 0.0, 0.32,  # left arm
        0.2, 0.0, 0.0, 0.32,  # right arm
    ),
    # Physical-motor-order defaults (20 slots, ID 9 = 0.0)
    default_motor_angles=(
        0.0, -0.28, 0.79, 0.0, -0.28, 0.79,  # physical motors 0..5
        0.0, 0.0, 0.0,                       # physical motors 6..8
        0.0,                                  # physical motor 9 (unused)
        -0.52, -0.52,                        # physical motors 10..11
        0.2, 0.0, 0.0, 0.32,                 # physical motors 12..15
        0.2, 0.0, 0.0, 0.32,                 # physical motors 16..19
    ),
    joint2motor=h1_19dof.joint2motor,
    motor2joint=h1_19dof.motor2joint,
    dof_names=h1_19dof.dof_names,
    dof_names_upper_body=h1_19dof.dof_names_upper_body,
    dof_names_lower_body=h1_19dof.dof_names_lower_body,
    # Physical-motor-order gains (20 slots, ID 9 = 0.0)
    motor_kp=(
        300.0, 300.0, 300.0, 300.0, 300.0, 300.0,  # motors 0-5
        300.0, 300.0, 300.0,                         # motors 6-8
        0.0,                                          # motor 9 (unused)
        60.0, 60.0,                                   # motors 10-11 (ankles)
        60.0, 60.0, 60.0, 60.0,                       # motors 12-15
        60.0, 60.0, 60.0, 60.0,                       # motors 16-19
    ),
    motor_kd=(
        5.0, 5.0, 6.0, 5.0, 5.0, 6.0,
        6.0, 5.0, 5.0,
        0.0,                                # motor 9 (unused)
        1.5, 1.5,
        1.5, 1.5, 1.5, 1.5,
        1.5, 1.5, 1.5, 1.5,
    ),
    torso_link_name=h1_19dof.torso_link_name,
    left_hand_link_name=h1_19dof.left_hand_link_name,
    right_hand_link_name=h1_19dof.right_hand_link_name,
    unitree_legged_const=h1_19dof.unitree_legged_const,
    # weak_motor_joint_index: joint name → physical motor ID
    weak_motor_joint_index={
        "left_hip_yaw_joint": 7,
        "left_hip_roll_joint": 3,
        "left_hip_pitch_joint": 4,
        "left_knee_joint": 5,
        "left_ankle_joint": 10,
        "right_hip_yaw_joint": 8,
        "right_hip_roll_joint": 0,
        "right_hip_pitch_joint": 1,
        "right_knee_joint": 2,
        "right_ankle_joint": 11,
        "torso_joint": 6,
        "left_shoulder_pitch_joint": 16,
        "left_shoulder_roll_joint": 17,
        "left_shoulder_yaw_joint": 18,
        "left_elbow_joint": 19,
        "right_shoulder_pitch_joint": 12,
        "right_shoulder_roll_joint": 13,
        "right_shoulder_yaw_joint": 14,
        "right_elbow_joint": 15,
    },
    motion=h1_19dof.motion,
)


# =============================================================================
# Default Configurations Dictionary
# =============================================================================

# Core defaults - no extension imports at module load time
DEFAULTS = {
    "g1-29dof": g1_29dof,
    "t1-29dof": t1_29dof,
    "h1-19dof": h1_19dof,
    "h1-19dof-real-go": h1_19dof_real_go,
}
"""Dictionary of all available robot configurations.

Keys use hyphen-case naming convention for CLI compatibility.
"""

# Track whether extensions have been loaded
_extensions_loaded = False


def _load_extensions() -> None:
    """Lazily load extension configs from entry points.

    This is deferred to avoid circular imports when extensions import
    from holosoma_inference.config at module load time.
    """
    global _extensions_loaded  # noqa: PLW0603
    if _extensions_loaded:
        return
    _extensions_loaded = True
    for ep in entry_points(group="holosoma.config.robot"):
        DEFAULTS[ep.name] = ep.load()


def get_defaults() -> dict:
    """Get all robot config defaults, including extensions.

    Returns:
        Dictionary mapping config names to RobotConfig instances.
    """
    _load_extensions()
    return DEFAULTS
