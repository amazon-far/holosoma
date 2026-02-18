"""Default robot configurations for holosoma_inference.

This module provides pre-configured robot hardware and control parameters
for different robot types.
"""

from __future__ import annotations

from importlib.metadata import entry_points

from holosoma_inference.config.config_types.robot import RobotConfig

# =============================================================================
# G1 Robot Config
# =============================================================================

# fmt: off

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
# Default Configurations Dictionary
# =============================================================================

DEFAULTS = {
    "g1-29dof": g1_29dof,
    "t1-29dof": t1_29dof,
}
"""Dictionary of all available robot configurations.

Keys use hyphen-case naming convention for CLI compatibility.
"""

# Auto-discover robot configs from installed extensions
for ep in entry_points(group="holosoma.config.robot"):
    DEFAULTS[ep.name] = ep.load()
