"""Constants for the G1 high-level arm/loco clients.

Joint gains, index maps, and initialization trajectories are inlined here so
this package is self-contained with no external dependency.
"""

from __future__ import annotations

from enum import IntEnum


class HardwareG1j29JointIndex(IntEnum):
    # Left leg
    k_left_hip_pitch = 0
    k_left_hip_roll = 1
    k_left_hip_yaw = 2
    k_left_knee = 3
    k_left_ankle_pitch = 4
    k_left_ankle_roll = 5

    # Right leg
    k_right_hip_pitch = 6
    k_right_hip_roll = 7
    k_right_hip_yaw = 8
    k_right_knee = 9
    k_right_ankle_pitch = 10
    k_right_ankle_roll = 11

    k_waist_yaw = 12
    k_waist_roll = 13
    k_waist_pitch = 14

    # Left arm
    k_left_shoulder_pitch = 15
    k_left_shoulder_roll = 16
    k_left_shoulder_yaw = 17
    k_left_elbow = 18
    k_left_wrist_roll = 19
    k_left_wrist_pitch = 20
    k_left_wrist_yaw = 21

    # Right arm
    k_right_shoulder_pitch = 22
    k_right_shoulder_roll = 23
    k_right_shoulder_yaw = 24
    k_right_elbow = 25
    k_right_wrist_roll = 26
    k_right_wrist_pitch = 27
    k_right_wrist_yaw = 28

    # not used
    k_not_used_joint_0 = 29
    k_not_used_joint_1 = 30
    k_not_used_joint_2 = 31
    k_not_used_joint_3 = 32
    k_not_used_joint_4 = 33
    k_not_used_joint_5 = 34


class HardwareG1j29JointArmIndex(IntEnum):
    # Left arm
    k_left_shoulder_pitch = 15
    k_left_shoulder_roll = 16
    k_left_shoulder_yaw = 17
    k_left_elbow = 18
    k_left_wrist_roll = 19
    k_left_wrist_pitch = 20
    k_left_wrist_yaw = 21

    # Right arm
    k_right_shoulder_pitch = 22
    k_right_shoulder_roll = 23
    k_right_shoulder_yaw = 24
    k_right_elbow = 25
    k_right_wrist_roll = 26
    k_right_wrist_pitch = 27
    k_right_wrist_yaw = 28


KP_HOLOSOMA_MIXED_GAINS = {
    HardwareG1j29JointIndex.k_left_hip_pitch: 300,
    HardwareG1j29JointIndex.k_left_hip_roll: 300,
    HardwareG1j29JointIndex.k_left_hip_yaw: 300,
    HardwareG1j29JointIndex.k_left_knee: 300,
    HardwareG1j29JointIndex.k_left_ankle_pitch: 150,
    HardwareG1j29JointIndex.k_left_ankle_roll: 300,
    HardwareG1j29JointIndex.k_right_hip_pitch: 300,
    HardwareG1j29JointIndex.k_right_hip_roll: 300,
    HardwareG1j29JointIndex.k_right_hip_yaw: 300,
    HardwareG1j29JointIndex.k_right_knee: 300,
    HardwareG1j29JointIndex.k_right_ankle_pitch: 150,
    HardwareG1j29JointIndex.k_right_ankle_roll: 300,
    HardwareG1j29JointIndex.k_waist_yaw: 300,
    HardwareG1j29JointIndex.k_waist_roll: 300,
    HardwareG1j29JointIndex.k_waist_pitch: 300,
    HardwareG1j29JointIndex.k_left_shoulder_pitch: 14.250623098,
    HardwareG1j29JointIndex.k_left_shoulder_roll: 14.250623098,
    HardwareG1j29JointIndex.k_left_shoulder_yaw: 14.250623098,
    HardwareG1j29JointIndex.k_left_elbow: 14.250623098,
    HardwareG1j29JointIndex.k_left_wrist_roll: 14.250623098,
    HardwareG1j29JointIndex.k_left_wrist_pitch: 16.778327481,
    HardwareG1j29JointIndex.k_left_wrist_yaw: 16.778327481,
    HardwareG1j29JointIndex.k_right_shoulder_pitch: 14.250623098,
    HardwareG1j29JointIndex.k_right_shoulder_roll: 14.250623098,
    HardwareG1j29JointIndex.k_right_shoulder_yaw: 14.250623098,
    HardwareG1j29JointIndex.k_right_elbow: 14.250623098,
    HardwareG1j29JointIndex.k_right_wrist_roll: 14.250623098,
    HardwareG1j29JointIndex.k_right_wrist_pitch: 16.778327481,
    HardwareG1j29JointIndex.k_right_wrist_yaw: 16.778327481,
    HardwareG1j29JointIndex.k_not_used_joint_0: 0,
    HardwareG1j29JointIndex.k_not_used_joint_1: 0,
    HardwareG1j29JointIndex.k_not_used_joint_2: 0,
    HardwareG1j29JointIndex.k_not_used_joint_3: 0,
    HardwareG1j29JointIndex.k_not_used_joint_4: 0,
    HardwareG1j29JointIndex.k_not_used_joint_5: 0,
}

KD_HOLOSOMA_MIXED_GAINS = {
    HardwareG1j29JointIndex.k_left_hip_pitch: 3.0,
    HardwareG1j29JointIndex.k_left_hip_roll: 3.0,
    HardwareG1j29JointIndex.k_left_hip_yaw: 3.0,
    HardwareG1j29JointIndex.k_left_knee: 3.0,
    HardwareG1j29JointIndex.k_left_ankle_pitch: 4.0,
    HardwareG1j29JointIndex.k_left_ankle_roll: 3.0,
    HardwareG1j29JointIndex.k_right_hip_pitch: 3.0,
    HardwareG1j29JointIndex.k_right_hip_roll: 3.0,
    HardwareG1j29JointIndex.k_right_hip_yaw: 3.0,
    HardwareG1j29JointIndex.k_right_knee: 3.0,
    HardwareG1j29JointIndex.k_right_ankle_pitch: 4.0,
    HardwareG1j29JointIndex.k_right_ankle_roll: 3.0,
    HardwareG1j29JointIndex.k_waist_yaw: 3.0,
    HardwareG1j29JointIndex.k_waist_roll: 3.0,
    HardwareG1j29JointIndex.k_waist_pitch: 3.0,
    HardwareG1j29JointIndex.k_left_shoulder_pitch: 0.907222843,
    HardwareG1j29JointIndex.k_left_shoulder_roll: 0.907222843,
    HardwareG1j29JointIndex.k_left_shoulder_yaw: 0.907222843,
    HardwareG1j29JointIndex.k_left_elbow: 0.907222843,
    HardwareG1j29JointIndex.k_left_wrist_roll: 0.907222843,
    HardwareG1j29JointIndex.k_left_wrist_pitch: 1.068141502,
    HardwareG1j29JointIndex.k_left_wrist_yaw: 1.068141502,
    HardwareG1j29JointIndex.k_right_shoulder_pitch: 0.907222843,
    HardwareG1j29JointIndex.k_right_shoulder_roll: 0.907222843,
    HardwareG1j29JointIndex.k_right_shoulder_yaw: 0.907222843,
    HardwareG1j29JointIndex.k_right_elbow: 0.907222843,
    HardwareG1j29JointIndex.k_right_wrist_roll: 0.907222843,
    HardwareG1j29JointIndex.k_right_wrist_pitch: 1.068141502,
    HardwareG1j29JointIndex.k_right_wrist_yaw: 1.068141502,
    HardwareG1j29JointIndex.k_not_used_joint_0: 0,
    HardwareG1j29JointIndex.k_not_used_joint_1: 0,
    HardwareG1j29JointIndex.k_not_used_joint_2: 0,
    HardwareG1j29JointIndex.k_not_used_joint_3: 0,
    HardwareG1j29JointIndex.k_not_used_joint_4: 0,
    HardwareG1j29JointIndex.k_not_used_joint_5: 0,
}

# Per-joint effort limits from URDF (Nm). Used to clamp position error so
# that kp * |q_target - q_actual| never exceeds the motor's rated torque.
EFFORT_LIMITS = {
    HardwareG1j29JointIndex.k_left_shoulder_pitch: 25.0,
    HardwareG1j29JointIndex.k_left_shoulder_roll: 25.0,
    HardwareG1j29JointIndex.k_left_shoulder_yaw: 25.0,
    HardwareG1j29JointIndex.k_left_elbow: 25.0,
    HardwareG1j29JointIndex.k_left_wrist_roll: 25.0,
    HardwareG1j29JointIndex.k_left_wrist_pitch: 5.0,
    HardwareG1j29JointIndex.k_left_wrist_yaw: 5.0,
    HardwareG1j29JointIndex.k_right_shoulder_pitch: 25.0,
    HardwareG1j29JointIndex.k_right_shoulder_roll: 25.0,
    HardwareG1j29JointIndex.k_right_shoulder_yaw: 25.0,
    HardwareG1j29JointIndex.k_right_elbow: 25.0,
    HardwareG1j29JointIndex.k_right_wrist_roll: 25.0,
    HardwareG1j29JointIndex.k_right_wrist_pitch: 5.0,
    HardwareG1j29JointIndex.k_right_wrist_yaw: 5.0,
}

# Arm initialization trajectory.
# Each waypoint is 14 floats: 7 left-arm joints then 7 right-arm joints,
# shoulder_pitch..wrist_yaw ordering.
G1_INIT_TRAJECTORY = [
    # Waypoint 0
    [
        # Left arm
        0.1011228933930397,
        0.2980724573135376,
        0.05482783168554306,
        1.2476897239685059,
        0.07854460924863815,
        0.12445186823606491,
        -0.45126649737358093,
        # Right arm
        0.15423697233200073,
        -0.2980724573135376,
        -0.20365992188453674,
        1.3728169202804565,
        -0.1312272697687149,
        -0.0011428158031776547,
        0.30169567465782166,
    ],
    # Waypoint 1
    [
        # Left arm
        0.13676397502422333,
        0.2980724573135376,
        0.059549614787101746,
        1.1739747524261475,
        -1.4571739435195923,
        0.29164811968803406,
        0.10575263947248459,
        # Right arm (mirrored from left)
        0.18770891427993774,
        -0.28143835067749023,
        -0.22148047387599945,
        1.3743269443511963,
        1.5250645875930786,
        0.16367575526237488,
        -0.019680975005030632,
    ],
    # Waypoint 2
    [
        # Left arm
        0.23777900636196136,
        0.30227890610694885,
        0.18426944315433502,
        1.3974086046218872,
        -1.5251843929290771,
        0.14970886707305908,
        -1.257028341293335,
        # Right arm
        0.23777900636196136,
        -0.30227890610694885,
        -0.18426944315433502,
        1.3974086046218872,
        1.5251843929290771,
        0.14970886707305908,
        1.257028341293335,
    ],
]
