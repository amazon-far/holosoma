"""Locomotion action presets for the G1 robot."""

from holosoma.config_types.action import ActionManagerCfg, ActionTermCfg

ARM_HOLD_JOINT_ANGLES = {
    "left_shoulder_pitch_joint": -0.75,
    "left_shoulder_roll_joint": 0.35,
    "left_shoulder_yaw_joint": 0.08,
    "left_elbow_joint": 0.60,
    "left_wrist_roll_joint": 0.00,
    "left_wrist_pitch_joint": 0.20,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": -0.75,
    "right_shoulder_roll_joint": -0.40,
    "right_shoulder_yaw_joint": 0.35,
    "right_elbow_joint": 0.65,
    "right_wrist_roll_joint": 0.00,
    "right_wrist_pitch_joint": -0.10,
    "right_wrist_yaw_joint": 0.0,
}

g1_29dof_joint_pos = ActionManagerCfg(
    terms={
        "joint_control": ActionTermCfg(
            func="holosoma.managers.action.terms.joint_control:JointPositionActionTerm",
            params={},
            scale=1.0,
            clip=None,
        ),
    }
)

g1_29dof_arm_hold_joint_pos = ActionManagerCfg(
    terms={
        "joint_control": ActionTermCfg(
            func="holosoma.managers.action.terms.joint_control:FixedJointPositionActionTerm",
            params={"fixed_joint_angles": ARM_HOLD_JOINT_ANGLES},
            scale=1.0,
            clip=None,
        ),
    }
)

__all__ = ["ARM_HOLD_JOINT_ANGLES", "g1_29dof_arm_hold_joint_pos", "g1_29dof_joint_pos"]
