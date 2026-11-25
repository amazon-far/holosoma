"""Locomotion action and controller presets for the T1 robot."""

from holosoma.config_types.action import ActionManagerCfg, ActionTermCfg, JointControllerCfg

t1_29dof_joint_pos = ActionManagerCfg(
    terms={
        "joint_control": ActionTermCfg(
            func="holosoma.managers.action.terms.joint_control:JointPositionActionTerm",
            params={},
            scale=0.25,
            clip=(-100.0, 100.0),
            controller=JointControllerCfg(
                control_type="P",
                stiffness={
                    "AAHead_yaw": 5,
                    "Head_pitch": 5,
                    "#hip_yaw": 200,
                    "#hip_roll": 200,
                    "#hip_pitch": 200,
                    "#knees": 200,
                    "#ankle_pitch": 50,
                    "#ankle_roll": 50,
                    "Waist": 200,
                    "#shoulder_pitch": 20,
                    "#shoulder_roll": 20,
                    "#elbow_pitch": 20,
                    "#elbow_yaw": 20,
                    "#wrist_pitch": 20,
                    "#wrist_yaw": 20,
                    "#hand_roll": 20,
                },
                damping={
                    "AAHead_yaw": 0.5,
                    "Head_pitch": 0.5,
                    "#hip_yaw": 5,
                    "#hip_roll": 5,
                    "#hip_pitch": 5,
                    "#knees": 5,
                    "#ankle_pitch": 3,
                    "#ankle_roll": 3,
                    "Waist": 5,
                    "#shoulder_pitch": 0.5,
                    "#shoulder_roll": 0.5,
                    "#elbow_pitch": 0.5,
                    "#elbow_yaw": 0.5,
                    "#wrist_pitch": 0.5,
                    "#wrist_yaw": 0.5,
                    "#hand_roll": 0.5,
                },
            ),
        ),
    }
)

__all__ = ["t1_29dof_joint_pos"]
