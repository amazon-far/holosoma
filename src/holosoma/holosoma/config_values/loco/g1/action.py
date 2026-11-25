"""Locomotion action and controller presets for the G1 robot."""

from holosoma.config_types.action import ActionManagerCfg, ActionTermCfg, JointControllerCfg

g1_29dof_joint_pos = ActionManagerCfg(
    terms={
        "joint_control": ActionTermCfg(
            func="holosoma.managers.action.terms.joint_control:JointPositionActionTerm",
            params={},
            scale=0.25,
            clip=(-100.0, 100.0),
            controller=JointControllerCfg(
                control_type="P",
                stiffness={
                    "#hip_yaw": 40.179238471,
                    "#hip_roll": 99.098427777,
                    "#hip_pitch": 40.179238471,
                    "#knees": 99.098427777,
                    "#ankle_pitch": 28.501246196,
                    "#ankle_roll": 28.501246196,
                    "waist_yaw_joint": 40.179238471,
                    "waist_roll_joint": 28.501246196,
                    "waist_pitch_joint": 28.501246196,
                    "#shoulder_pitch": 14.250623098,
                    "#shoulder_roll": 14.250623098,
                    "#shoulder_yaw": 14.250623098,
                    "#elbow": 14.250623098,
                    "#wrist_roll": 14.250623098,
                    "#wrist_pitch": 16.778327481,
                    "#wrist_yaw": 16.778327481,
                },
                damping={
                    "#hip_yaw": 2.557889765,
                    "#hip_roll": 6.308801854,
                    "#hip_pitch": 2.557889765,
                    "#knees": 6.308801854,
                    "#ankle_pitch": 1.814445687,
                    "#ankle_roll": 1.814445687,
                    "waist_yaw_joint": 2.557889765,
                    "waist_roll_joint": 1.814445687,
                    "waist_pitch_joint": 1.814445687,
                    "#shoulder_pitch": 0.907222843,
                    "#shoulder_roll": 0.907222843,
                    "#shoulder_yaw": 0.907222843,
                    "#elbow": 0.907222843,
                    "#wrist_roll": 0.907222843,
                    "#wrist_pitch": 1.068141502,
                    "#wrist_yaw": 1.068141502,
                },
            ),
        ),
    }
)

__all__ = ["g1_29dof_joint_pos"]
