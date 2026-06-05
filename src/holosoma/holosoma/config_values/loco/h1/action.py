"""Locomotion action presets for the H1 robot."""

from holosoma.config_types.action import ActionManagerCfg, ActionTermCfg

h1_19dof_joint_pos = ActionManagerCfg(
    terms={
        "joint_control": ActionTermCfg(
            func="holosoma.managers.action.terms.joint_control:JointPositionActionTerm",
            params={},
            scale=1.0,
            clip=None,
        ),
    }
)

__all__ = ["h1_19dof_joint_pos"]

