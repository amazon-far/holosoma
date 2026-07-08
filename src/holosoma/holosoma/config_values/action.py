"""Default action manager configurations."""

from holosoma.config_values.loco.g1.action import g1_29dof_arm_hold_joint_pos, g1_29dof_joint_pos
from holosoma.config_values.loco.t1.action import t1_29dof_joint_pos

none = None

DEFAULTS = {
    "none": none,
    "t1_29dof_joint_pos": t1_29dof_joint_pos,
    "g1_29dof_arm_hold_joint_pos": g1_29dof_arm_hold_joint_pos,
    "g1_29dof_joint_pos": g1_29dof_joint_pos,
}
