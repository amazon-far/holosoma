"""Default action manager configurations."""

from holosoma.config_values.loco.g1.action import g1_29dof_joint_pos
from holosoma.config_values.loco.h1.action import h1_19dof_joint_pos
from holosoma.config_values.loco.t1.action import t1_29dof_joint_pos

none = None

DEFAULTS = {
    "none": none,
    "t1_29dof_joint_pos": t1_29dof_joint_pos,
    "g1_29dof_joint_pos": g1_29dof_joint_pos,
    "h1_19dof_joint_pos": h1_19dof_joint_pos,
}
