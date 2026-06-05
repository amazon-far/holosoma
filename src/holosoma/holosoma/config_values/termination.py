"""Default termination manager configurations."""

from holosoma.config_values.loco.g1.termination import g1_29dof_termination
from holosoma.config_values.loco.h1.termination import h1_19dof_termination
from holosoma.config_values.loco.t1.termination import t1_29dof_termination
from holosoma.config_values.wbt.g1.termination import g1_29dof_wbt_termination

none = None

DEFAULTS = {
    "none": none,
    "t1_29dof": t1_29dof_termination,
    "g1_29dof": g1_29dof_termination,
    "h1_19dof": h1_19dof_termination,
    "g1_29dof_wbt": g1_29dof_wbt_termination,
}
