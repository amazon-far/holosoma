"""Default termination manager configurations."""

from holosoma.config_types.termination import TerminationManagerCfg
from holosoma.config_values.loco.g1.termination import g1_29dof_termination
from holosoma.config_values.loco.t1.termination import t1_29dof_termination
from holosoma.config_values.wbt.g1.termination import g1_29dof_wbt_termination
from holosoma.utils.config_registry import ConfigRegistry

TERMINATION_REGISTRY = ConfigRegistry(TerminationManagerCfg, group="holosoma.config.termination")

none = TERMINATION_REGISTRY.add("none", None)
TERMINATION_REGISTRY.add("t1_29dof", t1_29dof_termination)
TERMINATION_REGISTRY.add("g1_29dof", g1_29dof_termination)
TERMINATION_REGISTRY.add("g1_29dof_wbt", g1_29dof_wbt_termination)

from holosoma.utils.config_registry import (  # noqa: E402
    deprecated_defaults_alias as _deprecated_defaults_alias,
)

__getattr__ = _deprecated_defaults_alias(__name__, TERMINATION_REGISTRY)
