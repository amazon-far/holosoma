"""Default command manager configurations."""

from holosoma.config_types.command import CommandManagerCfg
from holosoma.config_values.loco.g1.command import g1_29dof_command
from holosoma.config_values.loco.t1.command import t1_29dof_command
from holosoma.config_values.wbt.g1.command import (
    g1_29dof_wbt_command,
    g1_29dof_wbt_command_w_object,
)
from holosoma.utils.config_registry import ConfigRegistry

COMMAND_REGISTRY = ConfigRegistry(CommandManagerCfg, group="holosoma.config.command")

none = COMMAND_REGISTRY.add("none", None)
COMMAND_REGISTRY.add("t1_29dof", t1_29dof_command)
COMMAND_REGISTRY.add("g1_29dof", g1_29dof_command)
COMMAND_REGISTRY.add("g1_29dof_wbt", g1_29dof_wbt_command)
COMMAND_REGISTRY.add("g1_29dof_wbt_w_object", g1_29dof_wbt_command_w_object)

from holosoma.utils.config_registry import (  # noqa: E402
    deprecated_defaults_alias as _deprecated_defaults_alias,
)

__getattr__ = _deprecated_defaults_alias(__name__, COMMAND_REGISTRY)
