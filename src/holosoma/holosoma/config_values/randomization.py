"""Default randomization manager configurations."""

from holosoma.config_types.randomization import RandomizationManagerCfg
from holosoma.config_values.loco.g1.randomization import g1_29dof_randomization
from holosoma.config_values.loco.t1.randomization import t1_29dof_randomization
from holosoma.config_values.wbt.g1.randomization import g1_29dof_wbt_randomization, g1_29dof_wbt_randomization_w_object
from holosoma.utils.config_registry import ConfigRegistry

RANDOMIZATION_REGISTRY = ConfigRegistry(RandomizationManagerCfg, group="holosoma.config.randomization")

none = RANDOMIZATION_REGISTRY.add("none", None)
RANDOMIZATION_REGISTRY.add("t1_29dof", t1_29dof_randomization)
RANDOMIZATION_REGISTRY.add("g1_29dof", g1_29dof_randomization)
RANDOMIZATION_REGISTRY.add("g1_29dof_wbt", g1_29dof_wbt_randomization)
RANDOMIZATION_REGISTRY.add("g1_29dof_wbt_w_object", g1_29dof_wbt_randomization_w_object)

from holosoma.utils.config_registry import (  # noqa: E402
    deprecated_defaults_alias as _deprecated_defaults_alias,
)

__getattr__ = _deprecated_defaults_alias(__name__, RANDOMIZATION_REGISTRY)
