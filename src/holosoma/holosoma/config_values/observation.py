"""Default observation manager configurations."""

from holosoma.config_types.observation import ObservationManagerCfg
from holosoma.config_values.loco.g1.observation import g1_29dof_loco_single_wolinvel
from holosoma.config_values.loco.t1.observation import t1_29dof_loco_single_wolinvel
from holosoma.config_values.wbt.g1.observation import g1_29dof_wbt_observation, g1_29dof_wbt_observation_w_object
from holosoma.utils.config_registry import ConfigRegistry

OBSERVATION_REGISTRY = ConfigRegistry(ObservationManagerCfg, group="holosoma.config.observation")

none = OBSERVATION_REGISTRY.add("none", None)
OBSERVATION_REGISTRY.add("t1_29dof_loco_single_wolinvel", t1_29dof_loco_single_wolinvel)
OBSERVATION_REGISTRY.add("g1_29dof_loco_single_wolinvel", g1_29dof_loco_single_wolinvel)
OBSERVATION_REGISTRY.add("g1_29dof_wbt", g1_29dof_wbt_observation)
OBSERVATION_REGISTRY.add("g1_29dof_wbt_w_object", g1_29dof_wbt_observation_w_object)

from holosoma.utils.config_registry import (  # noqa: E402
    deprecated_defaults_alias as _deprecated_defaults_alias,
)

__getattr__ = _deprecated_defaults_alias(__name__, OBSERVATION_REGISTRY)
