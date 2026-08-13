from holosoma.config_types.logger import DisabledLoggerConfig, WandbLoggerConfig
from holosoma.utils.config_registry import ConfigRegistry

LOGGER_REGISTRY = ConfigRegistry((DisabledLoggerConfig, WandbLoggerConfig), group="holosoma.config.logger")

disabled = LOGGER_REGISTRY.add("disabled", DisabledLoggerConfig())
wandb = LOGGER_REGISTRY.add("wandb", WandbLoggerConfig(mode="online"))
wandb_offline = LOGGER_REGISTRY.add("wandb_offline", WandbLoggerConfig(mode="offline"))

from holosoma.utils.config_registry import (  # noqa: E402
    deprecated_defaults_alias as _deprecated_defaults_alias,
)

__getattr__ = _deprecated_defaults_alias(__name__, LOGGER_REGISTRY)
