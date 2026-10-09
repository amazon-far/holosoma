"""Light presets, registered so a scene's ``lights`` entries can pick a type by name.

Each preset is one light type (dome / distant / sphere / rect); a scene preset assembles them into
``SceneConfig.lights``. Mirrors ``config_values/plugin.py``.
"""

from holosoma.config_types.light import (
    DefaultLightConfig,
    DistantLightConfig,
    DomeLightConfig,
    LightConfigBase,
    RectLightConfig,
    SphereLightConfig,
)
from holosoma.utils.config_registry import ConfigRegistry

LIGHT_REGISTRY = ConfigRegistry(LightConfigBase, group="holosoma.config.light")

default = LIGHT_REGISTRY.add("default", DefaultLightConfig())
dome = LIGHT_REGISTRY.add("dome", DomeLightConfig())
distant = LIGHT_REGISTRY.add("distant", DistantLightConfig())
sphere = LIGHT_REGISTRY.add("sphere", SphereLightConfig())
rect = LIGHT_REGISTRY.add("rect", RectLightConfig())
