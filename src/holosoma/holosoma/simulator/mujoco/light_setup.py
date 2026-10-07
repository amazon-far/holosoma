"""MuJoCo scene lighting.

Adds the scene's lights (``SceneConfig.lights``) to the world spec. MuJoCo supports directional and
point lights; a light's ``color`` becomes its diffuse (other types are skipped with a warning). A
``DefaultLightConfig`` keeps MuJoCo's camera headlight on and authors no light of its own; an empty
rig turns the headlight off (only the scene's baked lights remain); a populated rig turns the
headlight off and authors exactly those lights.
"""

from __future__ import annotations

from collections.abc import Mapping

import mujoco
from loguru import logger

from holosoma.config_types.light import (
    DefaultLightConfig,
    DistantLightConfig,
    LightConfigBase,
    SphereLightConfig,
    kelvin_to_rgb,
    light_ray_direction,
)


def build_lights(spec: mujoco.MjSpec, lights: Mapping[str, LightConfigBase]) -> None:
    """Add the scene's lights to ``spec``. The camera headlight stays on only for the default rig; an
    authored rig turns it off and becomes the sole source."""
    has_default = any(isinstance(cfg, DefaultLightConfig) for cfg in lights.values())
    spec.visual.headlight.active = 1 if has_default else 0
    for name, cfg in lights.items():
        if cfg.enabled and not isinstance(cfg, DefaultLightConfig):
            _add_light(spec, name, cfg)


def _add_light(spec: mujoco.MjSpec, name: str, cfg: LightConfigBase) -> None:
    # MuJoCo's light color is its diffuse: resolve the color temperature (if set) and fold in
    # exposure, since MuJoCo has no separate color-temperature or exposure control.
    base = kelvin_to_rgb(cfg.color_temperature_k) if cfg.color_temperature_k is not None else cfg.color
    scale = 2.0**cfg.exposure
    diffuse = [float(c) * scale for c in base]
    if isinstance(cfg, DistantLightConfig):
        spec.worldbody.add_light(
            dir=list(light_ray_direction(cfg.orientation)),
            diffuse=diffuse,
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        )
    elif isinstance(cfg, SphereLightConfig):
        spec.worldbody.add_light(
            pos=list(cfg.position),
            diffuse=diffuse,
            type=mujoco.mjtLightType.mjLIGHT_POINT,
        )
    else:
        logger.warning(f"MuJoCo does not support light '{name}' ({type(cfg).__name__}); skipping")
