"""IsaacGym scene lighting.

Applies the scene's lights (``SceneConfig.lights``) to the sim. IsaacGym supports only global
directional lights (up to four); a light's ``color`` carries its intensity, and other light types are
skipped. The default rig authors IsaacGym's viewer directional light; an empty rig leaves IsaacGym's
built-in lighting; a rig of directional lights authors those. (IsaacGym has no baked-scene lights.)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaacgym import gymapi
from loguru import logger

from holosoma.config_types.light import DefaultLightConfig, DistantLightConfig, kelvin_to_rgb, light_ray_direction

if TYPE_CHECKING:
    from holosoma.simulator.isaacgym.isaacgym import IsaacGym


def build_lights(sim: IsaacGym) -> None:
    """Apply the scene's directional lights: keep the enabled directional entries (at most four), map
    each ``color`` to the light's intensity, and turn each orientation into a ray direction."""
    directional: list[DistantLightConfig | DefaultLightConfig] = []
    for name, cfg in sim.scene_config.lights.items():
        if not cfg.enabled:
            continue
        if isinstance(cfg, (DistantLightConfig, DefaultLightConfig)):
            directional.append(cfg)
        else:
            logger.warning(f"IsaacGym supports only directional lights; skipping '{name}' ({type(cfg).__name__})")
    if len(directional) > 4:
        logger.warning(f"IsaacGym supports at most 4 lights; using the first 4 of {len(directional)}")
        directional = directional[:4]
    for index, cfg in enumerate(directional):
        direction = gymapi.Vec3(*light_ray_direction(cfg.orientation))
        # IsaacGym's light color carries brightness: resolve color temperature (if set) and fold in
        # exposure, since it has no separate color-temperature or exposure control.
        base = kelvin_to_rgb(cfg.color_temperature_k) if cfg.color_temperature_k is not None else cfg.color
        scale = 2.0**cfg.exposure
        color = gymapi.Vec3(*(c * scale for c in base))
        sim.gym.set_light_parameters(sim.sim, index, color, gymapi.Vec3(0.0, 0.0, 0.0), direction)
