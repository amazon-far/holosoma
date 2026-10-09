"""Isaac Sim light authoring.

Builds the scene's lights (``SceneConfig.lights``) as Isaac Sim lights. Dome and distant lights are
global (no position), so they are authored once at ``/World``. Positioned lights (sphere, rect) are
authored under the first environment before cloning, so each environment gets its own copy. Uses Isaac
Lab's light spawners; rectangular area lights (which Isaac Lab has no spawner for) are authored via
``UsdLux``. A light type Isaac Sim cannot build is skipped with a warning.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from loguru import logger

from holosoma.config_types.light import (
    DefaultLightConfig,
    DistantLightConfig,
    DomeLightConfig,
    LightConfigBase,
    RectLightConfig,
    SphereLightConfig,
)
from holosoma.utils.path import resolve_path

if TYPE_CHECKING:
    from holosoma.simulator.isaacsim.isaacsim import IsaacSim

# Global lights (default sun, dome, distant) illuminate the whole stage, so one copy lives at /World.
# Positioned lights (sphere, rect) are authored under the first environment; cloning replicates them per env.
_GLOBAL_LIGHTS_ROOT = "/World/Lights"
_PER_ENV_LIGHTS_ROOT = "/World/envs/env_0/Lights"

# Light types with no position: authored once, not replicated per environment.
_GLOBAL_LIGHT_TYPES = (DefaultLightConfig, DomeLightConfig, DistantLightConfig)

# Angular diameter of the default sun, in degrees (~0.53°, matching the real sun).
_DEFAULT_SUN_ANGLE_DEG = math.degrees(0.0093)


def build_lights(sim: IsaacSim) -> None:
    """Author each enabled light in the scene's rig, global lights once and positioned lights per env.
    The default rig's ``default_light`` is authored as a directional sun."""
    for name, cfg in sim.scene_config.lights.items():
        if cfg.enabled:
            root = _GLOBAL_LIGHTS_ROOT if isinstance(cfg, _GLOBAL_LIGHT_TYPES) else _PER_ENV_LIGHTS_ROOT
            _author_light(f"{root}/{name}", cfg)


def _author_light(prim_path: str, cfg: LightConfigBase) -> None:
    """Author one light at ``prim_path`` for its type; skip (with a warning) unsupported types."""
    import isaaclab.sim as sim_utils

    if isinstance(cfg, DefaultLightConfig):
        # The default rig's directional sun; Isaac Sim headless stages have no built-in light.
        spawn = sim_utils.DistantLightCfg(**_common(cfg), angle=_DEFAULT_SUN_ANGLE_DEG)
        spawn.func(prim_path, spawn, orientation=cfg.orientation)
    elif isinstance(cfg, DomeLightConfig):
        texture_file = resolve_path(cfg.texture_file) if cfg.texture_file is not None else None
        spawn = sim_utils.DomeLightCfg(**_common(cfg), texture_file=texture_file, texture_format=cfg.texture_format)
        spawn.func(prim_path, spawn, orientation=cfg.orientation)
    elif isinstance(cfg, DistantLightConfig):
        # UsdLux/Isaac Lab express the source angular diameter in degrees; the config is radians.
        spawn = sim_utils.DistantLightCfg(**_common(cfg), angle=math.degrees(cfg.angle))
        spawn.func(prim_path, spawn, orientation=cfg.orientation)
    elif isinstance(cfg, SphereLightConfig):
        spawn = sim_utils.SphereLightCfg(**_common(cfg), radius=cfg.radius)
        spawn.func(prim_path, spawn, translation=cfg.position)
    elif isinstance(cfg, RectLightConfig):
        _author_rect_light(prim_path, cfg)
    else:
        logger.warning(f"Isaac Sim does not support light {prim_path!r} ({type(cfg).__name__}); skipping")


def _common(cfg: LightConfigBase) -> dict[str, Any]:
    """The color / intensity / exposure / color-temperature fields every Isaac Lab light spawner takes."""
    kwargs: dict[str, Any] = {"color": cfg.color, "intensity": cfg.intensity, "exposure": cfg.exposure}
    if cfg.color_temperature_k is not None:
        kwargs["enable_color_temperature"] = True
        kwargs["color_temperature"] = cfg.color_temperature_k
    return kwargs


def _author_rect_light(prim_path: str, cfg: RectLightConfig) -> None:
    """Author a rectangular area light via ``UsdLux`` (Isaac Lab has no rectangular-light spawner)."""
    import isaacsim.core.utils.stage as stage_utils
    from pxr import Gf, UsdGeom, UsdLux

    light = UsdLux.RectLight.Define(stage_utils.get_current_stage(), prim_path)
    light.CreateWidthAttr().Set(float(cfg.width))
    light.CreateHeightAttr().Set(float(cfg.height))
    light.CreateNormalizeAttr().Set(bool(cfg.normalize))
    light.CreateIntensityAttr().Set(float(cfg.intensity))
    light.CreateExposureAttr().Set(float(cfg.exposure))
    if cfg.color_temperature_k is not None:
        light.CreateEnableColorTemperatureAttr().Set(True)
        light.CreateColorTemperatureAttr().Set(float(cfg.color_temperature_k))
    else:
        light.CreateColorAttr().Set(Gf.Vec3f(*(float(c) for c in cfg.color)))
    xf = UsdGeom.Xformable(light.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*(float(p) for p in cfg.position)))
    xf.AddOrientOp().Set(Gf.Quatf(*(float(q) for q in cfg.orientation)))
