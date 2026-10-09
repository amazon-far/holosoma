"""Dependency-light conversion from Holosoma camera config to Isaac Lab spawn config."""

from __future__ import annotations

import math
from typing import Any

from holosoma.config_types.sensor import (
    ISAACSIM_FISHEYE_CONFIG_FIELDS,
    CameraSensorConfig,
)


def camera_spawn_cfg_for(cam: CameraSensorConfig, sim_utils: Any) -> Any:
    """Build the configured native Isaac Lab pinhole or fisheye camera.

    ``sim_utils`` is injected by :mod:`sensor_setup` so this conversion remains importable for
    pure tests without loading Isaac Sim or requiring an AppLauncher.
    """
    isaacsim_cfg = cam.isaacsim
    lens_kwargs: dict[str, Any] = {
        "focal_length": isaacsim_cfg.focal_length,
        "f_stop": isaacsim_cfg.f_stop,
        "focus_distance": isaacsim_cfg.focus_distance,
        "clipping_range": (cam.near, cam.far),
    }

    projection_type = isaacsim_cfg.projection_type
    if projection_type != "pinhole":
        fisheye_cfg = isaacsim_cfg.fisheye
        fisheye_kwargs = {f"fisheye_{field}": getattr(fisheye_cfg, field) for field in ISAACSIM_FISHEYE_CONFIG_FIELDS}
        return sim_utils.FisheyeCameraCfg(
            projection_type=projection_type,
            **lens_kwargs,
            **fisheye_kwargs,
        )

    v_aperture = 2.0 * isaacsim_cfg.focal_length * math.tan(math.radians(cam.vertical_fov) / 2)
    h_aperture = v_aperture * (cam.width / cam.height)
    return sim_utils.PinholeCameraCfg(
        vertical_aperture=v_aperture,
        horizontal_aperture=h_aperture,
        **lens_kwargs,
    )
