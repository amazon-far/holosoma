"""Declarative light configuration.

A scene's light rig is ``SceneConfig.lights`` — a dict of named lights, each one of the light-type
configs below (dome / distant / sphere / rect). A scene preset assembles this dict from the per-type
presets in ``config_values/light.py``. Each backend builds the light types it supports and skips the
rest.

Example (in a scene preset)::

    SceneConfig(lights={
        "sun": DistantLightConfig(intensity=2000.0, color_temperature_k=6500.0),
        "lamp": SphereLightConfig(position=(1.0, 0.0, 3.0), intensity=1500.0),
    })

A preset light's fields can then be overridden on the CLI, e.g. ``--scene.lights.lamp.position 1 0 3``.
"""

from __future__ import annotations

import math
from dataclasses import field
from typing import Literal, Union

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from typing_extensions import TypeAlias

from holosoma.config_types.value_types import UnitQuaternionWXYZTuple

# Reject unknown fields so a field placed on the wrong light subclass fails loud at construction.
_FORBID_EXTRA = ConfigDict(extra="forbid")

# Conventional key for the default-lighting entry (a :class:`DefaultLightConfig`) in a rig. The default
# ``SceneConfig.lights`` holds this one entry; an empty ``lights`` authors nothing instead.
DEFAULT_LIGHT_KEY = "default_light"


def light_ray_direction(orientation: UnitQuaternionWXYZTuple) -> tuple[float, float, float]:
    """The ray direction of a light with a validated unit quaternion (w, x, y, z): the default
    straight-down (-Z) emission axis rotated by the quaternion, normalized. Backends that take a
    direction vector (MuJoCo, IsaacGym) build it from this."""
    w, x, y, z = orientation
    dx = -2.0 * (x * z + w * y)
    dy = -2.0 * (y * z - w * x)
    dz = -(1.0 - 2.0 * (x * x + y * y))
    norm = (dx * dx + dy * dy + dz * dz) ** 0.5 or 1.0
    return (dx / norm, dy / norm, dz / norm)


def kelvin_to_rgb(kelvin: float) -> tuple[float, float, float]:
    """Neil Bartlett's blackbody color-temperature to linear RGB approximation (each channel [0, 1]).

    Backends without a native color-temperature control (MuJoCo, IsaacGym) convert
    ``color_temperature_k`` to a light color with this, so it matches Isaac Sim's native handling.
    Continuous over ~1000-40000 K: ~6600 K is near-white, lower is warm, higher is cool."""
    t = min(max(kelvin, 1000.0), 40000.0) / 100.0

    if t <= 66.0:
        r = 1.0
    else:
        x = t - 55.0
        r = 1.38030159086 + 0.00044786845 * x - 0.15785750233 * math.log(x)

    if t <= 10.0:
        g = 0.0
    elif t <= 66.0:
        x = t - 2.0
        g = -0.60884257109 - 0.00174890002 * x + 0.40977318429 * math.log(x)
    else:
        x = t - 50.0
        g = 1.27627220616 + 0.00031150810 * x - 0.11013841706 * math.log(x)

    if t <= 20.0:
        b = 0.0
    elif t <= 66.0:
        x = t - 10.0
        b = -0.99909549742 + 0.00324474355 * x + 0.45364683926 * math.log(x)
    else:
        b = 1.0

    return (min(max(r, 0.0), 1.0), min(max(g, 0.0), 1.0), min(max(b, 0.0), 1.0))


@dataclass(frozen=True, config=_FORBID_EXTRA)
class LightConfigBase:
    """Fields shared by every light type. Instantiate a concrete subclass."""

    enabled: bool = True
    """Author the light. False leaves it out of the rig."""

    intensity: float = 1000.0
    """Light intensity in radiance units (Isaac Sim). MuJoCo/IsaacGym have no radiance scale — their
    brightness comes from ``color`` (and ``exposure``)."""

    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Linear RGB color, used when ``color_temperature_k`` is None."""

    exposure: float = 0.0
    """Exposure in EV. Isaac Sim applies it natively on top of ``intensity``; MuJoCo/IsaacGym fold it
    into the light color (``color * 2 ** exposure``)."""

    color_temperature_k: float | None = None
    """Color temperature in Kelvin; overrides ``color`` when set. Isaac Sim applies it natively;
    MuJoCo/IsaacGym convert it to an RGB color via :func:`kelvin_to_rgb`."""


@dataclass(frozen=True, config=_FORBID_EXTRA)
class DefaultLightConfig(LightConfigBase):
    """Each backend's native default lighting: Isaac Sim a directional sun, IsaacGym its viewer
    directional light, MuJoCo its camera headlight. ``intensity``/``color``/``orientation`` tune the
    backends that author a real light (Isaac Sim, IsaacGym); MuJoCo's headlight ignores them."""

    orientation: UnitQuaternionWXYZTuple = field(default=(1.0, 0.0, 0.0, 0.0), metadata={"validate_default": True})
    """Orientation quaternion (w, x, y, z) of the sun; rotates the default straight-down (-Z) ray."""


@dataclass(frozen=True, config=_FORBID_EXTRA)
class DomeLightConfig(LightConfigBase):
    """Environment dome — a flat color or an HDRI environment map. Isaac Sim only."""

    texture_file: str | None = None
    """HDR/EXR environment-map path. None => a flat dome of ``color``."""

    texture_format: Literal["automatic", "latlong", "mirroredBall", "angular", "cubeMapVerticalCross"] = "latlong"
    """Projection of ``texture_file`` onto the dome."""

    orientation: UnitQuaternionWXYZTuple = field(default=(1.0, 0.0, 0.0, 0.0), metadata={"validate_default": True})
    """Orientation quaternion (w, x, y, z) of the dome."""


@dataclass(frozen=True, config=_FORBID_EXTRA)
class DistantLightConfig(LightConfigBase):
    """Directional (sun-like) light: parallel rays from an orientation, no position."""

    angle: float = 0.0093
    """Angular diameter of the source in radians (0 = hard shadows; the sun ≈ 0.0093, i.e. ~0.53°)."""

    orientation: UnitQuaternionWXYZTuple = field(default=(1.0, 0.0, 0.0, 0.0), metadata={"validate_default": True})
    """Orientation quaternion (w, x, y, z); rotates the default straight-down (-Z) ray direction."""


@dataclass(frozen=True, config=_FORBID_EXTRA)
class SphereLightConfig(LightConfigBase):
    """Finite spherical (point/area) light at a position. Isaac Sim + MuJoCo (as a point light)."""

    radius: float = 0.5
    """Sphere radius in metres (0 ≈ a point light; larger = softer shadows). Isaac Sim."""

    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """World-frame position in metres."""


@dataclass(frozen=True, config=_FORBID_EXTRA)
class RectLightConfig(LightConfigBase):
    """Finite rectangular area light (e.g. a ceiling panel). Isaac Sim only."""

    width: float = 0.5
    height: float = 0.5
    """Panel size in metres."""

    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """World-frame position in metres."""

    orientation: UnitQuaternionWXYZTuple = field(default=(1.0, 0.0, 0.0, 0.0), metadata={"validate_default": True})
    """Orientation quaternion (w, x, y, z). Identity emits straight down (-Z) in a Z-up world."""

    normalize: bool = True
    """Normalize power by area so intensity is independent of panel size."""


# Every concrete light type. ``SceneConfig.lights`` values are typed as this union so a saved config
# reconstructs each light as its concrete type: a plain dict of a dome's fields rebuilds a
# DomeLightConfig rather than the field-less base. Extend it when adding a light type.
LightConfig: TypeAlias = Union[
    DefaultLightConfig, DomeLightConfig, DistantLightConfig, SphereLightConfig, RectLightConfig
]
