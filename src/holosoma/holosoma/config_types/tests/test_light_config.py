"""Unit tests for the light configs and the Kelvin->RGB helper (pure; no simulator)."""

from __future__ import annotations

import dataclasses
import json

import pytest

from holosoma.config_types.light import (
    DEFAULT_LIGHT_KEY,
    DefaultLightConfig,
    DistantLightConfig,
    DomeLightConfig,
    LightConfigBase,
    kelvin_to_rgb,
)
from holosoma.config_types.scene import SceneConfig

pytestmark = pytest.mark.no_sim


# ----- kelvin_to_rgb -----


@pytest.mark.parametrize("kelvin", [1000.0, 2000.0, 4000.0, 6600.0, 10000.0, 20000.0, 40000.0])
def test_kelvin_to_rgb_channels_in_unit_range(kelvin: float) -> None:
    rgb = kelvin_to_rgb(kelvin)
    assert len(rgb) == 3
    assert all(0.0 <= c <= 1.0 for c in rgb)


def test_kelvin_to_rgb_warm_is_red_dominant() -> None:
    r, g, b = kelvin_to_rgb(2000.0)
    assert r > g > b  # warm: red strongest, little blue


def test_kelvin_to_rgb_cool_is_blue_dominant() -> None:
    r, _, b = kelvin_to_rgb(15000.0)
    assert b > r  # cool: blue outweighs red


def test_kelvin_to_rgb_neutral_is_near_white() -> None:
    assert all(c >= 0.9 for c in kelvin_to_rgb(6600.0))


def test_kelvin_to_rgb_clamps_out_of_range() -> None:
    assert kelvin_to_rgb(500.0) == kelvin_to_rgb(1000.0)  # below the low bound
    assert kelvin_to_rgb(50000.0) == kelvin_to_rgb(40000.0)  # above the high bound


# ----- DistantLightConfig -----


def test_distant_light_angle_default_is_radians() -> None:
    # The sun's ~0.53 degree angular diameter, expressed in radians.
    assert DistantLightConfig().angle == pytest.approx(0.0093, abs=1e-4)


# ----- SceneConfig.lights semantics -----


def test_scene_lights_default_is_the_default_light_entry() -> None:
    # The default rig is the "default_light" entry, which drives each backend's own default lighting.
    lights = SceneConfig().lights
    assert set(lights) == {DEFAULT_LIGHT_KEY}
    assert isinstance(lights[DEFAULT_LIGHT_KEY], DefaultLightConfig)


@pytest.mark.parametrize(
    "light",
    [DefaultLightConfig(), DomeLightConfig(texture_file="/tmp/env.exr"), DistantLightConfig(angle=0.02)],
)
def test_scene_lights_round_trip_preserves_type(light: LightConfigBase) -> None:
    # A plain dict of a light's fields (as a checkpoint stores it) reconstructs as its own concrete
    # type under the smart union, not as another light type (e.g. a dome must not become the default).
    raw = json.loads(json.dumps(dataclasses.asdict(light)))
    scene = SceneConfig(lights={"probe": raw})
    assert type(scene.lights["probe"]) is type(light)


def test_scene_lights_empty_authors_nothing() -> None:
    # {} => no default_light entry, so backends author nothing (scene-baked lights only).
    assert SceneConfig(lights={}).lights == {}


def test_scene_lights_populated_rig() -> None:
    scene = SceneConfig(lights={"sun": DistantLightConfig(intensity=2000.0)})
    assert DEFAULT_LIGHT_KEY not in scene.lights
    assert isinstance(scene.lights["sun"], DistantLightConfig)
