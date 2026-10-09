"""IsaacGym lighting: ``light_setup.build_lights`` applies the scene's global directional lights.

IsaacGym has only global directional lights (at most four), so ``build_lights`` keeps the enabled
``DistantLightConfig`` entries, skips every other type, caps the count at four, maps ``color`` to the
light's intensity, and turns each orientation quaternion into a ray direction. This drives the real
function against a recording stand-in for the sim — no sim is created, so it needs the isaacgym
runtime (``gymapi.Vec3``) but no GPU.
"""

from __future__ import annotations

import types

import pytest

# Guarded so the mujoco/isaacsim/no_sim CI jobs (no isaacgym) skip this module at collection.
pytest.importorskip("isaacgym")

from isaacgym import gymapi

from holosoma.config_types.light import DEFAULT_LIGHT_KEY, DefaultLightConfig, DistantLightConfig, SphereLightConfig
from holosoma.simulator.isaacgym import light_setup


class _RecordingGym:
    """Captures every ``set_light_parameters`` call instead of touching a real sim."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, gymapi.Vec3, gymapi.Vec3, gymapi.Vec3]] = []

    def set_light_parameters(self, sim, index, intensity, ambient, direction):
        self.calls.append((index, intensity, ambient, direction))


def _run(lights: dict) -> list[tuple[int, gymapi.Vec3, gymapi.Vec3, gymapi.Vec3]]:
    gym = _RecordingGym()
    fake = types.SimpleNamespace(
        gym=gym,
        sim=object(),  # opaque handle; the recording gym ignores it
        scene_config=types.SimpleNamespace(lights=lights),
    )
    light_setup.build_lights(fake)  # type: ignore[arg-type]  # stand-in for the IsaacGym sim
    return gym.calls


def test_applies_directional_skips_others_and_maps_color() -> None:
    # Orientation is a 90° rotation about X: straight-down (-Z) becomes +Y, checked independently of
    # the production quaternion math.
    lights = {
        "sun": DistantLightConfig(color=(0.9, 0.9, 0.8), orientation=(0.70710678, 0.70710678, 0.0, 0.0)),
        "lamp": SphereLightConfig(position=(1.0, 2.0, 3.0)),  # not directional -> skipped
        "off": DistantLightConfig(enabled=False),  # disabled -> skipped
    }
    calls = _run(lights)

    assert len(calls) == 1  # only the enabled directional light
    index, intensity, ambient, direction = calls[0]
    assert index == 0
    # color -> intensity.
    assert (intensity.x, intensity.y, intensity.z) == pytest.approx((0.9, 0.9, 0.8))
    # ambient is unused (black).
    assert (ambient.x, ambient.y, ambient.z) == pytest.approx((0.0, 0.0, 0.0))
    # orientation rotates -Z to +Y.
    assert (direction.x, direction.y, direction.z) == pytest.approx((0.0, 1.0, 0.0), abs=1e-6)


def test_caps_at_four_directional_lights() -> None:
    lights = {f"sun{i}": DistantLightConfig() for i in range(5)}
    calls = _run(lights)

    assert len(calls) == 4  # fifth directional light dropped
    assert [index for index, *_ in calls] == [0, 1, 2, 3]


def test_default_rig_authors_one_directional_light() -> None:
    # The default_light sentinel becomes IsaacGym's viewer directional light at slot 0, pointing down.
    calls = _run({DEFAULT_LIGHT_KEY: DefaultLightConfig()})
    assert len(calls) == 1
    index, _intensity, _ambient, direction = calls[0]
    assert index == 0
    assert (direction.x, direction.y, direction.z) == pytest.approx((0.0, 0.0, -1.0), abs=1e-6)


def test_empty_rig_sets_no_lights() -> None:
    assert _run({}) == []
