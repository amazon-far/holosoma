"""MuJoCo lighting: ``light_setup.build_lights`` adds the scene's lights to the world spec.

Builds only the ``MjSpec`` (no sim launch), so it is fast. ``MujocoSceneManager`` still owns the
render visuals (``apply_render``).
"""

from __future__ import annotations

import pytest

# Collection imports every test module regardless of the -m marker, so guard the mujoco-only imports:
# in the isaacgym/isaacsim CI jobs (no mujoco installed) this skips the module instead of erroring.
pytest.importorskip("mujoco")

import mujoco

import holosoma.config_values.run_sim as run_sim_values
from holosoma.config_types.light import (
    DEFAULT_LIGHT_KEY,
    DefaultLightConfig,
    DistantLightConfig,
    RectLightConfig,
    SphereLightConfig,
)
from holosoma.simulator.mujoco import light_setup
from holosoma.simulator.mujoco.scene_manager import MujocoSceneManager

# CPU MjSpec build with the classic-backend config (no Warp runtime) — mark it accordingly, matching
# the other spec-level mujoco tests. ``mujoco_classic`` implies the ``mujoco`` umbrella (see conftest).
pytestmark = pytest.mark.mujoco_classic


def _manager() -> MujocoSceneManager:
    # MujocoSceneManager annotates its arg as SimulatorConfig but reads .sim.fps off the inner
    # SimulatorInitConfig (.config); the real backend passes it with the same type: ignore.
    return MujocoSceneManager(run_sim_values.mujoco.config)  # type: ignore[arg-type]


def test_add_lighting_builds_supported_and_skips_unsupported() -> None:
    mgr = _manager()
    # Orientation is a 90° rotation about X: straight-down (-Z) becomes +Y, which the assertion checks
    # independently of the production math.
    lights = {
        "sun": DistantLightConfig(color=(0.9, 0.9, 0.8), orientation=(0.70710678, 0.70710678, 0.0, 0.0)),
        # High intensity, but MuJoCo lights have no scalar-intensity analogue: color alone sets diffuse,
        # so the diffuse assertion below must stay (1, 1, 1), unscaled by intensity.
        "lamp": SphereLightConfig(color=(1.0, 1.0, 1.0), intensity=5000.0, position=(1.0, 2.0, 3.0)),
        "panel": RectLightConfig(width=0.2, height=0.3),  # MuJoCo has no rect light -> skipped
        "off": SphereLightConfig(enabled=False),  # disabled -> skipped
    }
    light_setup.build_lights(mgr.world_spec, lights)

    added = list(mgr.world_spec.worldbody.lights)
    assert len(added) == 2  # sun (directional) + lamp (point); panel + off skipped
    directional = next(light for light in added if light.type == mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    point = next(light for light in added if light.type == mujoco.mjtLightType.mjLIGHT_POINT)

    # Sphere -> point light at the configured position; color becomes diffuse (intensity is ignored).
    assert list(point.pos) == pytest.approx([1.0, 2.0, 3.0])
    assert list(point.diffuse) == pytest.approx([1.0, 1.0, 1.0])  # not scaled by intensity=5000

    # Distant -> directional light; color becomes diffuse; the orientation rotates -Z to +Y.
    assert list(directional.diffuse) == pytest.approx([0.9, 0.9, 0.8])
    assert list(directional.dir) == pytest.approx([0.0, 1.0, 0.0], abs=1e-6)


def test_apply_render_sets_headlight_and_haze() -> None:
    mgr = _manager()
    mgr.apply_render()
    visual = mgr.world_spec.visual
    # Every field apply_render sets, so a dropped or wrong assignment is caught.
    assert list(visual.headlight.diffuse) == pytest.approx([0.6, 0.6, 0.6])
    assert list(visual.headlight.ambient) == pytest.approx([0.4, 0.4, 0.4])
    assert list(visual.headlight.specular) == pytest.approx([0.0, 0.0, 0.0])
    assert visual.global_.azimuth == pytest.approx(-130.0)
    assert visual.global_.elevation == pytest.approx(-20.0)
    assert list(visual.rgba.haze) == pytest.approx([0.15, 0.25, 0.35, 1.0])


def test_default_light_keeps_headlight_and_authors_nothing() -> None:
    mgr = _manager()
    light_setup.build_lights(mgr.world_spec, {DEFAULT_LIGHT_KEY: DefaultLightConfig()})
    # The default rig keeps MuJoCo's headlight and authors no worldbody light (the default sentinel
    # maps to the headlight, not a MuJoCo light element).
    assert list(mgr.world_spec.worldbody.lights) == []
    assert mgr.world_spec.visual.headlight.active == 1


def test_empty_rig_turns_headlight_off() -> None:
    mgr = _manager()
    light_setup.build_lights(mgr.world_spec, {})
    # No default rig: the headlight is off, leaving only the scene's baked lights.
    assert list(mgr.world_spec.worldbody.lights) == []
    assert mgr.world_spec.visual.headlight.active == 0


def test_populated_rig_turns_headlight_off() -> None:
    mgr = _manager()
    light_setup.build_lights(mgr.world_spec, {"sun": DistantLightConfig()})
    # An authored rig is the sole source, so the headlight is off.
    assert mgr.world_spec.visual.headlight.active == 0
    assert len(list(mgr.world_spec.worldbody.lights)) == 1
