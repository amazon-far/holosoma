"""Live IsaacSim scene-spawn matrix (GPU).

Each test runs the scene_spawn_assert.py harness on IsaacSim, covering every asset shape IsaacSim
supports — it loads USD and URDF:

  - 1->1 free rigid object, USD            (usd-box)
  - 1->1 free rigid object, URDF           (g1-largebox)
  - 1->1 static rigid object, USD          (static-box-usd)
  - 1->1 static rigid object, URDF         (static-box)
  - 1->1 free object w/ initial velocity   (velocity-box)
  - 1->N scene file, file-default types    (multibody, USD path)
  - 1->N scene file, per-object override   (multibody-override, USD path)
  - launcher settings + default sun enabled/disabled read-back

The spawn scenes run in one shared subprocess and Kit app, each on a fresh stage; the launcher
settings test needs its own launch settings, so it runs in its own process.

Marked ``isaacsim`` so only the IsaacSim CI job (``-m isaacsim``) collects it.
``importorskip("isaaclab")``/CUDA-gated so direct/
unfiltered runs skip cleanly where IsaacSim is absent. The harness asserts free bodies
fall, static bodies hold, get_actor_states shape, and (scene files) the body-to-body offset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.isaacsim

pytest.importorskip("isaaclab")
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("IsaacSim requires a CUDA device", allow_module_level=True)

from tests.simulators._run_harness import run_batched, run_harness  # noqa: E402

_HARNESS = Path(__file__).resolve().parents[1] / "scene_spawn_assert.py"
_TIMEOUT = 1800


def _cell(item):
    """Batch cell for a scene-spawn test item: one subprocess runs every selected scene."""
    if getattr(item, "module", None) is not sys.modules[__name__]:
        return None
    if item.originalname not in ("test_scene_spawn", "test_scene_spawn_multi_env"):
        return None
    args = ["--simulator", "isaacsim", "--scene", item.callspec.params["scene"]]
    if item.originalname == "test_scene_spawn_multi_env":
        args += ["--num-envs", "4"]
    return "scene_spawn/isaacsim", item.name, "scene_spawn_assert", args


# Scenes also in test_scene_spawn_multi_env run only there: the harness asserts every check in every
# env, and env 0 sits at the origin, so the num_envs=4 cell covers the num_envs=1 one.
@pytest.mark.parametrize("scene", ["g1-largebox", "static-box-usd", "static-box", "physics-box"])
def test_scene_spawn(scene, request):
    # physics-box exercises the per-object physics override read-back (live mass + bound-material
    # static friction vs. config). It ships USD + URDF, so it spawns on this backend.
    run_batched(request, _cell, label=f"isaacsim/{scene}", timeout=_TIMEOUT)


def test_launcher_settings_and_default_sun():
    """Read launcher settings and the default scene light back from live IsaacSim."""
    run_harness(
        _HARNESS,
        "--simulator",
        "isaacsim",
        "--scene",
        "g1-largebox",
        "--probe-isaacsim-render-config",
        "--rendering-mode",
        "quality",
        "--kit-arg-sentinel",
        "pr22-audit",
        "--scene-lights",
        "default",
        label="isaacsim/render-config defaults",
        timeout=900,
    )


# Multi-env (num_envs=4): the harness spreads each env to a distinct origin and asserts
# per-env placement — each free object at its OWN env_origin, and the 1->N body-to-body
# offset in EVERY env. IsaacSim scene objects rely on IsaacLab cloning for per-env spread
# (the registration path repeats the first-env pose), so this is the only check that the
# clones actually land at distinct per-env origins rather than stacking at one world point.
@pytest.mark.parametrize("scene", ["usd-box", "velocity-box", "multibody", "multibody-override"])
def test_scene_spawn_multi_env(scene, request):
    run_batched(request, _cell, label=f"isaacsim/{scene} (num_envs=4)", timeout=_TIMEOUT)
