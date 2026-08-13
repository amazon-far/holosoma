"""Live IsaacSim scene-spawn matrix (GPU), one sim per subprocess.

Each test runs the scene_spawn_assert.py harness in its own process (IsaacSim's
SimulationContext is a process singleton — one sim per process), covering every asset
shape IsaacSim supports — it loads USD and URDF:

  - 1->1 free rigid object, USD            (usd-box)
  - 1->1 free rigid object, URDF           (g1-largebox)
  - 1->1 static rigid object, USD          (static-box-usd)
  - 1->1 static rigid object, URDF         (static-box)
  - 1->1 free object w/ initial velocity   (velocity-box)
  - 1->N scene file, file-default types    (multibody, USD path)
  - 1->N scene file, per-object override   (multibody-override, USD path)

Marked ``isaacsim`` so only the IsaacSim CI job (``-m isaacsim``) collects it.
``importorskip("isaaclab")``/CUDA-gated so direct/
unfiltered runs skip cleanly where IsaacSim is absent. The harness asserts free bodies
fall, static bodies hold, get_actor_states shape, and (scene files) the body-to-body offset.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.isaacsim

pytest.importorskip("isaaclab")
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("IsaacSim requires a CUDA device", allow_module_level=True)

from tests.simulators._run_harness import run_harness  # noqa: E402

# The harness builds one sim and exits 0 on success; run it per scenario in its own process.
_HARNESS = Path(__file__).resolve().parents[1] / "scene_spawn_assert.py"


@pytest.mark.parametrize(
    "scene",
    [
        "usd-box",
        "g1-largebox",
        "static-box-usd",
        "static-box",
        "velocity-box",
        "physics-box",
        "multibody",
        "multibody-override",
    ],
)
def test_scene_spawn(scene):
    # physics-box exercises the per-object physics override read-back (live mass + bound-material
    # static friction vs. config). It ships USD + URDF, so it spawns on this backend.
    run_harness(
        _HARNESS,
        "--simulator",
        "isaacsim",
        "--scene",
        scene,
        label=f"isaacsim/{scene}",
        timeout=900,
    )


def test_friction_slide_multi_env():
    # Behavioral friction probe (NOT a value read-back): two boxes pushed identically, differing
    # ONLY in friction, must slide measurably different distances. Longer window than the default
    # 40 steps so the low/high gap clears the probe margin; asserted in every env.
    run_harness(
        _HARNESS,
        "--simulator",
        "isaacsim",
        "--scene",
        "friction-slide",
        "--num-envs",
        "4",
        "--steps",
        "150",
        label="isaacsim/friction-slide (num_envs=4)",
        timeout=900,
    )


# Multi-env (num_envs=4): the harness spreads each env to a distinct origin and asserts
# per-env placement — each free object at its OWN env_origin, and the 1->N body-to-body
# offset in EVERY env. IsaacSim scene objects rely on IsaacLab cloning for per-env spread
# (the registration path repeats the first-env pose), so this is the only check that the
# clones actually land at distinct per-env origins rather than stacking at one world point.
@pytest.mark.parametrize("scene", ["usd-box", "velocity-box", "multibody", "multibody-override"])
def test_scene_spawn_multi_env(scene):
    run_harness(
        _HARNESS,
        "--simulator",
        "isaacsim",
        "--scene",
        scene,
        "--num-envs",
        "4",
        label=f"isaacsim/{scene} (num_envs=4)",
        timeout=900,
    )
