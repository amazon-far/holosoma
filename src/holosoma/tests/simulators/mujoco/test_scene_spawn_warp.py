"""Live MuJoCo Warp (GPU, multi-env) scene-spawn matrix via the shared harness.

Routes the Warp backend through the SAME scene_spawn_assert.py harness the Isaac backends use,
so the full [x,y,z] cross-backend offset check and the per-env origin-spread check run on mjwarp.
(The in-process test_actor_state_gpu.py covers per-env state + the multibody offset, but only with
zero env_origins and on the x-axis.)

  - num_envs=4: each free object at its OWN env_origin, 1->N offset in EVERY env.
  - num_envs=1: only the scenes not in the multi-env list. The harness asserts every check in every
    env and env 0 sits at the origin, so a num_envs=4 cell covers its num_envs=1 counterpart.

MuJoCo ClassicBackend (CPU) cannot do >1 env, so multi-env is mjwarp-only. CUDA-gated; one
sim per case (the harness builds + tears down a full sim per run), in the shared harness worker.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")

# MuJoCo WarpBackend (CUDA) only.
pytestmark = pytest.mark.mujoco_warp

if not torch.cuda.is_available():
    pytest.skip("Warp multi-env scene-spawn requires a CUDA device", allow_module_level=True)

from tests.simulators._run_harness import run_harness  # noqa: E402

_HARNESS = Path(__file__).resolve().parents[1] / "scene_spawn_assert.py"


def _run(scene, num_envs, steps=None):
    args = ["--simulator", "mjwarp", "--scene", scene, "--num-envs", str(num_envs)]
    if steps is not None:
        args += ["--steps", str(steps)]
    run_harness(
        _HARNESS,
        *args,
        label=f"mjwarp/{scene} (num_envs={num_envs})",
        timeout=600,
        reuse_process=True,
    )


@pytest.mark.parametrize("scene", ["static-box", "physics-box"])
def test_scene_spawn(scene):
    # physics-box exercises the per-object physics override read-back (live body mass + geom
    # sliding friction vs. config). Both presets ship URDF (no XML); MuJoCo loads URDF when no XML
    # is given (select_asset_format prefers xml then urdf), as velocity-box already does here.
    _run(scene, num_envs=1)


@pytest.mark.parametrize("scene", ["g1-largebox", "velocity-box", "multibody", "multibody-override"])
def test_scene_spawn_multi_env(scene):
    _run(scene, num_envs=4)
