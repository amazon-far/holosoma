"""Live IsaacGym scene-spawn matrix (GPU), one sim per subprocess.

Each test runs the scene_spawn_assert.py harness in its own process (IsaacGym segfaults on
a second gymapi sim per process), covering every asset shape IsaacGym supports — it loads
URDF only:

  - 1->1 free rigid object               (g1-largebox)
  - 1->1 static rigid object (fixed=True) (static-box)
  - 1->1 free object w/ initial velocity  (velocity-box)
  - 1->N scene file, file-default types   (multibody)
  - 1->N scene file, per-object override  (multibody-override)

Unmarked + ``importorskip("isaacgym")``/CUDA-gated so the IsaacGym CI job (``-m "not
isaacsim"``) collects it and it skips cleanly elsewhere. The harness asserts free bodies
fall, static bodies hold, get_actor_states shape, and (for scene files) the body-to-body
offset. The IsaacSim matrix is in ../isaacsim/test_scene_spawn_isaacsim.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("isaacgym")
from holosoma.utils.safe_torch_import import torch

if not torch.cuda.is_available():
    pytest.skip("IsaacGym requires a CUDA device", allow_module_level=True)

from tests.simulators._run_harness import run_harness

# The harness builds one sim and exits 0 on success; run it per scenario in its own process.
_HARNESS = Path(__file__).resolve().parents[1] / "scene_spawn_assert.py"


@pytest.mark.parametrize(
    "scene",
    ["g1-largebox", "static-box", "velocity-box", "physics-box", "multibody", "multibody-override"],
)
def test_scene_spawn(scene):
    # physics-box exercises the per-object physics override read-back (live mass + sliding friction
    # vs. config). It ships URDF (IsaacGym loads URDF only), so it spawns on this backend.
    run_harness(
        _HARNESS,
        "--simulator",
        "isaacgym",
        "--scene",
        scene,
        label=f"isaacgym/{scene}",
        timeout=600,
    )


def test_friction_slide_multi_env():
    # Behavioral friction probe (NOT a value read-back): two boxes pushed identically, differing
    # ONLY in friction, must slide measurably different distances. Longer window than the default
    # 40 steps so the low/high gap clears the probe margin; asserted in every env.
    run_harness(
        _HARNESS,
        "--simulator",
        "isaacgym",
        "--scene",
        "friction-slide",
        "--num-envs",
        "4",
        "--steps",
        "150",
        label="isaacgym/friction-slide (num_envs=4)",
        timeout=600,
    )


# Multi-env (num_envs=4): the harness spreads each env to a distinct origin and asserts
# per-env placement — each free object sits at its OWN env_origin (not stacked), and the
# 1->N body-to-body offset holds in EVERY env. This is the path that exercises IsaacGym's
# per-env env_origin re-add in set_actor_states (a no-op at num_envs=1), so it catches
# per-env offset/stacking regressions the single-env matrix above cannot.
@pytest.mark.parametrize("scene", ["g1-largebox", "velocity-box", "multibody", "multibody-override"])
def test_scene_spawn_multi_env(scene):
    run_harness(
        _HARNESS,
        "--simulator",
        "isaacgym",
        "--scene",
        scene,
        "--num-envs",
        "4",
        label=f"isaacgym/{scene} (num_envs=4)",
        timeout=600,
    )


# Live object physics-DR on IsaacGym: `--object-dr` runs the mass + material DR terms after
# prepare_sim and asserts, against the real gym actor API, that mass increased by the configured
# additive band on every (object, env) and that an object_names subset narrows targeting.
# `free-and-static` gives two free bodies (per-env addressing + subset isolation) plus a fixed
# pillar (must be excluded).
@pytest.mark.parametrize("num_envs", ["1", "4"])
def test_object_dr_dispatch(num_envs):
    run_harness(
        _HARNESS,
        "--simulator",
        "isaacgym",
        "--scene",
        "free-and-static",
        "--num-envs",
        num_envs,
        "--object-dr",
        label=f"isaacgym object-DR (num_envs={num_envs})",
        timeout=600,
    )
