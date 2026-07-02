"""Regression test: the object-managers example runs end-to-end.

Runs ``holosoma.examples.object_managers_demo`` on the MuJoCo backends and asserts it exits 0,
so the worked example (scene spawn + object observations + reset/velocity-restore + pose
jitter + cross-backend physics DR, all in one scenario) stays runnable as the code evolves.
The example itself does the per-feature assertions and returns a nonzero code on any failure;
this test just guards that the whole thing still executes.

CPU/ClassicBackend always runs (single env). The Warp GPU path (multi-env) runs only when a
CUDA device is present.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")

from holosoma.examples.object_managers_demo import run_demo  # noqa: E402


@pytest.mark.mujoco_classic
def test_object_managers_demo_mujoco_cpu():
    """The example runs to completion on the MuJoCo classic (CPU) backend, single env."""
    assert run_demo("mujoco", num_envs=1, headless=True) == 0


@pytest.mark.mujoco_warp
@pytest.mark.skipif(not torch.cuda.is_available(), reason="mjwarp multi-env demo requires a CUDA device")
def test_object_managers_demo_mjwarp_multi_env():
    """The example runs to completion on the MuJoCo Warp (GPU) backend with multiple envs —
    the path where per-env placement, jitter, and DR are actually distinct per environment."""
    assert run_demo("mjwarp", num_envs=4, headless=True) == 0
