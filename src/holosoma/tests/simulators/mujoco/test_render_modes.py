"""MuJoCo render-mode contract: the run loop must step + render in BOTH modes.

The real run_sim loop calls sim.render() every step regardless of headless, so the
scene_spawn_assert.py harness does too. This is the path the spawn-matrix tests never
touched (they step but never render), and where a headless render that dereferences a
None viewer would crash. One scenario, two modes, both MuJoCo backends:

  - mujoco  (ClassicBackend, CPU)  — always runs
  - mjwarp  (WarpBackend, GPU)     — CUDA-gated

headless=true runs everywhere (CI included): render() must be a safe no-op (no viewer).
headless=false opens a real window and drives the headful render, so it needs a display
and is skipped when DISPLAY is unset (CI). Headless cases share the harness worker; headful
cases each get their own subprocess.
mjwarp headless is covered by test_scene_spawn_warp's multi-env g1-largebox cell, so only
its headful mode runs here.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")

from tests.simulators._run_harness import run_harness  # noqa: E402

_HARNESS = Path(__file__).resolve().parents[1] / "scene_spawn_assert.py"


def _cases():
    for headless in ("true", "false"):
        yield pytest.param("mujoco", headless, marks=pytest.mark.mujoco_classic, id=f"mujoco-{headless}")
    if torch.cuda.is_available():
        yield pytest.param("mjwarp", "false", marks=pytest.mark.mujoco_warp, id="mjwarp-false")


@pytest.mark.parametrize(("simulator", "headless"), list(_cases()))
def test_render_headless_and_headful(simulator, headless):
    """Step + render a free-box scene; headless must not crash, headful must draw."""
    if headless == "false" and not os.environ.get("DISPLAY"):
        pytest.skip("headful render needs a display (DISPLAY unset)")
    run_harness(
        _HARNESS,
        "--simulator",
        simulator,
        "--scene",
        "g1-largebox",
        "--headless",
        headless,
        label=f"{simulator} render (headless={headless})",
        timeout=600,
        reuse_process=headless == "true",
    )
