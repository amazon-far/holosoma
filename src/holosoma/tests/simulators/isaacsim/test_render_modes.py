"""IsaacSim render-mode contract: the run loop must step + render in BOTH modes.

The scene_spawn_assert.py harness now drives render() every step (as the real run_sim loop
does). IsaacSim's render() goes through self.sim.render() and never touches self.viewer, so
it is already headless-safe — this test locks that in alongside the IsaacGym/MuJoCo backends
so no future change reintroduces a headless-render crash.

  headless=true  — runs in CI (no display); render() must not crash.
  headless=false — opens the Kit window and draws; needs a display, skipped when DISPLAY unset.

Marked ``isaacsim`` so the IsaacSim CI job (``-m "isaacsim"``) collects it; one sim per
subprocess (IsaacSim's SimulationContext is a process singleton).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.isaacsim

pytest.importorskip("isaaclab")
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("IsaacSim requires a CUDA device", allow_module_level=True)

from tests.simulators._run_harness import run_harness  # noqa: E402

_HARNESS = Path(__file__).resolve().parents[1] / "scene_spawn_assert.py"


@pytest.mark.parametrize("headless", ["true", "false"])
def test_render_headless_and_headful(headless):
    """Step + render a free-box scene; headless must not crash, headful must draw."""
    if headless == "false" and not os.environ.get("DISPLAY"):
        pytest.skip("headful render needs a display (DISPLAY unset)")
    run_harness(
        _HARNESS,
        "--simulator",
        "isaacsim",
        "--scene",
        "g1-largebox",
        "--headless",
        headless,
        label=f"isaacsim render (headless={headless})",
        timeout=900,
    )
