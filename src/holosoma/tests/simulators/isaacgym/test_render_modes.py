"""IsaacGym render-mode contract: the run loop must step + render in BOTH modes.

Regression for the headless-render crash: IsaacGym only assigned self.viewer in
setup_viewer() (skipped when headless), and render() dereferenced it unconditionally, so a
headless run_sim loop died with AttributeError: 'IsaacGym' object has no attribute 'viewer'
on the first render(). The fix gives the base simulator a None viewer by default and makes
render() return early when it is None.

The scene_spawn_assert.py harness now drives render() every step (as the real loop does):
  headless=true  — render() must be a safe no-op (no viewer). Runs in CI (no display).
  headless=false — opens the gym viewer and draws; needs a display, skipped when DISPLAY unset.
One sim per subprocess (IsaacGym segfaults on a second gymapi sim per process).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("isaacgym")
from holosoma.utils.safe_torch_import import torch

if not torch.cuda.is_available():
    pytest.skip("IsaacGym requires a CUDA device", allow_module_level=True)

from tests.simulators._run_harness import run_harness

_HARNESS = Path(__file__).resolve().parents[1] / "scene_spawn_assert.py"


@pytest.mark.parametrize("headless", ["true", "false"])
def test_render_headless_and_headful(headless):
    """Step + render a free-box scene; headless must not crash, headful must draw."""
    if headless == "false" and not os.environ.get("DISPLAY"):
        pytest.skip("headful render needs a display (DISPLAY unset)")
    run_harness(
        _HARNESS,
        "--simulator",
        "isaacgym",
        "--scene",
        "g1-largebox",
        "--headless",
        headless,
        label=f"isaacgym render (headless={headless})",
        timeout=600,
    )
