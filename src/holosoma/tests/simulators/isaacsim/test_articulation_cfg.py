"""Grouped actuator and contact-sensor cfg invariants in a minimal Isaac Sim process."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.isaacsim

pytest.importorskip("isaaclab")

from tests.simulators._run_harness import run_harness  # noqa: E402

_HARNESS = Path(__file__).resolve().parents[1] / "isaacsim_articulation_cfg_assert.py"


def test_articulation_cfg_invariants(tmp_path):
    result_file = tmp_path / "articulation_cfg_result.txt"
    run_harness(
        _HARNESS,
        "--result-file",
        str(result_file),
        label="IsaacSim articulation cfg invariants",
        timeout=600,
        result_file=result_file,
    )
