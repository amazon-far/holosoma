"""Live Isaac Sim check that served state is current at every physics substep (state_freshness_assert.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.isaacsim

pytest.importorskip("isaaclab")
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("IsaacSim requires a CUDA device", allow_module_level=True)

from tests.simulators._run_harness import run_harness  # noqa: E402

_HARNESS = Path(__file__).resolve().parents[1] / "state_freshness_assert.py"


def test_state_freshness_isaacsim(tmp_path):
    result_file = tmp_path / "result.txt"
    run_harness(
        _HARNESS,
        "--simulator",
        "isaacsim",
        "--num-envs",
        "4",
        "--result-file",
        str(result_file),
        label="isaacsim/state-freshness",
        timeout=900,
        result_file=result_file,
    )
