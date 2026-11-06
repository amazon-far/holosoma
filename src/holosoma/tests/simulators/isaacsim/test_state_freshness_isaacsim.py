"""Live Isaac Sim check that served state is current at every physics substep (state_freshness_assert.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.isaacsim

pytest.importorskip("isaaclab")
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("IsaacSim requires a CUDA device", allow_module_level=True)

from holosoma.simulator.isaacsim.backends import get_backend  # noqa: E402
from tests.simulators._run_harness import run_harness  # noqa: E402

_HARNESS = Path(__file__).resolve().parents[1] / "state_freshness_assert.py"


@pytest.mark.parametrize(
    "contact_force_mode",
    [
        "normal",
        pytest.param(
            "total",
            marks=pytest.mark.skipif(
                not get_backend().is_isaaclab3, reason="total contact forces require Isaac Lab 3 / Isaac Sim 6.2"
            ),
        ),
    ],
)
def test_state_freshness_isaacsim(tmp_path, contact_force_mode):
    result_file = tmp_path / "result.txt"
    run_harness(
        _HARNESS,
        "--simulator",
        "isaacsim",
        "--num-envs",
        "4",
        "--contact-force-mode",
        contact_force_mode,
        "--result-file",
        str(result_file),
        label=f"isaacsim/state-freshness/{contact_force_mode}",
        timeout=900,
        result_file=result_file,
    )
