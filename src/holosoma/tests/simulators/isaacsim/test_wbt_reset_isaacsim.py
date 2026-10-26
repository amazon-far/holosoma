"""WBT reset_all() target regression on Isaac Sim."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.isaacsim

pytest.importorskip("isaaclab")
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("Isaac Sim requires CUDA", allow_module_level=True)

from tests.simulators._run_harness import run_harness  # noqa: E402

_HARNESS = Path(__file__).resolve().parents[1] / "wbt_reset_assert.py"


def test_wbt_reset_isaacsim(tmp_path):
    result_file = tmp_path / "isaacsim-wbt-reset.ok"
    run_harness(
        _HARNESS,
        "--simulator",
        "isaacsim",
        "--result-file",
        str(result_file),
        label="isaacsim/wbt-reset",
        timeout=900,
        result_file=result_file,
    )
