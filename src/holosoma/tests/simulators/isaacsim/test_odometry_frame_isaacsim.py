"""Arbitrary-frame odometry regression on Isaac Sim."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.isaacsim

pytest.importorskip("isaaclab")
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("Isaac Sim requires CUDA", allow_module_level=True)

from tests.simulators._run_harness import run_harness  # noqa: E402

_HARNESS = Path(__file__).resolve().parents[1] / "odometry_frame_assert.py"


def test_odometry_frame_isaacsim(tmp_path):
    result_file = tmp_path / "isaacsim-odometry.ok"
    run_harness(
        _HARNESS,
        "--simulator",
        "isaacsim",
        "--result-file",
        str(result_file),
        label="isaacsim/arbitrary-frame-odometry",
        timeout=900,
        result_file=result_file,
    )
