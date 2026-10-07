"""Arbitrary-frame odometry regression on Isaac Gym."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("isaacgym")
from holosoma.utils.safe_torch_import import torch

if not torch.cuda.is_available():
    pytest.skip("Isaac Gym requires CUDA", allow_module_level=True)

from tests.simulators._run_harness import run_harness

_HARNESS = Path(__file__).resolve().parents[1] / "odometry_frame_assert.py"


def test_odometry_frame_isaacgym(tmp_path):
    result_file = tmp_path / "isaacgym-odometry.ok"
    run_harness(
        _HARNESS,
        "--simulator",
        "isaacgym",
        "--result-file",
        str(result_file),
        label="isaacgym/arbitrary-frame-odometry",
        timeout=600,
        result_file=result_file,
    )
