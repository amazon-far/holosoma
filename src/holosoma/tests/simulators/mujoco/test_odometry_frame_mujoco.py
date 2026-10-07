"""Arbitrary-frame odometry regression on MuJoCo Classic and Warp."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")

from tests.simulators._run_harness import run_harness  # noqa: E402

_HARNESS = Path(__file__).resolve().parents[1] / "odometry_frame_assert.py"


@pytest.mark.mujoco_classic
def test_odometry_frame_classic(tmp_path):
    result_file = tmp_path / "mujoco-odometry.ok"
    run_harness(
        _HARNESS,
        "--simulator",
        "mujoco",
        "--result-file",
        str(result_file),
        label="mujoco-classic/arbitrary-frame-odometry",
        timeout=400,
        result_file=result_file,
    )


@pytest.mark.mujoco_warp
@pytest.mark.skipif(not torch.cuda.is_available(), reason="MuJoCo Warp requires CUDA")
def test_odometry_frame_warp(tmp_path):
    result_file = tmp_path / "mjwarp-odometry.ok"
    run_harness(
        _HARNESS,
        "--simulator",
        "mjwarp",
        "--result-file",
        str(result_file),
        label="mjwarp/arbitrary-frame-odometry",
        timeout=400,
        result_file=result_file,
    )
