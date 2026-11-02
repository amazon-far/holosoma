"""Live MuJoCo check that served state is current at every physics substep (state_freshness_assert.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")

from tests.simulators._run_harness import run_harness  # noqa: E402

_HARNESS = Path(__file__).resolve().parents[1] / "state_freshness_assert.py"


@pytest.mark.mujoco_classic
def test_state_freshness_classic(tmp_path):
    result_file = tmp_path / "result.txt"
    run_harness(
        _HARNESS,
        "--simulator",
        "mujoco",
        "--result-file",
        str(result_file),
        label="mujoco/state-freshness (classic)",
        timeout=600,
        result_file=result_file,
    )


@pytest.mark.mujoco_warp
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Warp requires a CUDA device")
def test_state_freshness_warp(tmp_path):
    result_file = tmp_path / "result.txt"
    run_harness(
        _HARNESS,
        "--simulator",
        "mjwarp",
        "--num-envs",
        "4",
        "--result-file",
        str(result_file),
        label="mjwarp/state-freshness",
        timeout=600,
        result_file=result_file,
    )
