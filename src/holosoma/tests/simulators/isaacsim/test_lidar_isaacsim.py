"""Live Isaac Sim mutable-pattern MultiMeshRayCaster LiDAR test."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.isaacsim

pytest.importorskip("isaaclab")
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("IsaacSim requires a CUDA device", allow_module_level=True)

from tests.simulators._run_harness import run_harness  # noqa: E402

_HARNESS = Path(__file__).resolve().parents[1] / "lidar_assert.py"


def test_lidar(tmp_path):
    result_file = tmp_path / "lidar_isaacsim.txt"
    run_harness(
        _HARNESS,
        "--simulator",
        "isaacsim",
        "--num-envs",
        "3",
        "--result-file",
        str(result_file),
        label="isaacsim/lidar",
        timeout=700,
        result_file=result_file,
    )
