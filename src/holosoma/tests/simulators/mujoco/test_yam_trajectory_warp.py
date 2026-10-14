"""Live MuJoCo Warp (GPU) run of the YAM joint-trajectory harness (see yam_trajectory_assert.py).

CUDA-gated; one sim per subprocess. The in-process Classic analogue is test_yam_trajectory_classic.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")

pytestmark = pytest.mark.mujoco_warp

if not torch.cuda.is_available():
    pytest.skip("Warp run requires a CUDA device", allow_module_level=True)

from tests.simulators._run_harness import run_harness  # noqa: E402

_HARNESS = Path(__file__).resolve().parents[1] / "yam_trajectory_assert.py"


def test_yam_trajectory_warp(tmp_path: Path) -> None:
    result_file = tmp_path / "yam_trajectory_mjwarp.txt"
    run_harness(
        _HARNESS,
        "--simulator",
        "mjwarp",
        "--result-file",
        str(result_file),
        label="mjwarp/yam-trajectory",
        timeout=900,
        result_file=result_file,
    )
