"""Live IsaacGym run of the YAM joint-trajectory harness (see yam_trajectory_assert.py), one sim per subprocess.

Unmarked + ``importorskip("isaacgym")``/CUDA-gated like the other IsaacGym harness wrappers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("isaacgym")
from holosoma.utils.safe_torch_import import torch

if not torch.cuda.is_available():
    pytest.skip("IsaacGym requires a CUDA device", allow_module_level=True)

from tests.simulators._run_harness import run_harness

_HARNESS = Path(__file__).resolve().parents[1] / "yam_trajectory_assert.py"


def test_yam_trajectory(tmp_path: Path) -> None:
    result_file = tmp_path / "yam_trajectory_isaacgym.txt"
    run_harness(
        _HARNESS,
        "--simulator",
        "isaacgym",
        "--result-file",
        str(result_file),
        label="isaacgym/yam-trajectory",
        timeout=900,
        result_file=result_file,
    )
