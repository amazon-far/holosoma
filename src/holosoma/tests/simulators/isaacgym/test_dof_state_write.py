from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("isaacgym")
from holosoma.utils.safe_torch_import import torch

if not torch.cuda.is_available():
    pytest.skip("IsaacGym requires a CUDA device", allow_module_level=True)


from tests.simulators._run_harness import run_harness

_HARNESS = Path(__file__).resolve().parents[1] / "dof_state_write_assert.py"


def test_subset_shaped_robot_dof_write_preserves_unselected_environments():
    run_harness(
        _HARNESS,
        "--device",
        "cuda:0",
        label="isaacgym/subset-shaped robot DOF write",
        timeout=600,
    )
