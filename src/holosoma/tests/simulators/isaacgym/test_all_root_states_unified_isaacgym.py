"""Live IsaacGym unified all_root_states (robot+object) (GPU), one sim per subprocess.

Runs all_root_states_unified_assert.py: index/read/write/clone/sentinel checks for names including
BOTH "robot" and the free object, in every env.

Unmarked (conftest's directory rule applies ``isaacgym``; the CI job selects ``-m isaacgym``)
collects it and it skips cleanly elsewhere.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("isaacgym")
from holosoma.utils.safe_torch_import import torch

if not torch.cuda.is_available():
    pytest.skip("IsaacGym requires a CUDA device", allow_module_level=True)

from tests.simulators._run_harness import run_harness

_HARNESS = Path(__file__).resolve().parents[1] / "all_root_states_unified_assert.py"


def test_all_root_states_unified():
    run_harness(
        _HARNESS,
        "--simulator",
        "isaacgym",
        "--num-envs",
        "4",
        label="isaacgym/all-root-states-unified (num_envs=4)",
        timeout=600,
    )
