"""Live IsaacGym NON-CONTIGUOUS env_ids subset write (GPU), one sim per subprocess.

Runs subset_write_assert.py: relocate a static pillar in ONLY envs [0, 2] of 4 to per-env-distinct
world targets, assert written envs land at their own target AND unwritten envs [1, 3] are untouched.
Guards the strided-env-subset write path through the unified WORLD-frame set_actor_states (after the
frame unification, IsaacGym writes world poses straight through with no env_origins re-add).

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

_HARNESS = Path(__file__).resolve().parents[1] / "subset_write_assert.py"


def test_subset_write():
    run_harness(
        _HARNESS,
        "--simulator",
        "isaacgym",
        "--num-envs",
        "4",
        label="isaacgym/subset-write (num_envs=4)",
        timeout=600,
    )
