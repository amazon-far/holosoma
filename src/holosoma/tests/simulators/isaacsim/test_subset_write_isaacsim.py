"""Live IsaacSim NON-CONTIGUOUS env_ids subset write (GPU), one sim per subprocess.

Runs subset_write_assert.py: relocate a static pillar in ONLY envs [0, 2] of 4 to per-env-distinct
world targets, assert written envs land at their own target AND unwritten envs [1, 3] are untouched.
Guards the regression that a strided env subset to set_static_body_pose/set_actor_states once
appeared to scramble poses on IsaacSim (root-caused to the pre-unification frame inconsistency, not
the index path; see memory holosoma-subset-env-actor-write-scramble).

Marked ``isaacsim`` so the IsaacSim CI job collects it; ``importorskip("isaaclab")``/CUDA-gated.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.isaacsim

pytest.importorskip("isaaclab")
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("IsaacSim requires a CUDA device", allow_module_level=True)

from tests.simulators._run_harness import run_harness  # noqa: E402

_HARNESS = Path(__file__).resolve().parents[1] / "subset_write_assert.py"


def test_subset_write():
    run_harness(
        _HARNESS,
        "--simulator",
        "isaacsim",
        "--num-envs",
        "4",
        label="isaacsim/subset-write (num_envs=4)",
        timeout=900,
    )
