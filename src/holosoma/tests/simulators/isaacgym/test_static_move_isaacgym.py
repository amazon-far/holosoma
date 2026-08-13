"""Live IsaacGym RUNTIME static-body relocation (GPU), one sim per subprocess.

Runs the static_move_assert.py harness in its own process (IsaacGym segfaults on a second
gymapi sim per process). The harness drops a free box onto a STATIC pillar, settles, then
teleports the pillar away via the unified set_static_body_pose API and asserts (a) the box
rested ON the pillar (static is solid at its spawned pose), (b) get_actor_states reports the
relocated pose (the kinematic write took on the live backend), and (c) the box then falls
below the pillar's old top (the move actually removed the support).

Unmarked + ``importorskip("isaacgym")``/CUDA-gated so the IsaacGym CI job (``-m "not
isaacsim"``) collects it and it skips cleanly elsewhere. The IsaacSim analogue is in
../isaacsim/test_static_move_isaacsim.py; the in-process Classic CPU analogue is in
test_static_body_move_classic.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("isaacgym")
from holosoma.utils.safe_torch_import import torch

if not torch.cuda.is_available():
    pytest.skip("IsaacGym requires a CUDA device", allow_module_level=True)

from tests.simulators._run_harness import run_harness

# The harness builds one sim and exits 0 on success; run it per scenario in its own process.
_HARNESS = Path(__file__).resolve().parents[1] / "static_move_assert.py"


@pytest.mark.parametrize("num_envs", ["1", "4"])
def test_static_move(num_envs):
    run_harness(
        _HARNESS,
        "--simulator",
        "isaacgym",
        "--num-envs",
        num_envs,
        label=f"isaacgym/static-move (num_envs={num_envs})",
        timeout=600,
    )
