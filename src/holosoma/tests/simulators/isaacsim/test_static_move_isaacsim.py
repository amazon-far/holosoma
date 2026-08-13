"""Live IsaacSim RUNTIME static-body relocation (GPU), one sim per subprocess.

Runs the static_move_assert.py harness in its own process (IsaacSim's SimulationContext is a
process singleton — one sim per process). The harness drops a free box onto a STATIC pillar,
settles, then teleports the pillar away via the unified set_static_body_pose API and asserts
(a) the box rested ON the pillar (static is solid at its spawned pose), (b) get_actor_states
reports the relocated pose (the kinematic write took on the live backend), and (c) the box
then falls below the pillar's old top (the move actually removed the support).

Marked ``isaacsim`` so only the IsaacSim CI job (``-m isaacsim``) collects it.
``importorskip("isaaclab")``/CUDA-gated so direct/unfiltered
runs skip cleanly where IsaacSim is absent. The IsaacGym analogue is in
../isaacgym/test_static_move_isaacgym.py; the in-process Classic CPU analogue is in
../mujoco/test_static_body_move_classic.py.
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

# The harness builds one sim and exits 0 on success; run it per scenario in its own process.
_HARNESS = Path(__file__).resolve().parents[1] / "static_move_assert.py"


@pytest.mark.parametrize("num_envs", ["1", "4"])
def test_static_move(num_envs):
    run_harness(
        _HARNESS,
        "--simulator",
        "isaacsim",
        "--num-envs",
        num_envs,
        label=f"isaacsim/static-move (num_envs={num_envs})",
        timeout=900,
    )
