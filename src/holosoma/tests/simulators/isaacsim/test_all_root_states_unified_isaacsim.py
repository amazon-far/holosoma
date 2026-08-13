"""Live IsaacSim unified all_root_states (robot+object) (GPU), one sim per subprocess.

Runs all_root_states_unified_assert.py: index/read/write/clone/sentinel checks for names including
BOTH "robot" and the free object, in every env.

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

_HARNESS = Path(__file__).resolve().parents[1] / "all_root_states_unified_assert.py"


def test_all_root_states_unified():
    run_harness(
        _HARNESS,
        "--simulator",
        "isaacsim",
        "--num-envs",
        "4",
        label="isaacsim/all-root-states-unified (num_envs=4)",
        timeout=900,
    )
