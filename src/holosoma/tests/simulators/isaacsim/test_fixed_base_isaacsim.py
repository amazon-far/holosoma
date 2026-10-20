"""Live IsaacSim fixed-base robot checks (GPU), one sim per subprocess.

Runs the fixed_base_robot_assert.py harness (the ``test-arm-fixed`` 3-DOF arm with a prismatic
joint, ``fix_base_link=True``): spawn pose == env origin + configured pose with zero velocity, the
base stays put under gravity while the arm moves, root writes teleport the base like a static
scene object (velocity ignored), DOF writes and prismatic torques take effect, and N envs spread
to their origins. Marked ``isaacsim`` and ``importorskip("isaaclab")``/CUDA-gated like the other
IsaacSim harness wrappers.
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

_HARNESS = Path(__file__).resolve().parents[1] / "fixed_base_robot_assert.py"


@pytest.mark.parametrize("num_envs", ["1", "4"])
def test_fixed_base_robot(num_envs: str) -> None:
    run_harness(
        _HARNESS,
        "--simulator",
        "isaacsim",
        "--num-envs",
        num_envs,
        label=f"isaacsim/fixed-base (num_envs={num_envs})",
        timeout=900,
    )
