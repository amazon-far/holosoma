"""Live IsaacGym fixed-base robot checks (GPU), one sim per subprocess.

Runs the fixed_base_robot_assert.py harness (the ``test-arm-fixed`` 3-DOF arm with a prismatic
joint, ``fix_base_link=True``): spawn pose == env origin + configured pose with zero velocity, the
base stays put under gravity while the arm moves, same-pose root writes are accepted as no-ops
(velocity ignored) while a pose change is refused with NotImplementedError (PhysX keeps a welded
articulation root where it spawned), DOF writes and prismatic torques take effect, and N envs spread
to their origins. Unmarked + ``importorskip("isaacgym")``/CUDA-gated like the other IsaacGym
harness wrappers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("isaacgym")
from holosoma.utils.safe_torch_import import torch

if not torch.cuda.is_available():
    pytest.skip("IsaacGym requires a CUDA device", allow_module_level=True)

from tests.simulators._run_harness import run_harness

_HARNESS = Path(__file__).resolve().parents[1] / "fixed_base_robot_assert.py"


@pytest.mark.parametrize("num_envs", ["1", "4"])
def test_fixed_base_robot(num_envs: str) -> None:
    run_harness(
        _HARNESS,
        "--simulator",
        "isaacgym",
        "--num-envs",
        num_envs,
        label=f"isaacgym/fixed-base (num_envs={num_envs})",
        timeout=600,
    )
