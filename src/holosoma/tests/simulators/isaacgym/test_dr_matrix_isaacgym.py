"""Full cross-backend DR matrix on IsaacGym (GPU, multi-env), one sim per subprocess.

Runs the dr_matrix_assert.py harness under IsaacGym in its own process (IsaacGym segfaults on a
second gymapi sim per process): it builds ONE real, fully-managed locomotion env (real
action_manager + randomization_manager, no shims) and asserts every robot + object DR term took
effect, reading back the native gym actor rigid-body / rigid-shape properties (mass, friction,
inertia Mat33, com). The MuJoCo columns are under ../mujoco/; IsaacSim under ../isaacsim/.

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

_HARNESS = Path(__file__).resolve().parents[1] / "dr_matrix_assert.py"


def test_dr_matrix_isaacgym(tmp_path):
    # Trust the result-file the harness writes after all checks pass (uniform with the other
    # backends, whose teardown can mask the exit code), not just the subprocess return code.
    result_file = tmp_path / "dr_matrix_result.txt"
    run_harness(
        _HARNESS,
        "--simulator",
        "isaacgym",
        "--num-envs",
        "4",
        "--result-file",
        str(result_file),
        label="isaacgym DR matrix",
        timeout=900,
        result_file=result_file,
    )
