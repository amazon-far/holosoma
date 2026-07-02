"""Full cross-backend DR matrix on the MuJoCo WarpBackend (GPU, multi-env), one sim per process.

Runs the dr_matrix_assert.py harness under the MuJoCo-Warp backend in its own process: it builds
ONE real, fully-managed locomotion env (real action_manager + randomization_manager, no shims)
and asserts every robot + object DR term took effect, reading back the per-world
``warp_model_bridge``. The classic-CPU column runs the same harness in-process
(test_dr_matrix_classic.py); the Isaac columns live under ../isaacgym/ and ../isaacsim/.

Skipped without CUDA / the MuJoCo-Warp stack.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")

# MuJoCo WarpBackend (CUDA) only.
pytestmark = pytest.mark.mujoco_warp

if not torch.cuda.is_available():
    pytest.skip("WarpBackend multi-env tests require a CUDA device", allow_module_level=True)

from tests.simulators._run_harness import run_harness  # noqa: E402

_HARNESS = Path(__file__).resolve().parents[1] / "dr_matrix_assert.py"


def test_dr_matrix_warp(tmp_path):
    # Trust the result-file the harness writes after all checks pass (uniform with the other
    # backends, whose teardown can mask the exit code), not just the subprocess return code.
    result_file = tmp_path / "dr_matrix_result.txt"
    run_harness(
        _HARNESS,
        "--simulator",
        "mjwarp",
        "--num-envs",
        "4",
        "--result-file",
        str(result_file),
        label="mjwarp DR matrix",
        timeout=900,
        result_file=result_file,
    )
