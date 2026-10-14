"""Live MuJoCo Warp (GPU, multi-env) run of the fixed-base env harness.

Routes the Warp backend through fixed_base_env_assert.py: the ``test_arm_fixed`` robot goes through the
manager env (reset_all, zero-action steps, a subset reset) and two PPO iterations with the welded base at
its spawn pose throughout. The in-process Classic analogue is holosoma/envs/tests/test_fixed_base_arm_e2e_mujoco.py
(single env).

CUDA-gated; one sim per subprocess.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")

# MuJoCo WarpBackend (CUDA) only.
pytestmark = pytest.mark.mujoco_warp

if not torch.cuda.is_available():
    pytest.skip("Warp fixed-base env run requires a CUDA device", allow_module_level=True)

from tests.simulators._run_harness import run_harness  # noqa: E402

_HARNESS = Path(__file__).resolve().parents[1] / "fixed_base_env_assert.py"


@pytest.mark.parametrize("num_envs", [1, 4])
def test_fixed_base_env_warp(tmp_path: Path, num_envs: int) -> None:
    result_file = tmp_path / f"fixed_base_env_mjwarp_{num_envs}.txt"
    run_harness(
        _HARNESS,
        "--simulator",
        "mjwarp",
        "--num-envs",
        str(num_envs),
        "--result-file",
        str(result_file),
        label=f"mjwarp/fixed-base-env (num_envs={num_envs})",
        timeout=900,
        result_file=result_file,
    )
