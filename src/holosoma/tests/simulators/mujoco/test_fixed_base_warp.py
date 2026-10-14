"""Live MuJoCo Warp (GPU, multi-env) run of the fixed-base robot harness.

Routes the Warp backend through the SAME fixed_base_robot_assert.py the Isaac backends use, so the
fixed-robot state contract (constant per-env spawn pose, live joints, velocity-ignoring teleports)
is proven per world: at ``num_envs=4`` every env's welded base sits at its own origin. The
in-process Classic analogue is test_fixed_base_classic.py (single env).

CUDA-gated; one sim per subprocess (the harness builds + tears down a full sim per run).
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")

# MuJoCo WarpBackend (CUDA) only.
pytestmark = pytest.mark.mujoco_warp

if not torch.cuda.is_available():
    pytest.skip("Warp fixed-base run requires a CUDA device", allow_module_level=True)

from tests.simulators._run_harness import run_harness  # noqa: E402

_HARNESS = Path(__file__).resolve().parents[1] / "fixed_base_robot_assert.py"


@pytest.mark.parametrize("num_envs", [1, 4])
def test_fixed_base_robot_warp(tmp_path: Path, num_envs: int) -> None:
    result_file = tmp_path / f"fixed_base_mjwarp_{num_envs}.txt"
    run_harness(
        _HARNESS,
        "--simulator",
        "mjwarp",
        "--num-envs",
        str(num_envs),
        "--result-file",
        str(result_file),
        label=f"mjwarp/fixed-base (num_envs={num_envs})",
        timeout=600,
        result_file=result_file,
    )
