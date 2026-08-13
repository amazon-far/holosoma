"""Live MuJoCo Warp (GPU, multi-env) RUNTIME static-body relocation via the shared harness.

Routes the Warp backend through the SAME static_move_assert.py harness the Isaac backends use, so
the cross-backend before/after "box rests on the static, the static moves AND stops supporting it"
proof runs on mjwarp. (The in-process Classic analogue test_static_body_move_classic.py asserts the
same relocation on the MuJoCo contact list, single-env.)

MuJoCo ClassicBackend (CPU) cannot do >1 env, so this multi-env path is mjwarp-only.
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
    pytest.skip("Warp multi-env static-move requires a CUDA device", allow_module_level=True)

from tests.simulators._run_harness import run_harness  # noqa: E402

_HARNESS = Path(__file__).resolve().parents[1] / "static_move_assert.py"


@pytest.mark.parametrize("num_envs", [1, 4])
def test_static_move(num_envs):
    run_harness(
        _HARNESS,
        "--simulator",
        "mjwarp",
        "--num-envs",
        str(num_envs),
        label=f"mjwarp/static-move (num_envs={num_envs})",
        timeout=600,
    )
