"""Live MuJoCo Warp (GPU, multi-env) unified all_root_states (robot+object) via the shared harness.

Routes the Warp backend through all_root_states_unified_assert.py: index/read/write/clone/sentinel
checks for names including BOTH "robot" and the free object, in every env.

MuJoCo ClassicBackend (CPU) cannot do >1 env; the CPU/1-env variant is test_all_root_states_unified_classic.py.
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
    pytest.skip("Warp multi-env unified all_root_states requires a CUDA device", allow_module_level=True)

from tests.simulators._run_harness import run_harness  # noqa: E402

_HARNESS = Path(__file__).resolve().parents[1] / "all_root_states_unified_assert.py"


def test_all_root_states_unified():
    run_harness(
        _HARNESS,
        "--simulator",
        "mjwarp",
        "--num-envs",
        "4",
        label="mjwarp/all-root-states-unified (num_envs=4)",
        timeout=600,
    )
