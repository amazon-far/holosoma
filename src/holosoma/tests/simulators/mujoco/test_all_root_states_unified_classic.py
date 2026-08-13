"""Live CPU (ClassicBackend) unified all_root_states (robot+object) via the shared harness.

Routes the MuJoCo ClassicBackend (CPU, 1 env) through all_root_states_unified_assert.py:
index/read/write/clone/sentinel checks for names including BOTH "robot" and the free object.

Runs in the MuJoCo (hsmujoco) CPU env — no CUDA. The Warp GPU / multi-env analogue is
test_all_root_states_unified_warp.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("mujoco")

# MuJoCo ClassicBackend (CPU) only.
pytestmark = pytest.mark.mujoco_classic

from tests.simulators._run_harness import run_harness  # noqa: E402

_HARNESS = Path(__file__).resolve().parents[1] / "all_root_states_unified_assert.py"


def test_all_root_states_unified():
    run_harness(
        _HARNESS,
        "--simulator",
        "mujoco",
        "--num-envs",
        "1",
        label="mujoco/all-root-states-unified (classic, num_envs=1)",
        timeout=600,
    )
