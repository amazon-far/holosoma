"""Live MuJoCo Warp (GPU, multi-env) run of dof_order_assert.py: DOF state and torques follow
``robot_config.dof_names`` when it differs from the MJCF joint order. Each case builds its own sim
in the shared harness worker.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")

# MuJoCo WarpBackend (CUDA) only.
pytestmark = pytest.mark.mujoco_warp

if not torch.cuda.is_available():
    pytest.skip("Warp multi-env DOF-order check requires a CUDA device", allow_module_level=True)

from tests.simulators._run_harness import run_harness  # noqa: E402

_HARNESS = Path(__file__).resolve().parents[1] / "dof_order_assert.py"


def test_dof_order():
    run_harness(_HARNESS, "--simulator", "mjwarp", label="mjwarp/dof-order", timeout=600, reuse_process=True)
