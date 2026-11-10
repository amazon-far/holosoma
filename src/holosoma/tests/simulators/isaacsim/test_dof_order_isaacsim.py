"""Live IsaacSim (GPU) run of dof_order_assert.py: DOF state and torques follow
``robot_config.dof_names`` when it differs from the asset's joint order. One sim per subprocess.

Marked ``isaacsim`` so the IsaacSim CI job collects it; ``importorskip("isaaclab")``/CUDA-gated.
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

_HARNESS = Path(__file__).resolve().parents[1] / "dof_order_assert.py"


def test_dof_order():
    run_harness(_HARNESS, "--simulator", "isaacsim", label="isaacsim/dof-order", timeout=900)
