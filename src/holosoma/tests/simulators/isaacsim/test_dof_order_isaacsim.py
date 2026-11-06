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


_HARNESS = Path(__file__).resolve().parents[1] / "dof_order_assert.py"


@pytest.mark.parametrize("use_newton_actuators", ["true", "false"], ids=["default", "fallback"])
def test_dof_order(use_newton_actuators: str, tmp_path, run_isaacsim_harness, isaacsim_backend):
    if use_newton_actuators == "false" and isaacsim_backend != "isaaclab3":
        pytest.skip("The alternate actuator execution path is an Isaac Lab 3 option")
    result_file = tmp_path / "dof_order.txt"
    run_isaacsim_harness(
        _HARNESS,
        "--use-newton-actuators",
        use_newton_actuators,
        "--result-file",
        str(result_file),
        label=f"isaacsim/dof-order/{use_newton_actuators}",
        timeout=900,
        result_file=result_file,
    )
