"""Full cross-backend DR matrix on IsaacSim (GPU, multi-env), one sim per subprocess.

Runs the dr_matrix_assert.py harness under IsaacSim in its own process (IsaacSim's
SimulationContext is a process singleton — one sim per process): it builds ONE real,
fully-managed locomotion env (real action_manager + randomization_manager, no shims) and asserts
every robot + object DR term took effect, reading back the isaaclab physx-view properties
(``get_masses`` / ``get_material_properties`` / ``get_inertias`` / ``get_coms``). The MuJoCo
columns are under ../mujoco/; IsaacGym under ../isaacgym/.

Marked ``isaacsim`` so only the IsaacSim CI job (``-m isaacsim``) collects it.
``importorskip("isaaclab")``/CUDA-gated otherwise.
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

_HARNESS = Path(__file__).resolve().parents[1] / "dr_matrix_assert.py"


def test_dr_matrix_isaacsim(tmp_path):
    # IsaacSim's app teardown can hard-terminate the process (masking the exit code and swallowing
    # late stdout), so trust the result-file the harness writes AFTER all checks pass, not rc.
    result_file = tmp_path / "dr_matrix_result.txt"
    run_harness(
        _HARNESS,
        "--simulator",
        "isaacsim",
        "--num-envs",
        "4",
        "--result-file",
        str(result_file),
        label="isaacsim DR matrix",
        timeout=900,
        result_file=result_file,
    )
