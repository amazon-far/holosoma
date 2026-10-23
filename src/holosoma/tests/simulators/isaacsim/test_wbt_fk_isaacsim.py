"""Live check that WBT default-pose URDF forward kinematics matches IsaacSim (GPU), one sim per subprocess.

Runs wbt_fk_assert.py on a real G1. Marked ``isaacsim`` so the IsaacSim CI job collects it;
``importorskip("isaaclab")``/CUDA-gated.
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

_HARNESS = Path(__file__).resolve().parents[1] / "wbt_fk_assert.py"


def test_wbt_urdf_fk_matches_isaacsim(tmp_path):
    # Judged on the sentinel, not the exit code: IsaacSim teardown swallows a non-zero status, so a
    # crashed harness exits 0 and this would report a pass.
    result_file = tmp_path / "wbt_fk_isaacsim.txt"
    run_harness(
        _HARNESS,
        "--simulator",
        "isaacsim",
        "--result-file",
        str(result_file),
        label="isaacsim/wbt-urdf-fk",
        timeout=900,
        result_file=result_file,
    )
