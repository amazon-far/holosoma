"""Live IsaacSim run of the YAM joint-trajectory harness (see yam_trajectory_assert.py), one sim per subprocess.

Marked ``isaacsim`` and ``importorskip("isaaclab")``/CUDA-gated like the other IsaacSim harness wrappers; the
verdict is the ``--result-file`` sentinel (IsaacSim teardown can corrupt the exit code).
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

_HARNESS = Path(__file__).resolve().parents[1] / "yam_trajectory_assert.py"


def test_yam_trajectory(tmp_path: Path) -> None:
    result_file = tmp_path / "yam_trajectory_isaacsim.txt"
    run_harness(
        _HARNESS,
        "--simulator",
        "isaacsim",
        "--result-file",
        str(result_file),
        label="isaacsim/yam-trajectory",
        timeout=1200,
        result_file=result_file,
    )
