"""Live MuJoCo check that served state is current at every physics substep (state_freshness_assert.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")

from tests.simulators._run_harness import run_harness  # noqa: E402

_HARNESS = Path(__file__).resolve().parents[1] / "state_freshness_assert.py"
_NEEDS_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


@pytest.mark.parametrize(
    ("label", "args"),
    [
        pytest.param("classic", ["--simulator", "mujoco"], marks=pytest.mark.mujoco_classic, id="classic"),
        pytest.param(
            "classic, cuda",
            ["--simulator", "mujoco", "--device", "cuda:0"],
            marks=[pytest.mark.mujoco_classic, _NEEDS_CUDA],
            id="classic-cuda",
        ),
        pytest.param(
            "warp",
            ["--simulator", "mjwarp", "--num-envs", "4"],
            marks=[pytest.mark.mujoco_warp, _NEEDS_CUDA],
            id="warp",
        ),
    ],
)
def test_state_freshness(tmp_path, label, args):
    result_file = tmp_path / "result.txt"
    run_harness(
        _HARNESS,
        *args,
        "--result-file",
        str(result_file),
        label=f"mujoco/state-freshness ({label})",
        timeout=600,
        reuse_process=True,
        result_file=result_file,
    )
