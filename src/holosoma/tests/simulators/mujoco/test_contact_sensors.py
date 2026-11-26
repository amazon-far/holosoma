"""Named contact contracts through the complete MuJoCo simulator lifecycle."""

from pathlib import Path

import pytest

from holosoma.utils.safe_torch_import import torch
from tests.simulators._run_harness import run_harness

pytest.importorskip("mujoco")
_HARNESS = Path(__file__).resolve().parents[1] / "contact_sensor_assert.py"


@pytest.mark.mujoco_classic
def test_named_contacts_classic(tmp_path: Path) -> None:
    result_file = tmp_path / "result.txt"
    run_harness(
        _HARNESS,
        "--simulator",
        "mujoco",
        "--result-file",
        str(result_file),
        label="mujoco/named-contacts",
        timeout=300,
        reuse_process=True,
        result_file=result_file,
    )


@pytest.mark.mujoco_warp
@pytest.mark.skipif(not torch.cuda.is_available(), reason="MuJoCo Warp requires a CUDA device")
def test_named_contacts_warp(tmp_path: Path) -> None:
    result_file = tmp_path / "result.txt"
    run_harness(
        _HARNESS,
        "--simulator",
        "mjwarp",
        "--num-envs",
        "2",
        "--result-file",
        str(result_file),
        label="mjwarp/named-contacts",
        timeout=600,
        reuse_process=True,
        result_file=result_file,
    )
