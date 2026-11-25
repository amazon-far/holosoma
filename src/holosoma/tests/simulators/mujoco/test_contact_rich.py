"""A real projectile changes contact partners through both MuJoCo backends."""

from pathlib import Path

import pytest

from holosoma.utils.safe_torch_import import torch
from tests.simulators._run_harness import run_harness

pytest.importorskip("mujoco")
_HARNESS = Path(__file__).resolve().parents[1] / "contact_rich_assert.py"


@pytest.mark.mujoco_classic
def test_contact_rich_classic(tmp_path: Path) -> None:
    result = tmp_path / "result.txt"
    run_harness(
        _HARNESS,
        "--simulator",
        "mujoco",
        "--result-file",
        str(result),
        label="mujoco/contact-rich",
        timeout=300,
        reuse_process=True,
        result_file=result,
    )


@pytest.mark.mujoco_warp
@pytest.mark.skipif(not torch.cuda.is_available(), reason="MuJoCo Warp requires a CUDA device")
def test_contact_rich_warp(tmp_path: Path) -> None:
    result = tmp_path / "result.txt"
    run_harness(
        _HARNESS,
        "--simulator",
        "mjwarp",
        "--num-envs",
        "2",
        "--result-file",
        str(result),
        label="mjwarp/contact-rich",
        timeout=600,
        result_file=result,
    )
