# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""Live MuJoCo IMU tests for classic and Warp backends."""

from pathlib import Path

import pytest

from holosoma.utils.safe_torch_import import torch
from tests.simulators._run_harness import run_harness

pytest.importorskip("mujoco")

_HARNESS = Path(__file__).resolve().parents[1] / "imu_assert.py"


@pytest.mark.mujoco_classic
def test_imu_classic(tmp_path):
    result_file = tmp_path / "imu_mujoco.txt"
    run_harness(
        _HARNESS,
        "--simulator",
        "mujoco",
        "--result-file",
        str(result_file),
        label="mujoco/imu",
        timeout=300,
        result_file=result_file,
    )


@pytest.mark.mujoco_warp
@pytest.mark.skipif(not torch.cuda.is_available(), reason="MuJoCo-Warp requires a CUDA device")
def test_imu_warp_multi_env(tmp_path):
    """Three envs, to exercise the batched ``sensordata`` read rather than the per-env MjData path."""
    result_file = tmp_path / "imu_mjwarp.txt"
    run_harness(
        _HARNESS,
        "--simulator",
        "mjwarp",
        "--num-envs",
        "3",
        "--result-file",
        str(result_file),
        label="mjwarp/imu",
        timeout=600,
        result_file=result_file,
    )
