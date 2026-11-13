"""Run the mounting-pose behavioral harness through the classic and Warp MuJoCo backends."""

from pathlib import Path

import pytest

from holosoma.utils.safe_torch_import import torch
from tests.simulators._run_harness import run_harness

pytest.importorskip("mujoco")

_HARNESS = Path(__file__).resolve().parents[1] / "sensor_mount_pose_assert.py"


@pytest.mark.mujoco_classic
def test_runtime_sensor_mounts_classic(tmp_path):
    result = tmp_path / "mounts_classic.txt"
    run_harness(
        _HARNESS,
        "--simulator",
        "mujoco",
        "--result-file",
        str(result),
        label="classic/runtime-sensor-mounts",
        timeout=400,
        result_file=result,
    )


@pytest.mark.mujoco_warp
@pytest.mark.skipif(not torch.cuda.is_available(), reason="MuJoCo Warp requires CUDA")
@pytest.mark.parametrize("num_envs", [1, 3])
def test_runtime_sensor_mounts_warp(tmp_path, num_envs):
    result = tmp_path / "mounts_warp.txt"
    run_harness(
        _HARNESS,
        "--simulator",
        "mjwarp",
        "--num-envs",
        str(num_envs),
        "--result-file",
        str(result),
        label="warp/runtime-sensor-mounts",
        timeout=500,
        result_file=result,
    )
