"""Live MuJoCo camera tests (classic CPU and Warp GPU), one sim per subprocess.

Each test runs a camera assert harness under both MuJoCo backends. The geometry harness mounts a
<camera> on the robot base looking at a known red panel and checks the [N,H,W,3] uint8 frame's
panel silhouette is centered, square, and FOV-consistent.

  - classic: ClassicBackend (CPU, single env). Needs a GL context for mujoco.Renderer (DISPLAY
    for GLX). mujoco_classic marker.
  - mjwarp: WarpBackend (GPU), batched renderer, multi-env. mujoco_warp marker and CUDA gate.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("mujoco")
from holosoma.utils.safe_torch_import import torch
from tests.simulators._run_harness import run_harness

_HARNESS = Path(__file__).resolve().parents[1] / "camera_geometry_assert.py"
_DEPTH_HARNESS = Path(__file__).resolve().parents[1] / "depth_assert.py"
_CONTRACT_HARNESS = Path(__file__).resolve().parents[1] / "camera_assert.py"
_FOLLOW_HARNESS = Path(__file__).resolve().parents[1] / "camera_follow_assert.py"
_MULTI_HARNESS = Path(__file__).resolve().parents[1] / "camera_multi_assert.py"
_ACTOR_HARNESS = Path(__file__).resolve().parents[1] / "camera_actor_mount_assert.py"
_WORLD_HARNESS = Path(__file__).resolve().parents[1] / "world_camera_assert.py"
_ORIENT_HARNESS = Path(__file__).resolve().parents[1] / "camera_orientation_assert.py"
_OBS_HARNESS = Path(__file__).resolve().parents[1] / "camera_obs_assert.py"


@pytest.mark.mujoco_classic
def test_camera_geometry_classic(tmp_path):
    result_file = tmp_path / "geo_mujoco.txt"
    run_harness(
        _HARNESS,
        "--simulator",
        "mujoco",
        "--result-file",
        str(result_file),
        label="mujoco/camera-geometry",
        timeout=300,
        result_file=result_file,
    )


@pytest.mark.mujoco_warp
@pytest.mark.skipif(not torch.cuda.is_available(), reason="MuJoCo-Warp requires a CUDA device")
@pytest.mark.parametrize("num_envs", ["1", "3"])
def test_camera_geometry_warp(tmp_path, num_envs):
    # Co-located envs: every world renders the same geometry in the batched [N,H,W,3] output.
    result_file = tmp_path / f"geo_mjwarp_{num_envs}.txt"
    run_harness(
        _HARNESS,
        "--simulator",
        "mjwarp",
        "--num-envs",
        num_envs,
        "--result-file",
        str(result_file),
        label=f"mjwarp/camera-geometry (num_envs={num_envs})",
        timeout=400,
        result_file=result_file,
    )


@pytest.mark.mujoco_classic
def test_camera_depth_classic(tmp_path):
    # Depth in meters, image-plane, +inf no-hit. Panel at a known distance reads ~that.
    result_file = tmp_path / "depth_mujoco.txt"
    run_harness(
        _DEPTH_HARNESS,
        "--simulator",
        "mujoco",
        "--result-file",
        str(result_file),
        label="mujoco/camera-depth",
        timeout=300,
        result_file=result_file,
    )


@pytest.mark.mujoco_warp
@pytest.mark.skipif(not torch.cuda.is_available(), reason="MuJoCo-Warp requires a CUDA device")
def test_camera_depth_warp(tmp_path):
    # Batched get_depth(depth_scale=far) -> meters with +inf no-hit.
    result_file = tmp_path / "depth_mjwarp.txt"
    run_harness(
        _DEPTH_HARNESS,
        "--simulator",
        "mjwarp",
        "--result-file",
        str(result_file),
        label="mjwarp/camera-depth",
        timeout=400,
        result_file=result_file,
    )


@pytest.mark.mujoco_classic
def test_camera_recorder_classic(tmp_path):
    # A viz config requesting recording yields a non-None camera_sensor_recorder that capture()s a frame.
    result_file = tmp_path / "rec_mujoco.txt"
    run_harness(
        _CONTRACT_HARNESS,
        "--simulator",
        "mujoco",
        "--check-recorder",
        "--result-file",
        str(result_file),
        label="mujoco/camera-recorder",
        timeout=300,
        result_file=result_file,
    )


@pytest.mark.mujoco_warp
@pytest.mark.skipif(not torch.cuda.is_available(), reason="MuJoCo-Warp requires a CUDA device")
def test_camera_recorder_warp(tmp_path):
    result_file = tmp_path / "rec_mjwarp.txt"
    run_harness(
        _CONTRACT_HARNESS,
        "--simulator",
        "mjwarp",
        "--check-recorder",
        "--result-file",
        str(result_file),
        label="mjwarp/camera-recorder",
        timeout=400,
        result_file=result_file,
    )


@pytest.mark.mujoco_classic
def test_camera_follow_classic(tmp_path):
    # Camera tracks its mount body: move the robot toward the panel, the panel depth drops by the moved distance.
    result_file = tmp_path / "follow_mujoco.txt"
    run_harness(
        _FOLLOW_HARNESS,
        "--simulator",
        "mujoco",
        "--result-file",
        str(result_file),
        label="mujoco/camera-follow",
        timeout=300,
        result_file=result_file,
    )


@pytest.mark.mujoco_warp
@pytest.mark.skipif(not torch.cuda.is_available(), reason="MuJoCo-Warp requires a CUDA device")
def test_camera_follow_warp(tmp_path):
    result_file = tmp_path / "follow_mjwarp.txt"
    run_harness(
        _FOLLOW_HARNESS,
        "--simulator",
        "mjwarp",
        "--result-file",
        str(result_file),
        label="mjwarp/camera-follow",
        timeout=400,
        result_file=result_file,
    )


@pytest.mark.mujoco_classic
def test_camera_multi_classic(tmp_path):
    # Two cameras with different views each return their own frame (no buffer mixup).
    result_file = tmp_path / "multi_mujoco.txt"
    run_harness(
        _MULTI_HARNESS,
        "--simulator",
        "mujoco",
        "--result-file",
        str(result_file),
        label="mujoco/camera-multi",
        timeout=300,
        result_file=result_file,
    )


@pytest.mark.mujoco_warp
@pytest.mark.skipif(not torch.cuda.is_available(), reason="MuJoCo-Warp requires a CUDA device")
def test_camera_multi_warp(tmp_path):
    result_file = tmp_path / "multi_mjwarp.txt"
    run_harness(
        _MULTI_HARNESS,
        "--simulator",
        "mjwarp",
        "--result-file",
        str(result_file),
        label="mjwarp/camera-multi",
        timeout=400,
        result_file=result_file,
    )


@pytest.mark.mujoco_classic
def test_camera_actor_mount_classic(tmp_path):
    # target_kind="actor": a camera on a scene object resolves, renders, and follows it.
    result_file = tmp_path / "actor_mujoco.txt"
    run_harness(
        _ACTOR_HARNESS,
        "--simulator",
        "mujoco",
        "--result-file",
        str(result_file),
        label="mujoco/camera-actor-mount",
        timeout=300,
        result_file=result_file,
    )


@pytest.mark.mujoco_warp
@pytest.mark.skipif(not torch.cuda.is_available(), reason="MuJoCo-Warp requires a CUDA device")
def test_camera_actor_mount_warp(tmp_path):
    result_file = tmp_path / "actor_mjwarp.txt"
    run_harness(
        _ACTOR_HARNESS,
        "--simulator",
        "mjwarp",
        "--result-file",
        str(result_file),
        label="mjwarp/camera-actor-mount",
        timeout=400,
        result_file=result_file,
    )


@pytest.mark.mujoco_classic
def test_camera_world_mount_classic(tmp_path):
    # target_kind="world": a free-floating camera fixed in the env frame does NOT follow the robot.
    result_file = tmp_path / "world_mujoco.txt"
    run_harness(
        _WORLD_HARNESS,
        "--simulator",
        "mujoco",
        "--result-file",
        str(result_file),
        label="mujoco/camera-world-mount",
        timeout=300,
        result_file=result_file,
    )


@pytest.mark.mujoco_warp
@pytest.mark.skipif(not torch.cuda.is_available(), reason="MuJoCo-Warp requires a CUDA device")
@pytest.mark.parametrize("num_envs", ["1", "3"])
def test_camera_world_mount_warp(tmp_path, num_envs):
    result_file = tmp_path / f"world_mjwarp_{num_envs}.txt"
    run_harness(
        _WORLD_HARNESS,
        "--simulator",
        "mjwarp",
        "--num-envs",
        num_envs,
        "--result-file",
        str(result_file),
        label=f"mjwarp/camera-world-mount (num_envs={num_envs})",
        timeout=400,
        result_file=result_file,
    )


@pytest.mark.mujoco_classic
def test_camera_orientation_classic(tmp_path):
    # Off-axis panel on a non-square camera lands in the expected image quadrant (catches mirror/flip/transpose).
    result_file = tmp_path / "orient_mujoco.txt"
    run_harness(
        _ORIENT_HARNESS,
        "--simulator",
        "mujoco",
        "--result-file",
        str(result_file),
        label="mujoco/camera-orientation",
        timeout=300,
        result_file=result_file,
    )


@pytest.mark.mujoco_warp
@pytest.mark.skipif(not torch.cuda.is_available(), reason="MuJoCo-Warp requires a CUDA device")
def test_camera_orientation_warp(tmp_path):
    result_file = tmp_path / "orient_mjwarp.txt"
    run_harness(
        _ORIENT_HARNESS,
        "--simulator",
        "mjwarp",
        "--result-file",
        str(result_file),
        label="mjwarp/camera-orientation",
        timeout=400,
        result_file=result_file,
    )


@pytest.mark.mujoco_classic
def test_camera_obs_pipeline_classic(tmp_path):
    # Drives render_sensors -> camera obs term -> transform -> obs dict each step; asserts CHW
    # float01 format and that a decimation=3 camera holds while a decimation=1 camera updates.
    result_file = tmp_path / "obs_mujoco.txt"
    run_harness(
        _OBS_HARNESS,
        "--simulator",
        "mujoco",
        "--result-file",
        str(result_file),
        label="mujoco/camera-obs-pipeline",
        timeout=300,
        result_file=result_file,
    )


@pytest.mark.mujoco_warp
@pytest.mark.skipif(not torch.cuda.is_available(), reason="MuJoCo-Warp requires a CUDA device")
def test_camera_obs_pipeline_warp(tmp_path):
    result_file = tmp_path / "obs_mjwarp.txt"
    run_harness(
        _OBS_HARNESS,
        "--simulator",
        "mjwarp",
        "--result-file",
        str(result_file),
        label="mjwarp/camera-obs-pipeline",
        timeout=400,
        result_file=result_file,
    )


@pytest.mark.mujoco_warp
@pytest.mark.skipif(not torch.cuda.is_available(), reason="MuJoCo-Warp requires a CUDA device")
def test_camera_follow_multi_env_warp(tmp_path):
    # num_envs=4 at distinct origins: yaw only env 0 and assert the other envs' frames are unchanged.
    result_file = tmp_path / "follow_multi_mjwarp.txt"
    run_harness(
        _FOLLOW_HARNESS,
        "--simulator",
        "mjwarp",
        "--num-envs",
        "4",
        "--result-file",
        str(result_file),
        label="mjwarp/camera-follow (num_envs=4)",
        timeout=500,
        result_file=result_file,
    )
