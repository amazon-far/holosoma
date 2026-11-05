"""Live MuJoCo camera tests (classic CPU and Warp GPU).

Each test runs a camera assert harness under one MuJoCo backend. All selected cells of a backend run
in one shared subprocess (one sim per cell, built and torn down in turn). The geometry harness mounts
a <camera> on the robot base looking at a known red panel and checks the [N,H,W,3] uint8 frame's
panel silhouette is centered, square, and FOV-consistent.

  - classic: ClassicBackend (CPU, single env). Needs a GL context for mujoco.Renderer (DISPLAY
    for GLX). mujoco_classic marker.
  - mjwarp: WarpBackend (GPU), batched renderer, multi-env. mujoco_warp marker and CUDA gate.
"""

from __future__ import annotations

import sys

import pytest

pytest.importorskip("mujoco")
from holosoma.utils.safe_torch_import import torch
from tests.simulators._run_harness import run_batched

_BATCH_TIMEOUT = 900

_requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="MuJoCo-Warp requires a CUDA device")

# test function name -> (harness module, args after --simulator). ``{num_envs}`` is the test's param.
_CELLS = {
    "test_camera_geometry_classic": ("camera_geometry_assert", []),
    "test_camera_geometry_warp": ("camera_geometry_assert", ["--num-envs", "{num_envs}"]),
    "test_camera_depth_classic": ("depth_assert", []),
    "test_camera_depth_warp": ("depth_assert", []),
    "test_camera_follow_classic": ("camera_follow_assert", []),
    "test_camera_follow_multi_env_warp": ("camera_follow_assert", ["--num-envs", "4"]),
    "test_camera_multi_classic": ("camera_multi_assert", []),
    "test_camera_multi_warp": ("camera_multi_assert", []),
    "test_camera_actor_mount_classic": ("camera_actor_mount_assert", []),
    "test_camera_actor_mount_warp": ("camera_actor_mount_assert", []),
    "test_camera_world_mount_classic": ("world_camera_assert", []),
    "test_camera_world_mount_warp": ("world_camera_assert", ["--num-envs", "{num_envs}"]),
    "test_camera_orientation_classic": ("camera_orientation_assert", []),
    "test_camera_orientation_warp": ("camera_orientation_assert", []),
    "test_camera_obs_pipeline_classic": ("camera_obs_assert", []),
    "test_camera_obs_pipeline_warp": ("camera_obs_assert", []),
}


def _cell(item):
    if getattr(item, "module", None) is not sys.modules[__name__] or item.originalname not in _CELLS:
        return None
    stem, args = _CELLS[item.originalname]
    simulator = "mjwarp" if item.originalname.endswith("_warp") else "mujoco"
    params = getattr(item, "callspec", None)
    num_envs = params.params.get("num_envs", "") if params else ""
    args = [a.format(num_envs=num_envs) for a in args]
    return f"camera/{simulator}", item.name, stem, ["--simulator", simulator, *args]


def _run(request, label):
    run_batched(request, _cell, label=label, timeout=_BATCH_TIMEOUT)


@pytest.mark.mujoco_classic
def test_camera_geometry_classic(request):
    _run(request, "mujoco/camera-geometry")


@pytest.mark.mujoco_warp
@_requires_cuda
@pytest.mark.parametrize("num_envs", ["1", "3"])
def test_camera_geometry_warp(request, num_envs):
    # Co-located envs: every world renders the same geometry in the batched [N,H,W,3] output.
    _run(request, f"mjwarp/camera-geometry (num_envs={num_envs})")


@pytest.mark.mujoco_classic
def test_camera_depth_classic(request):
    # Depth in meters, image-plane, +inf no-hit. Panel at a known distance reads ~that.
    _run(request, "mujoco/camera-depth")


@pytest.mark.mujoco_warp
@_requires_cuda
def test_camera_depth_warp(request):
    # Batched get_depth(depth_scale=far) -> meters with +inf no-hit.
    _run(request, "mjwarp/camera-depth")


@pytest.mark.mujoco_classic
def test_camera_follow_classic(request):
    # Camera tracks its mount body: move the robot toward the panel, the panel depth drops by the moved distance.
    _run(request, "mujoco/camera-follow")


@pytest.mark.mujoco_classic
def test_camera_multi_classic(request):
    # Two cameras with different views each return their own frame (no buffer mixup).
    _run(request, "mujoco/camera-multi")


@pytest.mark.mujoco_warp
@_requires_cuda
def test_camera_multi_warp(request):
    _run(request, "mjwarp/camera-multi")


@pytest.mark.mujoco_classic
def test_camera_actor_mount_classic(request):
    # target_kind="actor": a camera on a scene object resolves, renders, and follows it.
    _run(request, "mujoco/camera-actor-mount")


@pytest.mark.mujoco_warp
@_requires_cuda
def test_camera_actor_mount_warp(request):
    _run(request, "mjwarp/camera-actor-mount")


@pytest.mark.mujoco_classic
def test_camera_world_mount_classic(request):
    # target_kind="world": a free-floating camera fixed in the env frame does NOT follow the robot.
    _run(request, "mujoco/camera-world-mount")


@pytest.mark.mujoco_warp
@_requires_cuda
@pytest.mark.parametrize("num_envs", ["1", "3"])
def test_camera_world_mount_warp(request, num_envs):
    _run(request, f"mjwarp/camera-world-mount (num_envs={num_envs})")


@pytest.mark.mujoco_classic
def test_camera_orientation_classic(request):
    # Off-axis panel on a non-square camera lands in the expected image quadrant (catches mirror/flip/transpose).
    _run(request, "mujoco/camera-orientation")


@pytest.mark.mujoco_warp
@_requires_cuda
def test_camera_orientation_warp(request):
    _run(request, "mjwarp/camera-orientation")


@pytest.mark.mujoco_classic
def test_camera_obs_pipeline_classic(request):
    # Drives render_sensors -> camera obs term -> transform -> obs dict each step; asserts CHW
    # float01 format and that a decimation=3 camera holds while a decimation=1 camera updates.
    _run(request, "mujoco/camera-obs-pipeline")


@pytest.mark.mujoco_warp
@_requires_cuda
def test_camera_obs_pipeline_warp(request):
    _run(request, "mjwarp/camera-obs-pipeline")


@pytest.mark.mujoco_warp
@_requires_cuda
def test_camera_follow_multi_env_warp(request):
    # num_envs=4 at distinct origins: yaw only env 0 and assert the other envs' frames are unchanged.
    _run(request, "mjwarp/camera-follow (num_envs=4)")
