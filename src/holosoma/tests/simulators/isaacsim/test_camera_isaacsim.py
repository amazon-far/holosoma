"""Live IsaacSim camera tests (GPU).

Each test runs a camera assert harness against the IsaacSim backend. The geometry harness mounts
a TiledCamera on the robot base looking at a known red panel and checks the [N,H,W,3] uint8
frame's panel silhouette is centered, square, and FOV-consistent. Pass/fail is read from the
harness's --result-file sentinel. All selected headless cells run in one shared subprocess and Kit
app, each on a fresh stage; test_camera_obs_pipeline boots the windowed experience in its own process.

Marked ``isaacsim`` so the IsaacSim CI job (-m "isaacsim") collects it. importorskip("isaaclab")
and CUDA-gated. The CI job exports OMNI_KIT_ACCEPT_EULA=YES.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.isaacsim

pytest.importorskip("isaaclab")
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("IsaacSim requires a CUDA device", allow_module_level=True)

from tests.simulators._run_harness import run_batched, run_harness  # noqa: E402

_OBS_HARNESS = Path(__file__).resolve().parents[1] / "camera_obs_assert.py"
_TIMEOUT = 2400

# Test name -> (harness stem, args); ``{name}`` is filled from the test's parameters.
_CELLS = {
    "test_camera_geometry": ("camera_geometry_assert", ["--simulator", "isaacsim"]),
    "test_camera_depth": ("depth_assert", ["--simulator", "isaacsim"]),
    "test_camera_fisheye_projection": ("camera_projection_assert", ["--simulator", "isaacsim"]),
    "test_camera_multi": ("camera_multi_assert", ["--simulator", "isaacsim"]),
    "test_camera_actor_mount": ("camera_actor_mount_assert", ["--simulator", "isaacsim"]),
    "test_camera_world_mount": ("world_camera_assert", ["--simulator", "isaacsim", "--num-envs", "{num_envs}"]),
    "test_camera_follow_multi_env": ("camera_follow_assert", ["--simulator", "isaacsim", "--num-envs", "4"]),
    "test_camera_orientation": ("camera_orientation_assert", ["--simulator", "isaacsim"]),
    "test_camera_renders_at_its_mount_offset": ("camera_mount_fabric_assert", ["--num-envs", "{num_envs}"]),
}


def _cell(item):
    """Batch cell for a headless camera test item: one subprocess runs every selected one."""
    if getattr(item, "module", None) is not sys.modules[__name__] or item.originalname not in _CELLS:
        return None
    stem, args = _CELLS[item.originalname]
    params = item.callspec.params if hasattr(item, "callspec") else {}
    return "camera/isaacsim", item.name, stem, [arg.format(**params) for arg in args]


def test_camera_geometry(request):
    run_batched(request, _cell, label="isaacsim/camera-geometry", timeout=_TIMEOUT)


def test_camera_depth(request):
    # Depth in meters, image-plane, +inf no-hit. Panel at a known distance reads ~that.
    run_batched(request, _cell, label="isaacsim/camera-depth", timeout=_TIMEOUT)


def test_camera_fisheye_projection(request):
    # The authored f-theta calibration produces observably wider projection geometry than pinhole.
    run_batched(request, _cell, label="isaacsim/camera-fisheye", timeout=_TIMEOUT)


def test_camera_multi(request):
    # Two cameras with different views each return their own frame (no buffer mixup).
    run_batched(request, _cell, label="isaacsim/camera-multi", timeout=_TIMEOUT)


def test_camera_actor_mount(request):
    # Exercises target_kind='actor': a camera on a scene object resolves, renders, and follows it.
    run_batched(request, _cell, label="isaacsim/camera-actor-mount", timeout=_TIMEOUT)


@pytest.mark.parametrize("num_envs", ["1", "3"])
def test_camera_world_mount(request, num_envs):
    # target_kind='world': a free-floating camera (child of the env prim) fixed in the env frame
    # does NOT follow the robot, and each env's camera sits at its own origin.
    run_batched(request, _cell, label=f"isaacsim/camera-world-mount (num_envs={num_envs})", timeout=_TIMEOUT)


def test_camera_follow_multi_env(request):
    # num_envs=4: yaw only env 0 and assert the other envs' frames are unchanged.
    run_batched(request, _cell, label="isaacsim/camera-follow (num_envs=4)", timeout=_TIMEOUT)


def test_camera_orientation(request):
    # Off-axis panel on a non-square camera lands in the expected image quadrant (catches mirror/flip/transpose).
    run_batched(request, _cell, label="isaacsim/camera-orientation", timeout=_TIMEOUT)


def test_camera_obs_pipeline(tmp_path, monkeypatch):
    # Drives render_sensors -> camera obs term -> transform -> obs dict; asserts CHW float01 format
    # and that a decimation=3 camera holds while a decimation=1 camera updates.
    # Boots the windowed Kit experience, the one path where the launcher must enable offscreen render.
    monkeypatch.delenv("HEADLESS")
    result_file = tmp_path / "obs_isaacsim.txt"
    run_harness(
        _OBS_HARNESS,
        "--simulator",
        "isaacsim",
        "--result-file",
        str(result_file),
        label="isaacsim/camera-obs-pipeline",
        timeout=700,
        result_file=result_file,
    )


@pytest.mark.parametrize("num_envs", [1, 2])
def test_camera_renders_at_its_mount_offset(request, num_envs):
    # Cameras on links that move while Isaac Lab initializes sensors must still render from their
    # authored offset: Fabric camera world == mount body world x USD local, after startup and later.
    run_batched(request, _cell, label=f"isaacsim/camera-mount-fabric (num_envs={num_envs})", timeout=_TIMEOUT)
