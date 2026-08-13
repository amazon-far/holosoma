"""Live IsaacSim camera tests (GPU), one sim per subprocess.

Each test runs a camera assert harness against the IsaacSim backend. The geometry harness mounts
a TiledCamera on the robot base looking at a known red panel and checks the [N,H,W,3] uint8
frame's panel silhouette is centered, square, and FOV-consistent. Pass/fail is read from the
harness's --result-file sentinel, since IsaacSim teardown can mask the exit code.

Marked ``isaacsim`` so the IsaacSim CI job (-m "isaacsim") collects it. importorskip("isaaclab")
and CUDA-gated. The CI job exports OMNI_KIT_ACCEPT_EULA=YES.
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

_HARNESS = Path(__file__).resolve().parents[1] / "camera_geometry_assert.py"
_DEPTH_HARNESS = Path(__file__).resolve().parents[1] / "depth_assert.py"
_CONTRACT_HARNESS = Path(__file__).resolve().parents[1] / "camera_assert.py"
_FOLLOW_HARNESS = Path(__file__).resolve().parents[1] / "camera_follow_assert.py"
_MULTI_HARNESS = Path(__file__).resolve().parents[1] / "camera_multi_assert.py"
_ACTOR_HARNESS = Path(__file__).resolve().parents[1] / "camera_actor_mount_assert.py"
_WORLD_HARNESS = Path(__file__).resolve().parents[1] / "world_camera_assert.py"
_ORIENT_HARNESS = Path(__file__).resolve().parents[1] / "camera_orientation_assert.py"
_OBS_HARNESS = Path(__file__).resolve().parents[1] / "camera_obs_assert.py"


def test_camera_geometry(tmp_path):
    result_file = tmp_path / "geo_isaacsim.txt"
    run_harness(
        _HARNESS,
        "--simulator",
        "isaacsim",
        "--result-file",
        str(result_file),
        label="isaacsim/camera-geometry",
        timeout=700,
        result_file=result_file,
    )


def test_camera_depth(tmp_path):
    # Depth in meters, image-plane, +inf no-hit. Panel at a known distance reads ~that.
    result_file = tmp_path / "depth_isaacsim.txt"
    run_harness(
        _DEPTH_HARNESS,
        "--simulator",
        "isaacsim",
        "--result-file",
        str(result_file),
        label="isaacsim/camera-depth",
        timeout=700,
        result_file=result_file,
    )


def test_camera_recorder(tmp_path):
    # A viz config requesting recording yields a non-None camera_sensor_recorder that capture()s a frame.
    result_file = tmp_path / "rec_isaacsim.txt"
    run_harness(
        _CONTRACT_HARNESS,
        "--simulator",
        "isaacsim",
        "--check-recorder",
        "--result-file",
        str(result_file),
        label="isaacsim/camera-recorder",
        timeout=700,
        result_file=result_file,
    )


def test_camera_follow(tmp_path):
    # Camera tracks its mount body: move the robot, the panel depth drops by the moved distance.
    result_file = tmp_path / "follow_isaacsim.txt"
    run_harness(
        _FOLLOW_HARNESS,
        "--simulator",
        "isaacsim",
        "--result-file",
        str(result_file),
        label="isaacsim/camera-follow",
        timeout=700,
        result_file=result_file,
    )


def test_camera_multi(tmp_path):
    # Two cameras with different views each return their own frame (no buffer mixup).
    result_file = tmp_path / "multi_isaacsim.txt"
    run_harness(
        _MULTI_HARNESS,
        "--simulator",
        "isaacsim",
        "--result-file",
        str(result_file),
        label="isaacsim/camera-multi",
        timeout=700,
        result_file=result_file,
    )


def test_camera_actor_mount(tmp_path):
    # Exercises target_kind='actor': a camera on a scene object resolves, renders, and follows it.
    result_file = tmp_path / "actor_isaacsim.txt"
    run_harness(
        _ACTOR_HARNESS,
        "--simulator",
        "isaacsim",
        "--result-file",
        str(result_file),
        label="isaacsim/camera-actor-mount",
        timeout=700,
        result_file=result_file,
    )


@pytest.mark.parametrize("num_envs", ["1", "3"])
def test_camera_world_mount(tmp_path, num_envs):
    # target_kind='world': a free-floating camera (child of the env prim) fixed in the env frame
    # does NOT follow the robot, and each env's camera sits at its own origin.
    result_file = tmp_path / f"world_isaacsim_{num_envs}.txt"
    run_harness(
        _WORLD_HARNESS,
        "--simulator",
        "isaacsim",
        "--num-envs",
        num_envs,
        "--result-file",
        str(result_file),
        label=f"isaacsim/camera-world-mount (num_envs={num_envs})",
        timeout=700,
        result_file=result_file,
    )


def test_camera_follow_multi_env(tmp_path):
    # num_envs=4: yaw only env 0 and assert the other envs' frames are unchanged.
    result_file = tmp_path / "follow_isaacsim_multi.txt"
    run_harness(
        _FOLLOW_HARNESS,
        "--simulator",
        "isaacsim",
        "--num-envs",
        "4",
        "--result-file",
        str(result_file),
        label="isaacsim/camera-follow (num_envs=4)",
        timeout=700,
        result_file=result_file,
    )


def test_camera_orientation(tmp_path):
    # Off-axis panel on a non-square camera lands in the expected image quadrant (catches mirror/flip/transpose).
    result_file = tmp_path / "orient_isaacsim.txt"
    run_harness(
        _ORIENT_HARNESS,
        "--simulator",
        "isaacsim",
        "--result-file",
        str(result_file),
        label="isaacsim/camera-orientation",
        timeout=700,
        result_file=result_file,
    )


def test_camera_obs_pipeline(tmp_path):
    # Drives render_sensors -> camera obs term -> transform -> obs dict; asserts CHW float01 format
    # and that a decimation=3 camera holds while a decimation=1 camera updates.
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
