"""Live IsaacGym camera tests (GPU), one sim per subprocess.

Each test runs a camera assert harness against the IsaacGym backend. The geometry harness mounts
a camera on the robot base looking at a known red panel and checks the [N,H,W,3] uint8 frame's
panel silhouette is centered, square, and FOV-consistent.

importorskip("isaacgym") and CUDA-gated. Collected by the IsaacGym CI job (-m "not isaacsim").
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("isaacgym")
from holosoma.utils.safe_torch_import import torch

if not torch.cuda.is_available():
    pytest.skip("IsaacGym requires a CUDA device", allow_module_level=True)

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


def test_camera_geometry(tmp_path):
    result_file = tmp_path / "geo_isaacgym.txt"
    run_harness(
        _HARNESS,
        "--simulator",
        "isaacgym",
        "--result-file",
        str(result_file),
        label="isaacgym/camera-geometry",
        timeout=600,
        result_file=result_file,
    )


def test_camera_geometry_multi_env(tmp_path):
    # All envs co-located: every env renders the same geometry.
    result_file = tmp_path / "geo_isaacgym_multi.txt"
    run_harness(
        _HARNESS,
        "--simulator",
        "isaacgym",
        "--num-envs",
        "4",
        "--result-file",
        str(result_file),
        label="isaacgym/camera-geometry (num_envs=4)",
        timeout=600,
        result_file=result_file,
    )


def test_camera_depth(tmp_path):
    # Depth in positive meters, image-plane, +inf no-hit. Panel at a known distance reads ~that.
    result_file = tmp_path / "depth_isaacgym.txt"
    run_harness(
        _DEPTH_HARNESS,
        "--simulator",
        "isaacgym",
        "--result-file",
        str(result_file),
        label="isaacgym/camera-depth",
        timeout=600,
        result_file=result_file,
    )


def test_camera_recorder(tmp_path):
    # A viz config requesting recording yields a non-None camera_sensor_recorder that capture()s a frame.
    result_file = tmp_path / "rec_isaacgym.txt"
    run_harness(
        _CONTRACT_HARNESS,
        "--simulator",
        "isaacgym",
        "--check-recorder",
        "--result-file",
        str(result_file),
        label="isaacgym/camera-recorder",
        timeout=600,
        result_file=result_file,
    )


def test_camera_follow(tmp_path):
    # Camera tracks its mount body: move the robot, the panel depth drops by the moved distance.
    result_file = tmp_path / "follow_isaacgym.txt"
    run_harness(
        _FOLLOW_HARNESS,
        "--simulator",
        "isaacgym",
        "--result-file",
        str(result_file),
        label="isaacgym/camera-follow",
        timeout=600,
        result_file=result_file,
    )


def test_camera_follow_multi_env(tmp_path):
    # Camera tracks its mount body: move the robot, the panel depth drops by the moved distance.
    # Multi-env: each env's camera follows its own mount body.
    result_file = tmp_path / "follow_isaacgym_multi.txt"
    run_harness(
        _FOLLOW_HARNESS,
        "--simulator",
        "isaacgym",
        "--num-envs",
        "4",
        "--result-file",
        str(result_file),
        label="isaacgym/camera-follow (num_envs=4)",
        timeout=600,
        result_file=result_file,
    )


def test_camera_multi(tmp_path):
    # Two cameras with different views each return their own frame (no buffer mixup).
    result_file = tmp_path / "multi_isaacgym.txt"
    run_harness(
        _MULTI_HARNESS,
        "--simulator",
        "isaacgym",
        "--result-file",
        str(result_file),
        label="isaacgym/camera-multi",
        timeout=600,
        result_file=result_file,
    )


def test_camera_actor_mount(tmp_path):
    # Exercises target_kind='actor': a camera on a scene object resolves, renders, and follows it.
    result_file = tmp_path / "actor_isaacgym.txt"
    run_harness(
        _ACTOR_HARNESS,
        "--simulator",
        "isaacgym",
        "--result-file",
        str(result_file),
        label="isaacgym/camera-actor-mount",
        timeout=600,
        result_file=result_file,
    )


@pytest.mark.parametrize("num_envs", ["1", "3"])
def test_camera_world_mount(tmp_path, num_envs):
    # target_kind='world': a free-floating camera fixed in the env frame does NOT follow the robot,
    # and each env's camera sits at its own origin (set_camera_transform, not attach_camera_to_body).
    result_file = tmp_path / f"world_isaacgym_{num_envs}.txt"
    run_harness(
        _WORLD_HARNESS,
        "--simulator",
        "isaacgym",
        "--num-envs",
        num_envs,
        "--result-file",
        str(result_file),
        label=f"isaacgym/camera-world-mount (num_envs={num_envs})",
        timeout=600,
        result_file=result_file,
    )


def test_camera_orientation(tmp_path):
    # Off-axis panel on a non-square camera lands in the expected image quadrant (catches mirror/flip/transpose).
    result_file = tmp_path / "orient_isaacgym.txt"
    run_harness(
        _ORIENT_HARNESS,
        "--simulator",
        "isaacgym",
        "--result-file",
        str(result_file),
        label="isaacgym/camera-orientation",
        timeout=600,
        result_file=result_file,
    )


def test_camera_obs_pipeline(tmp_path):
    # Drives render_sensors -> camera obs term -> transform -> obs dict; asserts CHW float01 format
    # and that a decimation=3 camera holds while a decimation=1 camera updates.
    result_file = tmp_path / "obs_isaacgym.txt"
    run_harness(
        _OBS_HARNESS,
        "--simulator",
        "isaacgym",
        "--result-file",
        str(result_file),
        label="isaacgym/camera-obs-pipeline",
        timeout=600,
        result_file=result_file,
    )
