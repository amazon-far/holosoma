"""Existing Holosoma perception capabilities on Isaac Lab 3.0."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.isaacsim6
pytest.importorskip("isaaclab")
pytest.importorskip("isaaclab_physx")
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("Isaac Sim requires CUDA", allow_module_level=True)

from tests.simulators._run_harness import run_harness  # noqa: E402

_HARNESS_DIR = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("simulator", ["isaacsim6", "isaacsim6-newton"])
@pytest.mark.parametrize(
    "harness",
    [
        "lidar_assert.py",
        "camera_actor_mount_assert.py",
        "camera_follow_assert.py",
        "camera_multi_assert.py",
        "camera_geometry_assert.py",
        "camera_orientation_assert.py",
        "camera_projection_assert.py",
        "camera_obs_assert.py",
    ],
)
def test_existing_perception_capability(simulator: str, harness: str) -> None:
    result = run_harness(
        _HARNESS_DIR / harness,
        "--simulator",
        simulator,
        "--num-envs",
        "2",
        label=f"{simulator}/{harness}",
        timeout=900,
    )
    assert "PASS:" in result.stdout, result.stdout[-3000:] + result.stderr[-2000:]


@pytest.mark.parametrize("simulator", ["isaacsim6", "isaacsim6-newton"])
def test_headless_recording(simulator: str, tmp_path: Path) -> None:
    result = run_harness(
        _HARNESS_DIR / "video_recording_assert.py",
        "--simulator",
        simulator,
        "--output-dir",
        str(tmp_path),
        label=f"{simulator}/recording",
        timeout=900,
    )
    assert "PASS:" in result.stdout, result.stdout[-3000:] + result.stderr[-2000:]


@pytest.mark.parametrize("simulator", ["isaacsim6", "isaacsim6-newton"])
def test_contact_and_height_buffers(simulator: str) -> None:
    result = run_harness(
        _HARNESS_DIR / "isaac_sensor_state_assert.py",
        "--simulator",
        simulator,
        label=f"{simulator}/contact-and-height",
        timeout=900,
    )
    assert "PASS:" in result.stdout, result.stdout[-3000:] + result.stderr[-2000:]
