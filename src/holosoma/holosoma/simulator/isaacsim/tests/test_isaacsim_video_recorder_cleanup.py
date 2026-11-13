"""SDK movie extraction owns frames and respects native camera ownership."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from holosoma.config_types.simulator import IsaacLab3Config
from holosoma.config_types.video import VideoConfig
from holosoma.simulator.isaacsim.backends.isaaclab2 import IsaacLab2Backend
from holosoma.simulator.isaacsim.backends.isaaclab3 import IsaacLab3Backend
from holosoma.simulator.isaacsim.video_recorder import IsaacSimVideoRecorder

pytestmark = pytest.mark.no_sim


@pytest.mark.parametrize("major", [5, 6])
def test_sdk_movie_reuses_capture_and_keeps_owned_selected_environment(major: int) -> None:
    source = torch.zeros((2, 3, 4, 4), dtype=torch.uint8)
    source[0] = 17
    source[1] = 93
    output = source[1:2] if major == 5 else SimpleNamespace(torch=source)
    events: list[str] = []
    camera = SimpleNamespace(data=SimpleNamespace(output={"rgb": output}))
    backend = IsaacLab2Backend() if major == 5 else IsaacLab3Backend(IsaacLab3Config(renderer="ovrtx"))
    simulator = SimpleNamespace(
        backend=backend,
        scene=SimpleNamespace(sensors={"holosoma_video_camera": camera}),
        video_camera=camera,
        tiled_cameras={"perception": object()},
        _sim_step_counter=4,
        render_for_sensors=lambda: events.append("render"),
    )
    recorder = IsaacSimVideoRecorder(
        VideoConfig(record_env_id=1, use_recording_thread=False, show_command_overlay=False),
        simulator,  # type: ignore[arg-type]
    )
    # The frame-end sensor hook already rendered. Encoding must use that result.
    recorder.mark_frame_ready()
    assert recorder.camera_index == (0 if major == 5 else 1)
    recorder._capture_frame_impl()
    assert events == []
    np.testing.assert_array_equal(recorder.video_frames[0], np.full((3, 4, 3), 93, dtype=np.uint8))
    source.fill_(11)
    np.testing.assert_array_equal(recorder.video_frames[0], np.full((3, 4, 3), 93, dtype=np.uint8))
    assert recorder.video_frames[0].flags.owndata

    submitted: list[list[Any]] = []

    def refresh(cameras: list[Any]) -> None:
        submitted.append(cameras)

    backend.refresh_cameras = refresh  # type: ignore[method-assign]
    recorder.prepare_frame = lambda: events.append("pose")  # type: ignore[method-assign]
    simulator._sim_step_counter += 1
    recorder._capture_frame_impl()
    assert events == ["pose", "render"]
    expected = [simulator.tiled_cameras["perception"], camera] if major == 6 else [camera]
    assert submitted == [expected]
    np.testing.assert_array_equal(recorder.video_frames[1], np.full((3, 4, 3), 11, dtype=np.uint8))
    assert simulator._sim_step_counter == 5


def test_cleanup_leaves_native_camera_owned_by_scene_and_closes_pose_view_once() -> None:
    events: list[str] = []
    camera = object()
    scene = SimpleNamespace(sensors={"holosoma_video_camera": camera})
    recorder = IsaacSimVideoRecorder(
        VideoConfig(use_recording_thread=False),
        SimpleNamespace(backend=IsaacLab3Backend(), scene=scene, video_camera=camera),  # type: ignore[arg-type]
    )
    recorder._usd_view = SimpleNamespace(close=lambda: events.append("close"))
    recorder.cleanup()
    recorder.cleanup()
    assert scene.sensors["holosoma_video_camera"] is camera
    assert recorder.camera is None
    assert recorder.simulator.video_camera is None
    assert vars(recorder)["_usd_view"] is None
    assert events == ["close"]


def test_cleanup_releases_standalone_lab2_movie_camera() -> None:
    simulator = SimpleNamespace(backend=IsaacLab2Backend(), video_camera=object())
    recorder = IsaacSimVideoRecorder(VideoConfig(use_recording_thread=False), simulator)  # type: ignore[arg-type]
    recorder.cleanup()
    assert simulator.video_camera is None
    assert recorder.camera is None


def test_cleanup_retains_pose_view_for_retry_if_close_fails() -> None:
    attempts = 0

    def close() -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("native view close failed")

    recorder = IsaacSimVideoRecorder(
        VideoConfig(use_recording_thread=False),
        SimpleNamespace(backend=IsaacLab2Backend(), video_camera=object()),  # type: ignore[arg-type]
    )
    recorder._usd_view = SimpleNamespace(close=close)
    with pytest.raises(RuntimeError, match="native view close failed"):
        recorder.cleanup()
    assert recorder._usd_view is not None
    recorder.cleanup()
    assert vars(recorder)["_usd_view"] is None
    assert attempts == 2
