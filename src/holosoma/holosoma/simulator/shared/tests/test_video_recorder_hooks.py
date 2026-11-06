"""Video hooks must honor the configured environment on a global frame event."""

from unittest.mock import Mock

import pytest

from holosoma.config_types.video import FixedCameraConfig, VideoConfig
from holosoma.simulator.base_simulator.hooks import HookRegistry, Phase
from holosoma.simulator.shared.video_recorder import VideoRecorderInterface

pytestmark = pytest.mark.no_sim


class _Recorder(VideoRecorderInterface):
    frames = 0

    def _capture_frame_impl(self) -> None:
        self.frames += 1


@pytest.mark.parametrize("env_id", [0, 2])
def test_frame_hook_captures_selected_environment(env_id: int) -> None:
    config = VideoConfig(record_env_id=env_id, camera=FixedCameraConfig(), use_recording_thread=False)
    recorder = _Recorder(config, Mock(num_envs=3))
    hooks = HookRegistry()
    recorder.register_hooks(hooks)

    hooks.emit(Phase.FRAME_END)
    assert recorder.frames == 0
    recorder._is_recording = True
    hooks.emit(Phase.FRAME_END)
    assert recorder.frames == 1
    recorder.capture_frame(env_id=(env_id + 1) % 3)
    assert recorder.frames == 1
