"""Focused video worker shutdown tests."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from holosoma.simulator.shared.video_recorder import VideoRecorderInterface

pytestmark = pytest.mark.no_sim


class _Thread:
    def __init__(self, events: list[str], *, alive: bool) -> None:
        self.events = events
        self.alive = alive

    def join(self, timeout: float | None = None) -> None:
        assert timeout == 30.0
        self.events.append("join")

    def is_alive(self) -> bool:
        return self.alive


class _Recorder(VideoRecorderInterface):
    def _capture_frame_impl(self) -> None:
        return


def _recorder(events: list[str], *, worker_alive: bool) -> _Recorder:
    recorder = object.__new__(_Recorder)
    recorder.config = SimpleNamespace(use_recording_thread=True)  # type: ignore[assignment]
    recorder._closed = False
    recorder._is_recording = False
    recorder.video_frames = [object()]  # type: ignore[list-item]
    recorder.recording_thread = _Thread(events, alive=worker_alive)  # type: ignore[assignment]
    recorder.stop_recording_event = threading.Event()
    recorder.thread_active = True
    return recorder


def test_live_worker_blocks_provider_teardown_and_close_is_one_pass() -> None:
    events: list[str] = []
    recorder = _recorder(events, worker_alive=True)

    with pytest.raises(RuntimeError, match="did not terminate"):
        recorder.cleanup()

    assert events == ["join"]
    assert recorder.video_frames
    assert recorder.thread_active

    recorder.cleanup()
    assert events == ["join"]


def test_stopped_worker_releases_shared_state_once() -> None:
    events: list[str] = []
    recorder = _recorder(events, worker_alive=False)

    recorder.cleanup()
    recorder.cleanup()

    assert events == ["join"]
    assert recorder.video_frames == []
    assert recorder.recording_thread is None
    assert not recorder.thread_active
