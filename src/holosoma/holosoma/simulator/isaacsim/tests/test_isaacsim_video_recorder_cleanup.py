"""IsaacSim Replicator ownership cleanup tests at the external stage boundary."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from typing import TYPE_CHECKING, TypeAlias

import pytest

if TYPE_CHECKING:
    from holosoma.simulator.isaacsim.backends.base import IsaacLabBackend
    from holosoma.simulator.isaacsim.video_recorder import IsaacSimVideoRecorder

_RecorderFixture: TypeAlias = tuple["IsaacSimVideoRecorder", list[str], "IsaacLabBackend"]

pytestmark = pytest.mark.no_sim


@pytest.fixture(params=[2, 3], ids=["isaaclab2", "isaaclab3"])
def recorder_fixture(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> _RecorderFixture:
    events: list[str] = []

    class _Annotator:
        @staticmethod
        def detach(products: list[str]) -> None:
            assert products == ["/Render/VideoProduct"]
            events.append("detach")

    class _RenderProduct:
        path = "/Render/VideoProduct"

        @staticmethod
        def destroy() -> None:
            events.append("destroy")

    class _Stage:
        @staticmethod
        def RemovePrim(path: str) -> bool:
            assert path == "/World/VideoCamera"
            events.append("remove")
            return True

    class _View:
        @staticmethod
        def close() -> None:
            events.append("close")

    # Keep both production backend adapters. Supply their external stage accessors
    # independently, so using Kit's stage on Lab 3 fails instead of silently passing.
    omni = types.ModuleType("omni")
    omni_usd = types.ModuleType("omni.usd")
    omni.usd = omni_usd  # type: ignore[attr-defined]

    def _kit_context() -> SimpleNamespace:
        assert request.param == 2, "Lab 3 must use its active thread-local stage"
        return SimpleNamespace(get_stage=lambda: _Stage())

    omni_usd.get_context = _kit_context  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "omni", omni)
    monkeypatch.setitem(sys.modules, "omni.usd", omni_usd)

    lab_stage = types.ModuleType("isaaclab.sim.utils.stage")

    def _active_stage() -> _Stage:
        assert request.param == 3, "Lab 2 must use its Kit stage"
        return _Stage()

    lab_stage.get_current_stage = _active_stage  # type: ignore[attr-defined]
    lab_utils = types.ModuleType("isaaclab.sim.utils")
    lab_utils.stage = lab_stage  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "isaaclab.sim.utils", lab_utils)
    monkeypatch.setitem(sys.modules, "isaaclab.sim.utils.stage", lab_stage)

    from holosoma.simulator.isaacsim import video_recorder
    from holosoma.simulator.isaacsim.backends.isaaclab2 import IsaacLab2Backend
    from holosoma.simulator.isaacsim.backends.isaaclab3 import IsaacLab3Backend

    backend = IsaacLab3Backend() if request.param == 3 else IsaacLab2Backend()
    recorder = object.__new__(video_recorder.IsaacSimVideoRecorder)
    recorder.simulator = SimpleNamespace(backend=backend)  # type: ignore[assignment]
    recorder.config = SimpleNamespace(use_recording_thread=False)  # type: ignore[assignment]
    recorder._closed = False
    recorder._is_recording = False
    recorder.video_frames = [object()]  # type: ignore[list-item]
    recorder.recording_thread = None
    recorder._rgb_annotator = _Annotator()
    recorder._render_product = _RenderProduct()
    recorder._camera_prim_path = "/World/VideoCamera"
    recorder.camera_prim = object()
    recorder._view = _View()
    recorder._rtx_capture = None
    return recorder, events, backend


def test_cleanup_releases_view_before_removing_camera_and_is_idempotent(
    recorder_fixture: _RecorderFixture,
) -> None:
    recorder, events, backend = recorder_fixture

    recorder.cleanup()
    recorder.cleanup()

    expected = ["detach", "destroy", "close", "remove"] if backend.is_isaaclab3 else ["detach", "destroy", "remove"]
    assert events == expected
    assert recorder._rgb_annotator is None
    assert recorder._render_product is None
    assert recorder._view is None
    assert recorder._camera_prim_path is None
    assert recorder.camera_prim is None
    assert recorder.video_frames == []


@pytest.mark.parametrize("recorder_fixture", [3], indirect=True, ids=["isaaclab3"])
def test_cleanup_keeps_camera_for_retry_if_view_close_fails(recorder_fixture: _RecorderFixture) -> None:
    recorder, events, _backend = recorder_fixture

    class _FailOnceView:
        calls = 0

        def close(self) -> None:
            self.calls += 1
            events.append("close")
            if self.calls == 1:
                raise RuntimeError("view close failed")

    view = _FailOnceView()
    recorder._view = view
    with pytest.raises(RuntimeError, match="view close failed"):
        recorder.cleanup()

    assert events == ["detach", "destroy", "close"]
    assert recorder._view is view
    assert recorder._camera_prim_path == "/World/VideoCamera"
    assert recorder.camera_prim is not None

    recorder.cleanup()
    assert events == ["detach", "destroy", "close", "close", "remove"]
    assert getattr(recorder, "_view") is None
    assert recorder._camera_prim_path is None


def test_cleanup_keeps_camera_path_for_retry_if_stage_removal_fails(
    recorder_fixture: _RecorderFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder, events, backend = recorder_fixture
    # Replace only the external stage operation; cleanup, stage selection, and
    # backend resource ownership all stay in production code.
    module = sys.modules[recorder.__class__.__module__]
    original_stage = module._current_stage(backend)
    attempts = 0

    def _remove(path: str) -> bool:
        nonlocal attempts
        assert path == "/World/VideoCamera"
        attempts += 1
        events.append("remove")
        return attempts > 1

    monkeypatch.setattr(type(original_stage), "RemovePrim", staticmethod(_remove))

    with pytest.raises(RuntimeError, match="Failed to remove camera prim"):
        recorder.cleanup()

    assert recorder._camera_prim_path == "/World/VideoCamera"
    assert recorder.camera_prim is not None
    assert recorder._view is None
    recorder.cleanup()
    prefix = ["detach", "destroy", "close"] if backend.is_isaaclab3 else ["detach", "destroy"]
    assert events == prefix + ["remove", "remove"]
    assert recorder._camera_prim_path is None
