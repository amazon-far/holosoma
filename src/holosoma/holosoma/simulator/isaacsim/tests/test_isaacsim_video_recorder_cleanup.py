"""IsaacSim Replicator ownership cleanup tests at the external stage boundary."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from typing import TYPE_CHECKING, TypeAlias

import pytest

if TYPE_CHECKING:
    from holosoma.simulator.isaacsim.backends.base import IsaacSimBackendAdapter
    from holosoma.simulator.isaacsim.video_recorder import IsaacSimVideoRecorder

_RecorderFixture: TypeAlias = tuple["IsaacSimVideoRecorder", list[str], "IsaacSimBackendAdapter"]

pytestmark = pytest.mark.no_sim


@pytest.fixture(params=[2], ids=["isaacsim5"])
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

    # Supply the external Kit stage, keeping the recorder cleanup in production.
    omni = types.ModuleType("omni")
    omni_usd = types.ModuleType("omni.usd")
    omni.usd = omni_usd  # type: ignore[attr-defined]

    def _kit_context() -> SimpleNamespace:
        assert request.param == 2, "Lab 3 must use its active thread-local stage"
        return SimpleNamespace(get_stage=lambda: _Stage())

    omni_usd.get_context = _kit_context  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "omni", omni)
    monkeypatch.setitem(sys.modules, "omni.usd", omni_usd)

    from holosoma.simulator.isaacsim import video_recorder
    from holosoma.simulator.isaacsim.backends.isaacsim5 import IsaacSim5Backend

    backend = IsaacSim5Backend()
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
    return recorder, events, backend


def test_cleanup_releases_view_before_removing_camera_and_is_idempotent(
    recorder_fixture: _RecorderFixture,
) -> None:
    recorder, events, backend = recorder_fixture

    recorder.cleanup()
    recorder.cleanup()

    expected = ["detach", "destroy", "remove"]
    assert events == expected
    assert recorder._rgb_annotator is None
    assert recorder._render_product is None
    assert recorder._view is None
    assert recorder._camera_prim_path is None
    assert recorder.camera_prim is None
    assert recorder.video_frames == []


def test_cleanup_keeps_camera_path_for_retry_if_stage_removal_fails(
    recorder_fixture: _RecorderFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder, events, backend = recorder_fixture
    # Replace only the external stage operation; cleanup, stage selection, and
    # backend resource ownership all stay in production code.
    original_stage = sys.modules["omni.usd"].get_context().get_stage()
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
    prefix = ["detach", "destroy"]
    assert events == prefix + ["remove", "remove"]
    assert recorder._camera_prim_path is None
