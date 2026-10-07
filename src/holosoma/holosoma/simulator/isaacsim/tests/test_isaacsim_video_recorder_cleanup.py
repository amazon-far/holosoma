"""IsaacSim Replicator ownership cleanup tests."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.isaacsim

pytest.importorskip("isaaclab")


def test_cleanup_detaches_destroys_and_removes_camera_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    class _Annotator:
        @staticmethod
        def detach(_products: list[object]) -> None:
            events.append("detach")

    class _RenderProduct:
        @staticmethod
        def destroy() -> None:
            events.append("destroy")

    class _Stage:
        @staticmethod
        def RemovePrim(path: str) -> bool:
            assert path == "/World/VideoCamera"
            events.append("remove")
            return True

    omni = types.ModuleType("omni")
    omni_usd = types.ModuleType("omni.usd")
    omni.usd = omni_usd  # type: ignore[attr-defined]
    omni_usd.get_context = lambda: SimpleNamespace(get_stage=lambda: _Stage())  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "omni", omni)
    monkeypatch.setitem(sys.modules, "omni.usd", omni_usd)

    isaaclab_math = types.ModuleType("isaaclab.utils.math")
    isaaclab_math.create_rotation_matrix_from_view = object()  # type: ignore[attr-defined]
    isaaclab_math.quat_from_matrix = object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "isaaclab.utils.math", isaaclab_math)

    from holosoma.simulator.isaacsim.video_recorder import IsaacSimVideoRecorder

    recorder = object.__new__(IsaacSimVideoRecorder)
    recorder.config = SimpleNamespace(use_recording_thread=False)  # type: ignore[assignment]
    recorder._closed = False
    recorder._is_recording = False
    recorder.video_frames = [object()]  # type: ignore[list-item]
    recorder.recording_thread = None
    recorder._rgb_annotator = _Annotator()
    recorder._render_product = _RenderProduct()
    recorder._camera_prim_path = "/World/VideoCamera"
    recorder.camera_prim = object()
    recorder._view = object()

    recorder.cleanup()
    recorder.cleanup()

    assert events == ["detach", "destroy", "remove"]
    assert recorder._rgb_annotator is None
    assert recorder._render_product is None  # type: ignore[unreachable]  # cleanup clears the narrowed fixture.
    assert recorder._camera_prim_path is None
