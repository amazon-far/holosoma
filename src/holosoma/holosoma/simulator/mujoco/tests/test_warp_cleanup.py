"""MuJoCo Warp rendering ownership cleanup tests."""

from __future__ import annotations

from typing import Any, cast

import pytest

pytestmark = pytest.mark.mujoco_warp

pytest.importorskip("mujoco_warp")
pytest.importorskip("warp")

from holosoma.simulator.mujoco.backends import warp_backend  # noqa: E402


def test_close_synchronizes_releases_render_state_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class _ScopedDevice:
        def __init__(self, device: object) -> None:
            assert device == "cuda:test"

        def __enter__(self) -> None:
            events.append("device.enter")

        def __exit__(self, *_args: object) -> None:
            events.append("device.exit")

    monkeypatch.setattr(warp_backend.wp, "ScopedDevice", _ScopedDevice)
    monkeypatch.setattr(warp_backend.wp, "synchronize", lambda: events.append("synchronize"))

    backend = object.__new__(warp_backend.WarpBackend)
    backend._closed = False
    backend.mjw_device = cast("Any", "cuda:test")
    backend.step_graph = object()
    backend._render_graph = object()
    backend._render_rgb_out = {1: object()}
    backend._render_depth_out = {1: object()}
    backend._render_context = object()
    backend._lidar_render_context = object()
    backend._lidar_render_context_groups = (0, 1, 2, 4)
    backend._sensor_refit_graphs = {1: object()}
    backend._sensor_refit_contexts = {1}
    backend._torch_render_stream = object()
    backend._cam_ids = {"head": 1}

    backend.close()
    backend.close()

    assert events == ["device.enter", "synchronize", "device.exit"]
    assert backend.step_graph is None
    assert backend._render_graph is None
    assert backend._render_rgb_out == {}
    assert backend._render_depth_out == {}
    assert backend._render_context is None
    state = vars(backend)
    assert state["_lidar_render_context"] is None
    assert state["_lidar_render_context_groups"] is None
    assert state["_sensor_refit_graphs"] == {}
    assert state["_sensor_refit_contexts"] == set()
    assert state["_torch_render_stream"] is None
    assert backend._cam_ids == {}
    assert backend._closed
