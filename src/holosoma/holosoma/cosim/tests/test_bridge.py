"""Backend-neutral receiver and metadata control without GPU simulator SDKs."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from holosoma.cosim import bridge
from holosoma.cosim.cuda import CacheAllocator
from holosoma.envs.render_task import RenderTask

pytestmark = pytest.mark.no_sim


@pytest.mark.parametrize("frame", [(0, 0, 0), (1, -1, 0), (1, 0, float("nan")), (1, 0, -1)])
def test_invalid_frame_is_rejected(frame: tuple[int, int, float]) -> None:
    with pytest.raises(ValueError, match="must be valid"):
        bridge.Frame(*frame)


ReceiverFixture = tuple[bridge.Receiver, list[Any], list[dict[str, Any]], list[dict[str, Any]]]


@pytest.fixture
def receiver(monkeypatch: pytest.MonkeyPatch) -> ReceiverFixture:
    calls: list[Any] = []
    replies: list[dict[str, Any]] = []
    values = {"robot_root_states": torch.ones(1, 13), "dof_pos": torch.zeros(1, 2), "dof_vel": torch.zeros(1, 2)}
    messages: list[dict[str, Any]] = [
        {
            "type": "OPEN",
            "buffers": [],
            "ready_handle": "00",
            "gpu_uuid": "gpu-a",
            "dof_names": ["a", "b"],
            "object_names": [],
            "env_origins": [[0.0, 0.0, 0.0]],
        },
        {"type": "FRAME_AVAILABLE", "frame": {"sequence": 1, "physics_step": 7, "sim_time": 0.035}},
        {"type": "CLOSE"},
    ]
    socket = SimpleNamespace(send_json=replies.append)
    control = SimpleNamespace(
        socket=socket,
        receive=lambda: messages.pop(0),
        set_timeout=lambda ms: calls.append(("timeout", ms)),
        close=lambda: calls.append("close"),
    )
    monkeypatch.setattr(bridge, "ControlChannel", lambda **_: control)

    class Mapped:
        release_error: Exception | None = None

        def __init__(self, descriptions: Any, *, device: str) -> None:
            self.tensors = values

        def verify_read_only(self) -> None:
            calls.append("readonly")

        def release(self) -> None:
            calls.append("unmap")
            if self.release_error is not None:
                raise self.release_error

    monkeypatch.setattr(bridge, "MappedBuffers", Mapped)
    stream = SimpleNamespace(wait_event=lambda _: calls.append("ready"), record_event=lambda _: calls.append("done"))
    monkeypatch.setitem(
        sys.modules,
        "warp",
        SimpleNamespace(
            stream_from_torch=lambda _: stream,
            event_from_ipc_handle=lambda *_args, **_kw: object(),
            Event=lambda **_: SimpleNamespace(ipc_handle=lambda: b"\x01"),
        ),
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _: object())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _: calls.append("sync"))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(uuid="gpu-a"))
    sim = SimpleNamespace(
        sim_device="cuda:0",
        dof_names=["a", "b"],
        object_state_names=(),
        scene=SimpleNamespace(env_origins=torch.zeros(1, 3)),
        state=SimpleNamespace(writable_names=lambda: list(values)),
        apply_state=lambda inputs: calls.append(("apply", inputs is values)),
    )
    task = SimpleNamespace(simulator=sim, render_frame=lambda **kw: calls.append(("capture", kw["sim_time"])))
    return bridge.Receiver(cast("RenderTask", task), endpoint="ipc:///unused"), calls, replies, messages


def test_receiver_uses_only_native_state_and_capture_contracts(receiver: ReceiverFixture) -> None:
    worker, calls, replies, _messages = receiver
    worker.run()
    assert calls == [
        "ready",
        "sync",
        ("apply", True),
        ("capture", 0.035),
        "readonly",
        "done",
        ("timeout", bridge.FRAME_TIMEOUT_MS),
        "sync",
        "unmap",
        "close",
    ]
    assert replies == [
        {"type": "OPENED", "done_handle": "01"},
        {"type": "RENDER_COMPLETE", "sequence": 1},
        {"type": "UNMAPPED"},
    ]


@pytest.mark.parametrize("mismatch", ["dof_names", "object_names", "env_origins", "gpu_uuid"])
def test_layout_mismatch_fails_before_mapping(receiver: ReceiverFixture, mismatch: str) -> None:
    worker, calls, replies, messages = receiver
    messages[0][mismatch] = ["wrong"]
    with pytest.raises(ValueError, match="differs|differ"):
        worker.run()
    assert not any(isinstance(call, tuple) for call in calls)
    assert replies[-1]["type"] == "ERROR" and calls[-1] == "close"


def test_unwritable_state_field_fails_before_mapping(receiver: ReceiverFixture) -> None:
    worker, calls, replies, messages = receiver
    messages[0]["buffers"] = [{"name": "contact_forces", "shape": [1, 3], "handle": "00"}]
    with pytest.raises(ValueError, match="cannot apply state fields"):
        worker.run()
    assert not any(isinstance(call, tuple) for call in calls)
    assert replies[-1]["type"] == "ERROR" and calls[-1] == "close"


def test_duplicate_frame_is_rejected_and_import_is_released(
    receiver: ReceiverFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker, calls, replies, messages = receiver
    messages.insert(2, dict(messages[1]))
    monkeypatch.setattr(bridge.MappedBuffers, "release_error", RuntimeError("cleanup failed"))
    with pytest.raises(ValueError, match="consecutive"):
        worker.run()
    assert len([call for call in calls if isinstance(call, tuple) and call[0] == "capture"]) == 1
    assert replies[-1]["type"] == "ERROR" and calls[-2:] == ["unmap", "close"]


def test_control_timeout_is_propagated_and_socket_can_be_closed(tmp_path: Path) -> None:
    import zmq

    control = bridge.ControlChannel(endpoint=f"ipc://{tmp_path}/absent.sock", startup_timeout_ms=10)
    try:
        with pytest.raises(zmq.Again):
            control.exchange("OPEN")
    finally:
        control.close()


def test_publisher_rejects_fields_the_allocator_does_not_own(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {"dof_pos": torch.zeros(1, 2), "dof_vel": torch.zeros(1, 2)}
    state = SimpleNamespace(writable_names=lambda: list(values), get=values.__getitem__)
    monkeypatch.setitem(sys.modules, "warp", SimpleNamespace())
    monkeypatch.setattr(bridge, "ControlChannel", lambda **_: pytest.fail("connected before validation"))
    with pytest.raises(ValueError, match=r"does not own published state fields \['dof_vel'\]"):
        bridge.Publisher(
            cast("Any", SimpleNamespace(state=state)), CacheAllocator(("dof_pos",)), endpoint="ipc:///unused"
        )
