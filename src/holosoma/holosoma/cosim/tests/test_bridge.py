"""Backend-neutral receiver and metadata control without GPU simulator SDKs."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from holosoma.config_types.plugin import CosimRenderPluginConfig
from holosoma.config_types.sensor import CameraSensorConfig, SensorMountConfig
from holosoma.cosim import base as cosim_base
from holosoma.cosim import bridge
from holosoma.cosim import cuda as cosim_cuda
from holosoma.cosim.base import BufferHandler
from holosoma.cosim.cuda import BufferSpec, CacheAllocator
from holosoma.cosim.render.plugin import CosimRenderPlugin, camera_requests
from holosoma.cosim.render.renderer import RenderHandler, RenderTask
from holosoma.simulator.base_simulator.hooks import HookRegistry, Phase
from holosoma.simulator.shared.sensor_manager import SensorManager

pytestmark = pytest.mark.no_sim


@pytest.mark.parametrize("frame", [(0, 0), (1, float("nan")), (1, -1)])
def test_invalid_frame_is_rejected(frame: tuple[int, float]) -> None:
    with pytest.raises(ValueError, match="must be valid"):
        bridge.Frame(*frame)


def _camera(manager: SensorManager, name: str, *, remote: str | None, data_types: list[Any]) -> Any:
    mount = SensorMountConfig(target_kind="world", target="")
    config = CameraSensorConfig(mount=mount, width=4, height=3, data_types=data_types, remote=remote)
    return manager.register_camera(name, config)


def _fake_cuda(monkeypatch: pytest.MonkeyPatch, stream: Any, synchronize: Any = lambda _: None) -> None:
    """Replace Warp and Torch CUDA streams with fakes whose events go through ``stream``."""
    monkeypatch.setitem(
        sys.modules,
        "warp",
        SimpleNamespace(
            stream_from_torch=lambda _: stream,
            event_from_ipc_handle=lambda *_args, **_kw: object(),
            Event=lambda **_: SimpleNamespace(ipc_handle=lambda: b"\x01"),
            to_torch=lambda owner: owner,
        ),
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _: object())
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)


ReceiverFixture = tuple[bridge.Receiver, list[Any], list[dict[str, Any]], list[dict[str, Any]]]


@pytest.fixture
def receiver(monkeypatch: pytest.MonkeyPatch) -> ReceiverFixture:
    calls: list[Any] = []
    replies: list[dict[str, Any]] = []
    values = {"robot_root_states": torch.ones(1, 13), "dof_pos": torch.zeros(1, 2), "dof_vel": torch.zeros(1, 2)}
    messages: list[dict[str, Any]] = [
        {
            "type": "OPEN",
            "exports": [],
            "requests": [],
            "ready_handle": "00",
            "gpu_uuid": "gpu-a",
            "source_pid": os.getpid(),
            "meta": {"dof_names": ["a", "b"], "object_names": [], "env_origins": [[0.0, 0.0, 0.0]]},
        },
        {"type": "FRAME_AVAILABLE", "frame": {"sequence": 1, "sim_time": 0.035}},
        {"type": "CLOSE"},
    ]
    socket = SimpleNamespace(send_json=replies.append, poll=lambda _ms: True)
    control = SimpleNamespace(
        socket=socket,
        receive=lambda: messages.pop(0),
        set_timeout=lambda ms: calls.append(("timeout", ms)),
        close=lambda: calls.append("close"),
    )
    monkeypatch.setattr(bridge, "ControlChannel", lambda **_: control)

    class Mapped:
        release_error: Exception | None = None
        names: list[str] = []

        def __init__(self, descriptions: Any, *, device: str) -> None:
            calls.append("map")
            Mapped.names = [spec.name for spec in descriptions]
            self.tensors = values

        def verify_read_only(self) -> None:
            calls.append("readonly")

        def release(self) -> None:
            calls.append("unmap")
            if self.release_error is not None:
                raise self.release_error

    monkeypatch.setattr(bridge, "MappedBuffers", Mapped)
    stream = SimpleNamespace(wait_event=lambda _: calls.append("ready"), record_event=lambda _: calls.append("done"))
    _fake_cuda(monkeypatch, stream, synchronize=lambda _: calls.append("sync"))
    monkeypatch.setattr(
        cosim_base, "allocate_shared", lambda shape, dtype, _: torch.zeros(shape, dtype=getattr(torch, dtype))
    )
    monkeypatch.setattr(
        bridge, "share", lambda name, owner: BufferSpec(name, tuple(owner.shape), "00", str(owner.dtype)[6:])
    )
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(uuid="gpu-a"))
    sim = SimpleNamespace(
        sim_device="cuda:0",
        dof_names=["a", "b"],
        body_names=["pelvis"],
        object_state_names=(),
        scene=SimpleNamespace(env_origins=torch.zeros(1, 3)),
        state=SimpleNamespace(writable_names=lambda: list(values), get=values.__getitem__),
        apply_state=lambda inputs: calls.append(("apply", inputs is values)),
        set_kinematic_time=lambda sim_time: calls.append(("clock", sim_time)),
        sensor_manager=SensorManager("cpu", 50.0),
        match_env_origins=lambda origins: calls.append(("place", origins.tolist())),
    )
    task = SimpleNamespace(
        simulator=sim,
        num_envs=1,
        render_frame=lambda: calls.append("capture"),
    )
    handler = RenderHandler(cast("RenderTask", task))
    return bridge.Receiver(handler, endpoint="ipc:///unused"), calls, replies, messages


def _in_order(calls: list[Any], *expected: Any) -> bool:
    """Whether ``expected`` occurs in ``calls`` in this order, possibly with other calls between."""
    remaining = iter(calls)
    return all(item in remaining for item in expected)


def test_receiver_frame_lifecycle(receiver: ReceiverFixture) -> None:
    worker, calls, replies, _messages = receiver
    worker.run()
    # Origins are matched before mapping; each frame waits for the source, renders, then signals done.
    assert _in_order(calls, ("place", [[0.0, 0.0, 0.0]]), "map", "ready", ("clock", 0.035), ("apply", True), "capture")
    assert _in_order(calls, "capture", "readonly", "done", "unmap", "close")
    assert replies == [
        {"type": "OPENED", "done_handle": "01", "exports": [], "reads": []},
        {"type": "FRAME_COMPLETE", "sequence": 1},
        {"type": "UNMAPPED"},
    ]


@pytest.mark.parametrize("mismatch", ["dof_names", "object_names", "env_origins", "gpu_uuid", "state_layout"])
def test_layout_mismatch_fails_before_mapping(receiver: ReceiverFixture, mismatch: str) -> None:
    worker, calls, replies, messages = receiver
    if mismatch == "state_layout":
        messages[0]["exports"] = [{"name": "dof_pos", "shape": [1, 3], "handle": "00"}]
    else:
        wrong: list[Any] = [[1.0, 0.0, 0.0]] if mismatch == "env_origins" else ["wrong"]
        (messages[0] if mismatch == "gpu_uuid" else messages[0]["meta"])[mismatch] = wrong
    with pytest.raises(ValueError, match="differs|differ"):
        worker.run()
    assert not any(isinstance(call, tuple) and call[0] != "place" for call in calls)
    assert replies[-1]["type"] == "ERROR" and calls[-1] == "close"


@pytest.mark.parametrize(
    "change",
    [
        {"name": "missing/rgb"},
        {"name": "cam/depth"},
        {"shape": (1, 3, 8, 3)},
        {"dtype": "float32"},
        {"meta": {"vertical_fov": 90.0}},
    ],
)
def test_remote_camera_mismatch_fails_before_mapping(receiver: ReceiverFixture, change: dict[str, Any]) -> None:
    worker, calls, replies, messages = receiver
    record = _camera(cast("Any", worker.handler).simulator.sensor_manager, "cam", remote=None, data_types=["rgb"])
    (request,) = camera_requests(record, 1)
    if "meta" in change:
        change = {"meta": {**request.meta, **change["meta"]}}
    messages[0]["requests"] = [bridge.as_received(asdict(replace(request, **change)))]
    with pytest.raises(ValueError, match="differs|no camera output"):
        worker.run()
    assert "map" not in calls
    assert replies[-1]["type"] == "ERROR" and calls[-1] == "close"


class _ReadingHandler(BufferHandler):
    """A peer that only reads ``fields``, with no simulator."""

    def __init__(self, fields: tuple[str, ...]) -> None:
        super().__init__("cuda:0")
        self.fields = fields
        self.seen: list[dict[str, Any]] = []

    def reads(self, exports: Any) -> tuple[str, ...]:
        return self.fields

    def step(self, frame: Any, state: Any) -> None:
        self.seen.append(dict(state))


@pytest.mark.parametrize("reader", [False, True])
def test_receiver_maps_only_the_fields_its_handler_reads(receiver: ReceiverFixture, reader: bool) -> None:
    # The render handler reads what it can write; another handler may read derived fields without applying them.
    worker, calls, replies, messages = receiver
    messages[0]["exports"] = [
        {"name": "dof_pos", "shape": [1, 2], "handle": "00"},
        {"name": "contact_forces", "shape": [1, 3], "handle": "00"},
    ]
    handler = _ReadingHandler(("contact_forces",)) if reader else worker.handler
    bridge.Receiver(cast("Any", handler), endpoint="ipc:///unused").run()
    expected = ["contact_forces"] if reader else ["dof_pos"]
    assert replies[0]["reads"] == expected
    assert bridge.MappedBuffers.names == expected  # type: ignore[attr-defined]
    if reader:
        assert len(cast("Any", handler).seen) == 1
        assert not any(isinstance(call, tuple) and call[0] == "apply" for call in calls)


def test_reading_a_field_the_source_does_not_share_fails_before_mapping(receiver: ReceiverFixture) -> None:
    worker, calls, replies, _messages = receiver
    handler = _ReadingHandler(("contact_forces",))
    with pytest.raises(ValueError, match="does not share"):
        bridge.Receiver(cast("Any", handler), endpoint="ipc:///unused").run()
    assert "map" not in calls and replies[-1]["type"] == "ERROR"


def test_duplicate_frame_is_rejected_and_import_is_released(
    receiver: ReceiverFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker, calls, replies, messages = receiver
    messages.insert(2, dict(messages[1]))
    monkeypatch.setattr(bridge.MappedBuffers, "release_error", RuntimeError("cleanup failed"))
    with pytest.raises(ValueError, match="consecutive"):
        worker.run()
    assert calls.count("capture") == 1
    assert replies[-1]["type"] == "ERROR" and calls[-2:] == ["unmap", "close"]


def test_receiver_stops_waiting_when_the_source_exits(receiver: ReceiverFixture) -> None:
    worker, calls, _replies, messages = receiver
    source = subprocess.Popen([sys.executable, "-c", "pass"])
    source.wait()
    messages[0]["source_pid"] = source.pid
    cast("Any", worker.control).socket.poll = lambda _ms: False
    with pytest.raises(RuntimeError, match="exited without closing"):
        worker.run()
    assert calls[-2:] == ["unmap", "close"]


def test_exchange_after_an_interrupted_request_takes_the_right_reply(tmp_path: Path) -> None:
    import threading

    import zmq

    endpoint = f"ipc://{tmp_path}/control.sock"
    receiver = bridge.ControlChannel(endpoint=endpoint, receiver=True, startup_timeout_ms=3000)
    control = bridge.ControlChannel(endpoint=endpoint, startup_timeout_ms=100)

    def serve() -> None:
        for _ in range(2):  # Ends by timeout if the second request never arrives.
            message = receiver.receive()
            if message["type"] == "FRAME_AVAILABLE":
                threading.Event().wait(0.3)
            receiver.socket.send_json({"type": f"{message['type']}_REPLY"})

    server = threading.Thread(target=serve, daemon=True)
    server.start()
    try:
        with pytest.raises(zmq.Again):
            control.exchange("FRAME_AVAILABLE")
        control.set_timeout(2000)
        assert control.exchange("CLOSE") == {"type": "CLOSE_REPLY"}
    finally:
        server.join(timeout=5)
        control.close()
        receiver.close()


def test_publisher_without_shared_state_fails_before_connecting(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {"dof_pos": torch.zeros(1, 2)}
    state = SimpleNamespace(names=lambda: list(values), get=values.__getitem__)
    monkeypatch.setitem(sys.modules, "warp", SimpleNamespace())
    monkeypatch.setattr(bridge, "ControlChannel", lambda **_: pytest.fail("connected before validation"))
    with pytest.raises(ValueError, match="shares no state"):
        bridge.Publisher(
            cast("Any", SimpleNamespace(state=state)), CacheAllocator(("dof_pos",)), endpoint="ipc:///unused"
        )


class _Simulator:
    """A stand-in simulator that, like a real one, is hashable and weak-referenceable."""

    def __init__(self, **attributes: Any) -> None:
        self.__dict__.update(attributes)


PluginFixture = tuple[CosimRenderPlugin, SensorManager, dict[str, torch.Tensor], list[Any]]


@pytest.fixture
def plugin(monkeypatch: pytest.MonkeyPatch) -> PluginFixture:
    manager = SensorManager("cpu", 50.0, num_envs=2)
    _camera(manager, "local", remote=None, data_types=["rgb"])
    _camera(manager, "cam", remote="cosim", data_types=["rgb", "depth"])
    _camera(manager, "elsewhere", remote="other", data_types=["rgb"])  # another plugin's camera
    images = {"cam/rgb": torch.ones(2, 3, 4, 3, dtype=torch.uint8), "cam/depth": torch.ones(2, 3, 4, 1)}
    calls: list[Any] = []
    replies: dict[str, dict[str, Any]] = {
        "OPEN": {
            "type": "OPENED",
            "done_handle": "00",
            "exports": [
                {"name": name, "shape": list(image.shape), "dtype": str(image.dtype)[6:], "handle": "00"}
                for name, image in images.items()
            ],
            "reads": ["dof_pos"],
        },
        "CLOSE": {"type": "UNMAPPED"},
    }

    def exchange(command: str, **fields: Any) -> dict[str, Any]:
        calls.append((command, fields))
        if command == "FRAME_AVAILABLE":
            return {"type": "FRAME_COMPLETE", "sequence": fields["frame"]["sequence"]}
        return replies[command]

    control = SimpleNamespace(exchange=exchange, set_timeout=lambda _: None, close=lambda: None)
    monkeypatch.setattr(bridge, "ControlChannel", lambda **_: control)
    mapped = SimpleNamespace(tensors=images, release=lambda: calls.append("unmap"))
    monkeypatch.setattr(bridge, "MappedBuffers", lambda *_args, **_kw: mapped)
    monkeypatch.setattr(bridge, "gpu_uuid", lambda _: "gpu-a")
    _fake_cuda(monkeypatch, SimpleNamespace(wait_event=lambda _: None, record_event=lambda _: None))
    values = {"dof_pos": torch.zeros(2, 1), "dof_vel": torch.zeros(2, 1)}
    sim = _Simulator(
        sim_device="cuda:0",
        state=SimpleNamespace(names=lambda: list(values), get=values.__getitem__, set_buffer_allocator=lambda _: None),
        sensor_manager=manager,
        hooks=HookRegistry(),
        dof_names=["a"],
        body_names=["pelvis"],
        object_state_names=(),
        scene=SimpleNamespace(env_origins=torch.zeros(2, 3)),
        time=lambda: 0.5,
    )
    monkeypatch.setattr(
        cosim_cuda,
        "CacheAllocator",
        lambda fields: SimpleNamespace(fields=frozenset(fields), owners=dict.fromkeys(fields), exports=lambda _: []),
    )
    fake: Any = sim
    fake.sensor_capture_hook = fake.hooks.add(Phase.FRAME_END, manager.collect_due, name="sensors.capture")
    fake.sensor_render_hook = fake.hooks.add(
        Phase.FRAME_END, lambda: None, name="sensors.render", after=fake.sensor_capture_hook
    )
    with fake.hooks.installing("cosim"):
        source = CosimRenderPlugin(CosimRenderPluginConfig(endpoint="ipc:///unused"), cast("Any", sim))
    return source, manager, images, calls


def test_remote_due_cameras_get_copies_of_the_receiver_images(plugin: PluginFixture) -> None:
    source, manager, images, calls = plugin
    assert [request["name"] for request in calls[0][1]["requests"]] == ["cam/rgb", "cam/depth"]
    hooks = source.simulator.hooks
    remote = manager.get("cam")
    seen: list[set[str]] = []
    hooks.add(Phase.FRAME_END, lambda: seen.append(set(remote.buffers)), after=source.simulator.sensor_render_hook)
    hooks.emit(Phase.FRAME_END)
    assert calls[1][0] == "FRAME_AVAILABLE" and calls[1][1]["frame"]["phase"] == "frame_end"
    assert seen == [{"rgb", "depth"}]  # filled before consumers of the sensor render hook
    for data_type in ("rgb", "depth"):
        buffer = remote.buffers[data_type]
        assert (
            torch.equal(buffer, images[f"cam/{data_type}"])
            and buffer.data_ptr() != images[f"cam/{data_type}"].data_ptr()
        )
    assert not manager.get("local").buffers and not manager.get("elsewhere").buffers

    source.close()
    assert calls[-2:] == ["unmap", ("CLOSE", {})]
    hooks.emit(Phase.FRAME_END)
    assert calls[-1] == ("CLOSE", {})  # no more frames after close


@pytest.mark.parametrize(("failure", "closes"), [(RuntimeError("receiver failed"), False), (KeyboardInterrupt(), True)])
def test_only_an_interrupted_exchange_still_closes_with_the_receiver(
    plugin: PluginFixture, failure: BaseException, closes: bool
) -> None:
    source, _manager, _images, calls = plugin
    exchange = cast("Any", source.publisher.control).exchange

    def fail(command: str, **fields: Any) -> Any:
        if command != "CLOSE":
            calls.append((command, fields))
            raise failure
        return exchange(command, **fields)

    cast("Any", source.publisher.control).exchange = fail
    with pytest.raises(type(failure)):
        source.simulator.hooks.emit(Phase.FRAME_END)
    source.close()
    assert (calls[-1] == ("CLOSE", {})) is closes and "unmap" in calls
