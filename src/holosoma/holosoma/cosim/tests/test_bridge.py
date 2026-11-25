"""Backend-neutral peer bridge and metadata control without GPU simulator SDKs."""

from __future__ import annotations

import os
import subprocess
import sys
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
from holosoma.cosim.base import PeerHandler
from holosoma.cosim.cuda import BufferSpec, SharedStateAllocator
from holosoma.cosim.render.plugin import CosimRenderPlugin
from holosoma.cosim.render.renderer import RenderHandler, RenderTask
from holosoma.hooks import HookRegistry, Phase
from holosoma.simulator.shared.sensor_manager import SensorManager

pytestmark = pytest.mark.no_sim


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


PeerFixture = tuple[bridge.PeerBridge, list[Any], list[dict[str, Any]], list[dict[str, Any]]]


@pytest.fixture
def peer(monkeypatch: pytest.MonkeyPatch) -> PeerFixture:
    calls: list[Any] = []
    replies: list[dict[str, Any]] = []
    values = {"robot_root_states": torch.ones(1, 13), "dof_pos": torch.zeros(1, 2), "dof_vel": torch.zeros(1, 2)}
    sensors = SensorManager("cpu", 50.0)
    messages: list[dict[str, Any]] = [
        {
            "type": "OPEN_REQUEST",
            "fields": [],
            "handles": {},
            "writes": [],
            "ready_handle": "00",
            "gpu_uuid": "gpu-a",
            "head_pid": os.getpid(),
            "handshake": {"dof_names": ["a", "b"], "object_names": [], "env_origins": [[0.0, 0.0, 0.0]]},
        },
        {"type": "EXCHANGE_REQUEST", "snapshot": {"sequence": 1, "sim_time": 0.035}},
        {"type": "CLOSE_REQUEST"},
    ]
    socket = SimpleNamespace(send_json=replies.append, poll=lambda _ms: True)
    control = SimpleNamespace(
        socket=socket,
        receive=lambda: messages.pop(0),
        close=lambda: calls.append("close"),
    )
    monkeypatch.setattr(bridge, "ControlChannel", lambda **_: control)

    class Mapped:
        release_error: Exception | None = None
        names: list[str] = []

        def __init__(self, descriptions: Any, *, device: str) -> None:
            calls.append("map")
            Mapped.names = [spec.name for spec in descriptions]
            self.tensors = {spec.name: values[spec.name] for spec in descriptions if spec.name in values}

        def serve(self, tensors: Any) -> None:
            calls.append(("serve", sorted(tensors)))

        def record_exchange(self) -> None:
            calls.append("record_exchange")

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
        state=SimpleNamespace(
            names=lambda: [*values, *sensors.state.names()],
            writable_names=lambda: list(values),
            get=lambda name: values[name] if name in values else sensors.state.get(name),
            link=lambda sources: {name: values[name] for name in sources},
            unlink=lambda names: calls.append(("unlink", list(names))),
        ),
        sensor_manager=sensors,
        match_env_origins=lambda origins: calls.append(("place", origins.tolist())),
    )
    task = SimpleNamespace(
        simulator=sim,
        render_frame=lambda: calls.append("capture"),
    )
    handler = RenderHandler(cast("RenderTask", task))
    return bridge.PeerBridge(handler, endpoint="ipc:///unused"), calls, replies, messages


def _shared(*fields: dict[str, Any]) -> dict[str, Any]:
    """The ``fields`` and ``handles`` of an OPEN_REQUEST sharing ``fields``."""
    return {"fields": list(fields), "handles": {field["name"]: "00" for field in fields}}


def _in_order(calls: list[Any], *expected: Any) -> bool:
    """Whether ``expected`` occurs in ``calls`` in this order, possibly with other calls between."""
    remaining = iter(calls)
    return all(item in remaining for item in expected)


def test_peer_snapshot_lifecycle(peer: PeerFixture) -> None:
    worker, calls, replies, _messages = peer
    worker.run()
    place = ("place", [[0.0, 0.0, 0.0]])
    assert _in_order(calls, place, "map", ("serve", []), "ready", "record_exchange", "capture")
    assert cast("Any", worker.handler).simulator.kinematic_time == 0.035
    assert _in_order(calls, "capture", "readonly", "done", "readonly", ("unlink", []), "unmap", "close")
    assert replies == [
        {"type": "OPEN_REPLY", "done_handle": "01", "exports": [], "reads": []},
        {"type": "EXCHANGE_REPLY", "sequence": 1},
        {"type": "CLOSE_REPLY"},
    ]


@pytest.mark.parametrize("mismatch", ["dof_names", "object_names", "env_origins", "gpu_uuid", "state_layout"])
def test_layout_mismatch_fails_before_mapping(peer: PeerFixture, mismatch: str) -> None:
    worker, calls, replies, messages = peer
    if mismatch == "state_layout":
        messages[0].update(_shared({"name": "dof_pos", "shape": [1, 3], "dtype": "float32"}))
    else:
        wrong: list[Any] = [[1.0, 0.0, 0.0]] if mismatch == "env_origins" else ["wrong"]
        (messages[0] if mismatch == "gpu_uuid" else messages[0]["handshake"])[mismatch] = wrong
    with pytest.raises(ValueError, match="differ"):
        worker.run()
    assert "map" not in calls
    assert replies[-1]["type"] == "ERROR" and calls[-1] == "close"


@pytest.mark.parametrize(
    "change",
    [{"name": "camera/missing/rgb"}, {"name": "camera/cam/depth"}, {"shape": (1, 3, 8, 3)}, {"dtype": "float32"}],
)
def test_remote_camera_mismatch_fails_before_mapping(peer: PeerFixture, change: dict[str, Any]) -> None:
    worker, calls, replies, messages = peer
    record = _camera(cast("Any", worker.handler).simulator.sensor_manager, "cam", remote=None, data_types=["rgb"])
    field = {"name": record.field_name("rgb"), "shape": list(record.buffers["rgb"].shape), "dtype": "uint8", **change}
    messages[0].update(fields=[bridge.as_received(field)], writes=[field["name"]])
    with pytest.raises(ValueError, match="differs|has no"):
        worker.run()
    assert "map" not in calls
    assert replies[-1]["type"] == "ERROR" and calls[-1] == "close"


class _ReadingHandler(PeerHandler):
    """A peer that only reads ``fields``, with no simulator."""

    def __init__(self, fields: tuple[str, ...]) -> None:
        super().__init__("cuda:0")
        self.fields = fields
        self.seen: list[dict[str, Any]] = []

    def reads(self, fields: Any) -> tuple[str, ...]:
        return self.fields

    def step(self, snapshot: Any, state: Any) -> None:
        self.seen.append(dict(state))


@pytest.mark.parametrize("reader", [False, True])
def test_peer_maps_only_the_fields_its_handler_reads(peer: PeerFixture, reader: bool) -> None:
    # The render handler reads, and links, what it can write; another handler may read derived fields.
    worker, calls, replies, messages = peer
    messages[0].update(
        _shared(
            {"name": "dof_pos", "shape": [1, 2], "dtype": "float32"},
            {"name": "contact_forces", "shape": [1, 3], "dtype": "float32"},
        )
    )
    handler = _ReadingHandler(("contact_forces",)) if reader else worker.handler
    bridge.PeerBridge(cast("Any", handler), endpoint="ipc:///unused").run()
    expected = ["contact_forces"] if reader else ["dof_pos"]
    assert replies[0]["reads"] == expected
    assert bridge.MappedBuffers.names == expected  # type: ignore[attr-defined]
    assert ("serve", [] if reader else ["dof_pos"]) in calls
    if reader:
        assert len(cast("Any", handler).seen) == 1


def test_reading_a_field_the_head_does_not_share_fails_before_mapping(peer: PeerFixture) -> None:
    worker, calls, replies, _messages = peer
    handler = _ReadingHandler(("contact_forces",))
    with pytest.raises(ValueError, match="does not share"):
        bridge.PeerBridge(cast("Any", handler), endpoint="ipc:///unused").run()
    assert "map" not in calls and replies[-1]["type"] == "ERROR"


def test_duplicate_snapshot_is_rejected_and_import_is_released(
    peer: PeerFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker, calls, replies, messages = peer
    messages.insert(2, dict(messages[1]))
    monkeypatch.setattr(bridge.MappedBuffers, "release_error", RuntimeError("cleanup failed"))
    with pytest.raises(ValueError, match="consecutive"):
        worker.run()
    assert calls.count("capture") == 1
    assert replies[-1]["type"] == "ERROR" and calls[-3:] == [("unlink", []), "unmap", "close"]


def test_peer_stops_waiting_when_the_head_exits(peer: PeerFixture) -> None:
    worker, calls, _replies, messages = peer
    head = subprocess.Popen([sys.executable, "-c", "pass"])
    head.wait()
    messages[0]["head_pid"] = head.pid
    cast("Any", worker.control).socket.poll = lambda _ms: False
    with pytest.raises(RuntimeError, match="exited without closing"):
        worker.run()
    assert calls[-3:] == [("unlink", []), "unmap", "close"]


def test_exchange_after_an_interrupted_request_takes_the_right_reply(tmp_path: Path) -> None:
    import threading

    import zmq

    endpoint = f"ipc://{tmp_path}/control.sock"
    peer = bridge.ControlChannel(endpoint=endpoint, peer=True, timeout_ms=3000)
    control = bridge.ControlChannel(endpoint=endpoint, timeout_ms=100)

    def serve() -> None:
        for _ in range(2):  # Ends by timeout if the second request never arrives.
            message = peer.receive()
            if message["type"] == "EXCHANGE_REQUEST":
                threading.Event().wait(0.3)
            peer.socket.send_json({"type": message["type"].removesuffix("_REQUEST") + "_REPLY"})

    server = threading.Thread(target=serve, daemon=True)
    server.start()
    try:
        with pytest.raises(zmq.Again):
            control.exchange(bridge.MessageType.EXCHANGE_REQUEST)
        control.set_timeout(2000)
        assert control.exchange(bridge.MessageType.CLOSE_REQUEST) == {"type": "CLOSE_REPLY"}
    finally:
        server.join(timeout=5)
        control.close()
        peer.close()


class _Owner:
    """A stand-in for an imported Warp array."""

    def __init__(self, shape: tuple[int, ...], dtype: str) -> None:
        self.tensor = torch.zeros(shape, dtype=getattr(torch, dtype))


def test_mapped_buffers_serve_linked_tensors_and_tell_owner_writes_from_local_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allocations: list[_Owner] = []

    def import_buffer(_handle: bytes, *, shape: tuple[int, ...], dtype: str, device: str) -> _Owner:
        allocations.append(_Owner(shape, dtype))
        return allocations[-1]

    warp = SimpleNamespace(from_ipc_handle=import_buffer, to_torch=lambda owner: owner.tensor, float32="float32")
    monkeypatch.setitem(sys.modules, "warp", warp)
    mapped = cosim_cuda.MappedBuffers([BufferSpec("x", (2, 3), "00")], device="cpu")
    linked = torch.ones(2, 3)
    linked.set_(mapped.tensors["x"])
    mapped.serve({"x": linked})
    assert mapped.tensors["x"] is linked
    with pytest.raises(ValueError, match="does not use the import's memory"):
        mapped.serve({"x": torch.zeros(2, 3)})

    version = linked._version
    mapped.record_exchange()  # the owner may have written it
    mapped.verify_read_only()
    assert linked._version == version + 1

    linked.add_(1)  # a local write
    with pytest.raises(RuntimeError, match=r"\['x'\] were written in place"):
        mapped.record_exchange()


def test_head_bridge_without_shared_state_fails_before_connecting(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {"dof_pos": torch.zeros(1, 2)}
    state = SimpleNamespace(names=lambda: list(values), get=values.__getitem__)
    monkeypatch.setitem(sys.modules, "warp", SimpleNamespace())
    monkeypatch.setattr(bridge, "ControlChannel", lambda **_: pytest.fail("connected before validation"))
    with pytest.raises(ValueError, match="shares no state"):
        bridge.HeadBridge(
            cast("Any", SimpleNamespace(state=state)), SharedStateAllocator(("dof_pos",)), endpoint="ipc:///unused"
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
    images = {
        "camera/cam/rgb": torch.ones(2, 3, 4, 3, dtype=torch.uint8),
        "camera/cam/depth": torch.ones(2, 3, 4, 1),
    }
    calls: list[Any] = []
    replies: dict[str, dict[str, Any]] = {
        "OPEN_REQUEST": {
            "type": "OPEN_REPLY",
            "done_handle": "00",
            "exports": [
                {"name": name, "shape": list(image.shape), "dtype": str(image.dtype)[6:], "handle": "00"}
                for name, image in images.items()
            ],
            "reads": ["dof_pos"],
        },
        "CLOSE_REQUEST": {"type": "CLOSE_REPLY"},
    }

    def exchange(command: str, **fields: Any) -> dict[str, Any]:
        calls.append((command, fields))
        if command == "EXCHANGE_REQUEST":
            return {"type": "EXCHANGE_REPLY", "sequence": fields["snapshot"]["sequence"]}
        return replies[command]

    control = SimpleNamespace(exchange=exchange, set_timeout=lambda _: None, close=lambda: None)
    monkeypatch.setattr(bridge, "ControlChannel", lambda **_: control)
    mapped = SimpleNamespace(
        tensors=images,
        serve=lambda tensors: calls.append(("serve", sorted(tensors))),
        verify_read_only=lambda: calls.append("readonly"),
        record_exchange=lambda: calls.append("record_exchange"),
        release=lambda: calls.append("unmap"),
    )
    monkeypatch.setattr(bridge, "MappedBuffers", lambda *_args, **_kw: mapped)
    monkeypatch.setattr(bridge, "gpu_uuid", lambda _: "gpu-a")
    _fake_cuda(monkeypatch, SimpleNamespace(wait_event=lambda _: None, record_event=lambda _: None))
    values = {"dof_pos": torch.zeros(2, 1), "dof_vel": torch.zeros(2, 1)}
    sim = _Simulator(
        sim_device="cuda:0",
        state=SimpleNamespace(
            names=lambda: [*values, *manager.state.names()],
            get=lambda name: values[name] if name in values else manager.state.get(name),
            set_buffer_allocator=lambda _: None,
            writable_names=list,
            producing=manager.state.producing,
            link=manager.state.link,
            unlink=manager.state.unlink,
        ),
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
        "SharedStateAllocator",
        lambda fields: SimpleNamespace(allocations=dict.fromkeys(fields), specs=lambda _: []),
    )
    fake: Any = sim
    fake.sensor_capture_hook = fake.hooks.add(Phase.FRAME_END, manager.collect_due, name="sensors.capture")
    fake.sensor_render_hook = fake.hooks.add(
        Phase.FRAME_END, lambda: None, name="sensors.render", after=fake.sensor_capture_hook
    )
    with fake.hooks.installing("cosim"):
        head = CosimRenderPlugin(
            CosimRenderPluginConfig(endpoint="ipc:///unused"),
            cast("Any", SimpleNamespace(simulator=sim, hooks=fake.hooks)),
        )
    return head, manager, images, calls


def test_remote_due_cameras_read_the_peer_images_in_place(plugin: PluginFixture) -> None:
    head, manager, images, calls = plugin
    assert calls[0][1]["writes"] == ["camera/cam/rgb", "camera/cam/depth"]
    hooks = head.simulator.hooks
    remote = manager.get("cam")
    seen: list[set[str]] = []
    hooks.add(Phase.FRAME_END, lambda: seen.append(set(remote.written)), after=head.simulator.sensor_render_hook)
    hooks.emit(Phase.FRAME_END)
    assert calls[1] == ("serve", ["camera/cam/depth", "camera/cam/rgb"])  # linked at the handshake
    assert calls[2] == "readonly" and calls[4] == "record_exchange"
    assert calls[3][0] == "EXCHANGE_REQUEST" and calls[3][1]["snapshot"]["phase"] == "frame_end"
    assert seen == [{"rgb", "depth"}]  # filled before consumers of the sensor render hook
    for data_type in ("rgb", "depth"):
        buffer = remote.buffers[data_type]
        assert (
            torch.equal(buffer, images[f"camera/cam/{data_type}"])
            and buffer.data_ptr() == images[f"camera/cam/{data_type}"].data_ptr()
        )
    assert not manager.get("local").written and not manager.get("elsewhere").written

    head.close()
    assert calls[-2:] == ["unmap", ("CLOSE_REQUEST", {})]
    hooks.emit(Phase.FRAME_END)
    assert calls[-1] == ("CLOSE_REQUEST", {})  # no more frames after close


@pytest.mark.parametrize(("failure", "closes"), [(RuntimeError("peer failed"), False), (KeyboardInterrupt(), True)])
def test_only_an_interrupted_exchange_still_closes_with_the_peer(
    plugin: PluginFixture, failure: BaseException, closes: bool
) -> None:
    head, _manager, _images, calls = plugin
    exchange = cast("Any", head.bridge.control).exchange

    def fail(command: str, **fields: Any) -> Any:
        if command != "CLOSE_REQUEST":
            calls.append((command, fields))
            raise failure
        return exchange(command, **fields)

    cast("Any", head.bridge.control).exchange = fail
    with pytest.raises(type(failure)):
        head.simulator.hooks.emit(Phase.FRAME_END)
    head.close()
    assert (calls[-1] == ("CLOSE_REQUEST", {})) is closes and "unmap" in calls
