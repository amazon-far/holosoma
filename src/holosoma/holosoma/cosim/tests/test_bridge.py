"""Backend-neutral receiver and metadata control without GPU simulator SDKs."""

from __future__ import annotations

import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from holosoma.config_types.plugin import CosimRenderPluginConfig
from holosoma.config_types.sensor import CameraSensorConfig, SensorMountConfig
from holosoma.cosim import bridge
from holosoma.cosim import cuda as cosim_cuda
from holosoma.cosim import render as cosim_render
from holosoma.cosim.cameras import camera_requests
from holosoma.cosim.cuda import BufferSpec, CacheAllocator, MappedBuffers, share
from holosoma.cosim.render import CosimRenderPlugin, RenderHandler
from holosoma.envs.render_task import RenderTask
from holosoma.simulator.base_simulator.hooks import Phase
from holosoma.simulator.shared.sensor_manager import SensorManager

pytestmark = pytest.mark.no_sim


@pytest.mark.parametrize("frame", [(0, 0), (1, float("nan")), (1, -1)])
def test_invalid_frame_is_rejected(frame: tuple[int, float]) -> None:
    with pytest.raises(ValueError, match="must be valid"):
        bridge.Frame(*frame)


def _camera(manager: SensorManager, name: str, *, remote: bool, data_types: list[Any]) -> Any:
    mount = SensorMountConfig(target_kind="world", target="")
    config = CameraSensorConfig(mount=mount, width=4, height=3, data_types=data_types, remote=remote)
    return manager.register_camera(name, config)


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
            "scene": {"dof_names": ["a", "b"], "object_names": [], "env_origins": [[0.0, 0.0, 0.0]]},
        },
        {"type": "FRAME_AVAILABLE", "frame": {"sequence": 1, "sim_time": 0.035}},
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
        names: list[str] = []

        def __init__(self, descriptions: Any, *, device: str) -> None:
            calls.append("map")
            Mapped.names = [description["name"] for description in descriptions]
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
            to_torch=lambda owner: owner,
        ),
    )
    monkeypatch.setattr(
        cosim_render, "allocate_shared", lambda shape, dtype, _: torch.zeros(shape, dtype=getattr(torch, dtype))
    )
    monkeypatch.setattr(
        bridge, "share", lambda name, owner: BufferSpec(name, tuple(owner.shape), "00", str(owner.dtype)[6:])
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _: object())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _: calls.append("sync"))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(uuid="gpu-a"))
    sim = SimpleNamespace(
        sim_device="cuda:0",
        dof_names=["a", "b"],
        object_state_names=(),
        scene=SimpleNamespace(env_origins=torch.zeros(1, 3)),
        state=SimpleNamespace(writable_names=lambda: list(values), get=values.__getitem__),
        apply_state=lambda inputs: calls.append(("apply", inputs is values)),
        sensor_manager=SensorManager("cpu", 50.0),
        match_env_origins=lambda origins: calls.append(("place", origins.tolist())),
    )
    task = SimpleNamespace(
        simulator=sim,
        num_envs=1,
        render_frame=lambda **kw: calls.append(("capture", kw["sim_time"], *(["repeat"] if kw.get("repeat") else []))),
    )
    handler = RenderHandler(cast("RenderTask", task))
    return bridge.Receiver(handler, endpoint="ipc:///unused"), calls, replies, messages


def test_receiver_uses_only_native_state_and_capture_contracts(receiver: ReceiverFixture) -> None:
    worker, calls, replies, _messages = receiver
    worker.run()
    assert calls == [
        ("place", [[0.0, 0.0, 0.0]]),
        "map",
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
        {"type": "OPENED", "done_handle": "01", "exports": [], "reads": []},
        {"type": "FRAME_COMPLETE", "sequence": 1},
        {"type": "UNMAPPED"},
    ]


@pytest.mark.parametrize("mismatch", ["dof_names", "object_names", "env_origins", "gpu_uuid"])
def test_layout_mismatch_fails_before_mapping(receiver: ReceiverFixture, mismatch: str) -> None:
    worker, calls, replies, messages = receiver
    wrong: list[Any] = [[1.0, 0.0, 0.0]] if mismatch == "env_origins" else ["wrong"]
    (messages[0] if mismatch == "gpu_uuid" else messages[0]["scene"])[mismatch] = wrong
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
    record = _camera(worker.sim.sensor_manager, "cam", remote=False, data_types=["rgb"])
    (request,) = camera_requests(record, 1)
    if "meta" in change:
        change = {"meta": {**request.meta, **change["meta"]}}
    messages[0]["requests"] = [bridge.as_received(asdict(replace(request, **change)))]
    with pytest.raises(ValueError, match="differs|no camera output"):
        worker.run()
    assert "map" not in calls
    assert replies[-1]["type"] == "ERROR" and calls[-1] == "close"


def test_receiver_exports_its_images_after_each_render(
    receiver: ReceiverFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker, calls, replies, messages = receiver
    record = _camera(worker.sim.sensor_manager, "cam", remote=False, data_types=["rgb", "depth"])
    messages[0]["requests"] = [bridge.as_received(asdict(request)) for request in camera_requests(record, 1)]
    messages.insert(2, {"type": "REPEAT", "sim_time": 0.035, "phase": None})
    exports: list[torch.Tensor] = []
    exported_before_render: list[list[float]] = []
    allocate, render = cosim_render.allocate_shared, cast("Any", worker.handler).task.render_frame

    def allocate_shared(*args: Any) -> torch.Tensor:
        exports.append(allocate(*args))
        return exports[-1]

    def render_frame(**kw: Any) -> None:
        exported_before_render.append([float(export.max()) for export in exports])
        render(**kw)
        frame = len(exported_before_render)
        record.set_buffer("rgb", torch.full((1, 3, 4, 3), frame, dtype=torch.uint8))
        record.set_buffer("depth", torch.full((1, 3, 4, 1), float(frame)))

    monkeypatch.setattr(cosim_render, "allocate_shared", allocate_shared)
    monkeypatch.setattr(cast("Any", worker.handler).task, "render_frame", render_frame)
    worker.run()
    assert [(spec["name"], spec["dtype"]) for spec in replies[0]["exports"]] == [
        ("cam/rgb", "uint8"),
        ("cam/depth", "float32"),
    ]
    assert ("capture", 0.035, "repeat") in calls and replies[2] == {"type": "REPEATED"}
    assert exported_before_render == [[0.0, 0.0], [1.0, 1.0]]
    assert all(bool((export == 2).all()) for export in exports)


def test_receiver_reads_only_the_state_it_can_apply(receiver: ReceiverFixture) -> None:
    worker, _calls, replies, messages = receiver
    messages[0]["exports"] = [
        {"name": "dof_pos", "shape": [1, 2], "handle": "00"},
        {"name": "contact_forces", "shape": [1, 3], "handle": "00"},
    ]
    worker.run()
    assert replies[0]["reads"] == ["dof_pos"]
    assert bridge.MappedBuffers.names == ["dof_pos"]  # type: ignore[attr-defined]


class _ReadingHandler:
    """A handler that only reads ``fields``, without applying them anywhere."""

    def __init__(self, simulator: Any, fields: tuple[str, ...]) -> None:
        self.simulator, self.fields = simulator, fields
        self.seen: list[dict[str, Any]] = []

    def reads(self, exports: Any) -> tuple[str, ...]:
        return self.fields

    def allocate(self, requests: Any) -> dict[str, Any]:
        return {}

    def process(self, state: Any, sim_time: float, *, repeat: bool, phase: str | None) -> None:
        self.seen.append(dict(state))


def test_handler_reads_derived_state_without_applying_it(receiver: ReceiverFixture) -> None:
    worker, calls, replies, messages = receiver
    handler = _ReadingHandler(worker.sim, ("contact_forces",))
    messages[0]["exports"] = [
        {"name": "dof_pos", "shape": [1, 2], "handle": "00"},
        {"name": "contact_forces", "shape": [1, 3], "handle": "00"},
    ]
    bridge.Receiver(cast("Any", handler), endpoint="ipc:///unused").run()
    assert replies[0]["reads"] == ["contact_forces"]
    assert bridge.MappedBuffers.names == ["contact_forces"]  # type: ignore[attr-defined]
    assert len(handler.seen) == 1 and not any(isinstance(call, tuple) and call[0] == "apply" for call in calls)


def test_reading_a_field_the_source_does_not_share_fails_before_mapping(receiver: ReceiverFixture) -> None:
    worker, calls, replies, _messages = receiver
    handler = _ReadingHandler(worker.sim, ("contact_forces",))
    with pytest.raises(ValueError, match="does not share"):
        bridge.Receiver(cast("Any", handler), endpoint="ipc:///unused").run()
    assert "map" not in calls and replies[-1]["type"] == "ERROR"


def test_read_state_field_layout_mismatch_fails_before_mapping(receiver: ReceiverFixture) -> None:
    worker, calls, replies, messages = receiver
    messages[0]["exports"] = [{"name": "dof_pos", "shape": [1, 3], "handle": "00"}]
    with pytest.raises(ValueError, match="differs between simulators"):
        worker.run()
    assert not any(isinstance(call, tuple) and call[0] != "place" for call in calls)
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
    _camera(manager, "local", remote=False, data_types=["rgb"])
    _camera(manager, "cam", remote=True, data_types=["rgb", "depth"])
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
        "FRAME_AVAILABLE": {"type": "FRAME_COMPLETE", "sequence": 1},
        "REPEAT": {"type": "REPEATED"},
        "CLOSE": {"type": "UNMAPPED"},
    }

    def exchange(command: str, **fields: Any) -> dict[str, Any]:
        calls.append((command, fields))
        return replies[command]

    control = SimpleNamespace(exchange=exchange, set_timeout=lambda _: None, close=lambda: None)
    monkeypatch.setattr(bridge, "ControlChannel", lambda **_: control)
    mapped = SimpleNamespace(tensors=images, release=lambda: calls.append("unmap"))
    monkeypatch.setattr(bridge, "MappedBuffers", lambda *_args, **_kw: mapped)
    monkeypatch.setattr(bridge, "gpu_uuid", lambda _: "gpu-a")
    stream = SimpleNamespace(wait_event=lambda _: None, record_event=lambda _: None)
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
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _: None)
    values = {"dof_pos": torch.zeros(2, 1), "dof_vel": torch.zeros(2, 1)}
    sim = _Simulator(
        sim_device="cuda:0",
        state=SimpleNamespace(names=lambda: list(values), get=values.__getitem__, set_buffer_allocator=lambda _: None),
        sensor_manager=manager,
        hooks=SimpleNamespace(add=lambda *_args, **_kw: SimpleNamespace(remove=lambda: None), emitting=frozenset()),
        sensor_render_hook=None,
        dof_names=["a"],
        object_state_names=(),
        scene=SimpleNamespace(env_origins=torch.zeros(2, 3)),
        time=lambda: 0.5,
    )
    monkeypatch.setattr(
        cosim_cuda,
        "CacheAllocator",
        lambda fields: SimpleNamespace(fields=frozenset(fields), owners=dict.fromkeys(fields), exports=lambda _: []),
    )
    source = CosimRenderPlugin(CosimRenderPluginConfig(endpoint="ipc:///unused"), cast("Any", sim))
    return source, manager, images, calls


def test_remote_due_cameras_get_copies_of_the_receiver_images(plugin: PluginFixture) -> None:
    source, manager, images, calls = plugin
    assert source.publisher.fields == ("dof_pos",)
    assert [(request["name"], request["shape"], request["dtype"]) for request in calls[0][1]["requests"]] == [
        ("cam/rgb", (2, 3, 4, 3), "uint8"),
        ("cam/depth", (2, 3, 4, 1), "float32"),
    ]
    cast("Any", source.simulator).hooks.emitting = frozenset({Phase.FRAME_END})
    manager.render_remote(manager.collect_due())
    assert calls[1][1]["frame"]["phase"] == "frame_end"
    remote = manager.get("cam")
    for data_type in ("rgb", "depth"):
        buffer = remote.buffers[data_type]
        assert (
            torch.equal(buffer, images[f"cam/{data_type}"])
            and buffer.data_ptr() != images[f"cam/{data_type}"].data_ptr()
        )
    assert not manager.get("local").buffers and calls[1][0] == "FRAME_AVAILABLE"

    source.close()
    assert calls[-2:] == ["unmap", ("CLOSE", {})]
    with pytest.raises(RuntimeError, match="no remote renderer"):
        manager.render_remote(manager.collect_due())


def test_rerender_writes_only_the_reset_rows(plugin: PluginFixture) -> None:
    _source, manager, images, calls = plugin
    manager.render_remote(manager.collect_due())
    for image in images.values():
        image.fill_(7)
    manager.rerender(torch.tensor([1]))
    assert calls[-1] == ("REPEAT", {"sim_time": 0.5, "phase": None})
    for data_type in ("rgb", "depth"):
        buffer = manager.get("cam").buffers[data_type]
        assert bool((buffer[0] == 1).all()) and bool((buffer[1] == 7).all())


@pytest.mark.parametrize("dtype", ["uint8", "float32"])
def test_shared_buffer_dtype_survives_export_and_import(monkeypatch: pytest.MonkeyPatch, dtype: str) -> None:
    tensor = torch.zeros(2, 3, dtype=getattr(torch, dtype))

    class Owner:  # Weak-referenceable, like a Warp array.
        shape = (2, 3)

        def __init__(self) -> None:
            self.dtype = SimpleNamespace(__name__=dtype)

        def ipc_handle(self) -> bytes:
            return b"\x01"

    owner = Owner()
    imported: dict[str, Any] = {}

    def from_ipc_handle(handle: bytes, **kw: Any) -> Any:
        imported.update(kw, handle=handle)
        return owner

    warp = SimpleNamespace(
        uint8="wp.uint8", float32="wp.float32", from_ipc_handle=from_ipc_handle, to_torch=lambda _: tensor
    )
    monkeypatch.setitem(sys.modules, "warp", warp)
    mapped = MappedBuffers([asdict(share("cam/rgb", owner))], device="cuda:0")
    assert imported == {"handle": b"\x01", "shape": (2, 3), "dtype": f"wp.{dtype}", "device": "cuda:0"}
    assert mapped.tensors["cam/rgb"] is tensor
