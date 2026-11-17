"""Lockstep CUDA-IPC state bridge between a source and a receiver simulator."""

from __future__ import annotations

import contextlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from holosoma.cosim.cuda import BufferSpec, CacheAllocator, MappedBuffers, share

if TYPE_CHECKING:
    import torch

    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator

# Bounds each source exchange after the first frame; startup and the first frame use the startup timeout.
FRAME_TIMEOUT_MS = 30000
# How often an idle receiver checks that its source is still running.
SOURCE_POLL_MS = 1000


@dataclass(frozen=True)
class Frame:
    sequence: int
    sim_time: float
    phase: str | None = None
    """The source's hook phase the frame was published from, or ``None`` unless exactly one phase is emitting."""

    def __post_init__(self) -> None:
        if self.sequence < 1 or not math.isfinite(self.sim_time) or self.sim_time < 0:
            raise ValueError("Frame sequence and simulation time must be valid")


class ControlChannel:
    """Metadata-only REQ/REP exchange; callers supply their private endpoint."""

    def __init__(self, *, endpoint: str, receiver: bool = False, startup_timeout_ms: int = 300000) -> None:
        import zmq

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP if receiver else zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.set_timeout(startup_timeout_ms)
        try:
            (self.socket.bind if receiver else self.socket.connect)(endpoint)
        except BaseException:
            self.close()
            raise

    def set_timeout(self, timeout_ms: int) -> None:
        import zmq

        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)

    def exchange(self, command: str, **fields: Any) -> dict[str, Any]:
        self.socket.send_json(dict(type=command, **fields))
        reply = self.receive()
        if reply["type"] == "ERROR":
            raise RuntimeError(reply["message"])
        return reply

    def receive(self) -> dict[str, Any]:
        message = self.socket.recv_json()
        if not isinstance(message, dict):
            raise TypeError(f"Bridge messages must be JSON objects, got {type(message).__name__}")
        return message

    def close(self) -> None:
        self.socket.close()
        self.context.term()


def gpu_uuid(device: Any) -> str:
    """The physical GPU behind ``device``, independent of ``CUDA_VISIBLE_DEVICES`` numbering."""
    import torch

    return str(torch.cuda.get_device_properties(device).uuid)


def process_alive(pid: int) -> bool:
    """Whether process ``pid`` exists in this PID namespace."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def as_received(value: Any) -> Any:
    """``value`` as it arrives after a JSON round trip, so local and received values compare equal."""
    return json.loads(json.dumps(value))


@dataclass(frozen=True)
class BufferRequest:
    """A buffer one side reads from its peer: its layout and the settings both sides must share."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_message(cls, raw: Mapping[str, Any]) -> BufferRequest:
        return cls(raw["name"], tuple(raw["shape"]), raw["dtype"], dict(raw["meta"]))


def scene_layout(sim: BaseSimulator) -> dict[str, Any]:
    """The joint, object and environment layout both sides' state must share."""
    layout: dict[str, Any] = as_received(
        {
            "dof_names": list(sim.dof_names),
            "object_names": list(sim.object_state_names),
            "env_origins": sim.scene.env_origins.cpu().tolist(),
        }
    )
    return layout


class Publisher:
    """Share a source's state caches with a receiver and exchange frames with it in lockstep.

    The handshake runs at construction: every state field ``allocator`` shares, and ``requests``, go to
    the receiver. It answers with the fields it reads and one export per request, mapped as
    :attr:`imports`. Callers decide when to :meth:`publish`; each exchange refreshes the fields the
    receiver reads and orders later work after the receiver's. After a failed exchange the publisher
    refuses further exchanges and closes without contacting the receiver.
    """

    def __init__(
        self,
        sim: BaseSimulator,
        allocator: CacheAllocator,
        *,
        endpoint: str,
        requests: Sequence[BufferRequest] = (),
    ) -> None:
        import warp as wp

        self.sim, self.allocator = sim, allocator
        state = sim.state
        for name in sorted(allocator.fields):
            state.get(name)  # Allocate through the allocator.
        exported = tuple(name for name in state.names() if name in allocator.owners)
        if not exported:
            raise ValueError("The allocator shares no state; install it before simulator state is first read")
        self.fields = exported
        self.control = ControlChannel(endpoint=endpoint)
        self.ready = wp.Event(device=sim.sim_device, interprocess=True)
        self.sequence = 0
        self.closed = self.failed = False
        try:
            reply = self.control.exchange(
                "OPEN",
                exports=[asdict(spec) for spec in allocator.exports(self.values())],
                requests=[asdict(request) for request in requests],
                ready_handle=self.ready.ipc_handle().hex(),
                gpu_uuid=gpu_uuid(sim.sim_device),
                source_pid=os.getpid(),
                scene=scene_layout(sim),
            )
            if reply["type"] != "OPENED":
                raise RuntimeError("Receiver did not accept state buffers")
            self.done = wp.event_from_ipc_handle(bytes.fromhex(reply["done_handle"]), device=sim.sim_device)
            if unknown := sorted(set(reply["reads"]) - set(exported)):
                raise RuntimeError(f"Receiver reads state fields {unknown} the source does not share")
            self.fields = tuple(reply["reads"])
            specs = [BufferSpec.from_message(raw) for raw in reply["exports"]]
            images = {spec.name: (spec.shape, spec.dtype) for spec in specs}
            requested = {request.name: (request.shape, request.dtype) for request in requests}
            if images != requested:
                raise RuntimeError(f"Receiver exported {images}, requested {requested}")
            self._imports = MappedBuffers(specs, device=sim.sim_device)
        except BaseException:
            self.control.close()
            raise

    @property
    def imports(self) -> dict[str, torch.Tensor]:
        """The receiver's exports, by request name; valid after an exchange until the next one."""
        return self._imports.tensors

    def values(self) -> dict[str, Any]:
        return {name: self.sim.state.get(name) for name in self.fields}

    def _exchange(self, command: str, **fields: Any) -> dict[str, Any]:
        """Share current state, send ``command`` and order later work on this stream after the receiver's."""
        import torch
        import warp as wp

        if self.failed:
            raise RuntimeError("An earlier co-simulation exchange failed")
        self.values()  # Refresh the same persistent caches, without replacing storage.
        stream = wp.stream_from_torch(torch.cuda.current_stream(self.sim.sim_device))
        stream.record_event(self.ready)
        try:
            reply = self.control.exchange(command, **fields)
        except BaseException:
            self.failed = True
            raise
        stream.wait_event(self.done)
        return reply

    def _phase(self) -> str | None:
        emitting = self.sim.hooks.emitting
        return next(iter(emitting)).value if len(emitting) == 1 else None

    def publish(self) -> None:
        """Have the receiver process current state as the next frame, at the source's current time."""
        self.sequence += 1
        frame = Frame(self.sequence, self.sim.time(), self._phase())
        reply = self._exchange("FRAME_AVAILABLE", frame=asdict(frame))
        if reply["type"] != "FRAME_COMPLETE" or reply["sequence"] != frame.sequence:
            raise RuntimeError("Receiver completed the wrong frame")
        if frame.sequence == 1:
            self.control.set_timeout(FRAME_TIMEOUT_MS)

    def close(self) -> None:
        import torch

        if self.closed:
            return
        self.closed = True
        try:
            # Unmap imports before the receiver may free them.
            torch.cuda.synchronize(self.sim.sim_device)
            self._imports.release()
            if not self.failed:
                reply = self.control.exchange("CLOSE")
                if reply["type"] != "UNMAPPED":
                    raise RuntimeError("Receiver did not release shared state")
        finally:
            self.control.close()


class FrameHandler(Protocol):
    """What a receiver reads from a source, what it does with each frame, and the exports it answers requests with."""

    simulator: BaseSimulator

    def reads(self, exports: Sequence[BufferSpec]) -> tuple[str, ...]:
        """Return the exported state fields to map and refresh each frame; raise if one is unusable."""

    def allocate(self, requests: list[BufferRequest]) -> dict[str, Any]:
        """Check the source's requests and return one shared array per request, by name."""

    def process(self, state: Mapping[str, torch.Tensor], sim_time: float, *, phase: str | None) -> None:
        """Process the source's ``state`` at source time ``sim_time`` and fill the exports.

        ``state`` holds the read fields, read-only and valid only during the call. ``phase`` is the
        source's hook phase the frame came from, or ``None`` unless exactly one phase is emitting.
        """


class Receiver:
    """Map the source state ``handler`` reads and have ``handler`` process each frame.

    After the handshake it waits for frames as long as the source process is running, so the source
    and receiver must share a PID namespace.
    """

    def __init__(self, handler: FrameHandler, *, endpoint: str) -> None:
        self.handler = handler
        self.sim = handler.simulator
        self.control = ControlChannel(endpoint=endpoint, receiver=True)

    def run(self) -> None:
        import torch
        import warp as wp

        mapped: MappedBuffers | None = None
        ready = done = None
        source_pid: int | None = None
        sequence = 0
        stream = wp.stream_from_torch(torch.cuda.current_stream(self.sim.sim_device))

        def receive() -> dict[str, Any]:
            if source_pid is not None:
                while not self.control.socket.poll(SOURCE_POLL_MS):
                    if not process_alive(source_pid):
                        raise RuntimeError(f"The co-simulation source (pid {source_pid}) exited without closing")
            return self.control.receive()

        try:
            while True:
                message = receive()
                try:
                    command = message["type"]
                    if command == "OPEN" and mapped is None:
                        scene = message["scene"]
                        origins = self.sim.scene.env_origins
                        self.sim.match_env_origins(
                            torch.tensor(scene["env_origins"], dtype=torch.float32, device=origins.device)
                        )
                        own_scene = scene_layout(self.sim)
                        if differing := sorted(
                            key for key in {*scene, *own_scene} if scene.get(key) != own_scene.get(key)
                        ):
                            raise ValueError(f"Scene {differing} differs between simulators")
                        if (source_gpu := message["gpu_uuid"]) != (own_gpu := gpu_uuid(self.sim.sim_device)):
                            raise ValueError(f"Source GPU {source_gpu} differs from receiver GPU {own_gpu}")
                        exported = [BufferSpec.from_message(raw) for raw in message["exports"]]
                        reads = self.handler.reads(exported)
                        if unknown := sorted(set(reads) - {spec.name for spec in exported}):
                            raise ValueError(f"Receiver reads state fields {unknown} the source does not share")
                        exports = self.handler.allocate(
                            [BufferRequest.from_message(raw) for raw in message["requests"]]
                        )
                        mapped = MappedBuffers(
                            [spec for spec in exported if spec.name in reads], device=self.sim.sim_device
                        )
                        ready = wp.event_from_ipc_handle(
                            bytes.fromhex(message["ready_handle"]),
                            device=self.sim.sim_device,
                        )
                        done = wp.Event(device=self.sim.sim_device, interprocess=True)
                        source_pid = int(message["source_pid"])
                        self.control.socket.send_json(
                            {
                                "type": "OPENED",
                                "done_handle": done.ipc_handle().hex(),
                                "exports": [asdict(share(name, owner)) for name, owner in exports.items()],
                                "reads": list(reads),
                            }
                        )
                    elif command == "FRAME_AVAILABLE" and mapped is not None:
                        frame = Frame(**message["frame"])
                        if frame.sequence != sequence + 1:
                            raise ValueError("Frames must be consecutive")
                        stream.wait_event(ready)
                        self.handler.process(mapped.tensors, frame.sim_time, phase=frame.phase)
                        mapped.verify_read_only()
                        stream.record_event(done)
                        sequence = frame.sequence
                        self.control.socket.send_json({"type": "FRAME_COMPLETE", "sequence": sequence})
                    elif command == "CLOSE" and mapped is not None:
                        torch.cuda.synchronize(self.sim.sim_device)
                        mapped.release()
                        mapped = None
                        self.control.socket.send_json({"type": "UNMAPPED"})
                        return
                    else:
                        raise ValueError(f"Unexpected bridge command: {command}")
                except Exception as error:
                    self.control.socket.send_json({"type": "ERROR", "message": str(error)})
                    raise
        except BaseException:
            if mapped is not None:
                with contextlib.suppress(Exception):  # Keep the original failure.
                    torch.cuda.synchronize(self.sim.sim_device)
                    mapped.release()
            raise
        finally:
            self.control.close()
