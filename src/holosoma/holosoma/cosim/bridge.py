"""Lockstep CUDA-IPC exchange between a head and a peer process on the same GPU.

:class:`HeadBridge` and :class:`PeerBridge` exchange metadata over a ZeroMQ REQ/REP socket; state and
exports stay in shared CUDA allocations, ordered by interprocess ready and done events.

- ``OPEN_REQUEST`` -> ``OPEN_REPLY``: the head sends its SimState ``fields`` (name, shape, dtype), the
  ``handles`` of those it shares, the names of the fields the peer is to ``write``, ``meta``, ready event, GPU
  UUID and PID. The peer's handler accepts ``meta``, picks the shared fields it reads and allocates one export
  per written field; the reply carries the reads, the exports and its done event.
- ``EXCHANGE_REQUEST`` -> ``EXCHANGE_REPLY``: one per :meth:`HeadBridge.publish`, with a :class:`Snapshot`
  numbered consecutively. Each hands over every shared buffer: the side receiving it bumps the versions
  of everything it maps, as if each were written locally.
- ``CLOSE_REQUEST`` -> ``CLOSE_REPLY``: both sides release their mappings.
- ``ERROR``: the peer's failure, raised on the head.

Each buffer is written only by the side that allocated it; the side mapping it checks every exchange that
it made no in-place write.

The head waits for each exchange. The handshake and the first exchange may take up to
:data:`STARTUP_TIMEOUT_MS` (first-use compilation); later head exchanges time out after
:data:`EXCHANGE_TIMEOUT_MS`. The peer waits while the head process runs, so both must share a PID
namespace. There is no reconnection; after losing a peer, keep exported allocations until it has exited
or unmapped them.
"""

from __future__ import annotations

import contextlib
import enum
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Protocol

from holosoma.cosim.cuda import BufferSpec, FieldSpec, MappedBuffers, SharedStateAllocator, share

if TYPE_CHECKING:
    import torch

    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator

STARTUP_TIMEOUT_MS = 300000
EXCHANGE_TIMEOUT_MS = 30000
# How often an idle peer checks that its head is still running.
HEAD_POLL_MS = 1000


class MessageType(str, enum.Enum):
    """The ``type`` of a bridge message: a head request, the peer's reply to it, or the peer's ``ERROR``."""

    OPEN_REQUEST = "OPEN_REQUEST"
    OPEN_REPLY = "OPEN_REPLY"
    EXCHANGE_REQUEST = "EXCHANGE_REQUEST"
    EXCHANGE_REPLY = "EXCHANGE_REPLY"
    CLOSE_REQUEST = "CLOSE_REQUEST"
    CLOSE_REPLY = "CLOSE_REPLY"
    ERROR = "ERROR"


@dataclass(frozen=True)
class Snapshot:
    """The head's state as handed over at one exchange."""

    sequence: int
    sim_time: float
    phase: str | None = None
    """The head's hook phase the snapshot was published from, or ``None`` unless exactly one phase is emitting."""

    def __post_init__(self) -> None:
        if self.sequence < 1 or not math.isfinite(self.sim_time) or self.sim_time < 0:
            raise ValueError("Snapshot sequence and simulation time must be valid")


class ControlChannel:
    """Metadata-only REQ/REP exchange; callers supply their private endpoint."""

    def __init__(self, *, endpoint: str, peer: bool = False, timeout_ms: int = STARTUP_TIMEOUT_MS) -> None:
        import zmq

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP if peer else zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.awaiting_reply = False
        self.set_timeout(timeout_ms)
        try:
            (self.socket.bind if peer else self.socket.connect)(endpoint)
        except BaseException:
            self.close()
            raise

    def set_timeout(self, timeout_ms: int) -> None:
        import zmq

        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)

    def exchange(self, command: MessageType, **fields: Any) -> dict[str, Any]:
        if self.awaiting_reply:
            # A request interrupted before its reply, e.g. by a signal, leaves REQ waiting for that reply.
            self.receive()
            self.awaiting_reply = False
        self.socket.send_json(dict(type=command, **fields))
        self.awaiting_reply = True
        reply = self.receive()
        self.awaiting_reply = False
        if reply["type"] == MessageType.ERROR:
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


class HeadBridge:
    """Share a head's state caches with a peer and exchange snapshots with it in lockstep.

    The handshake runs at construction: the head's SimState fields (shared ones with their memory), the
    names of those the peer is to ``write`` and ``meta``, which the peer interprets, go to the peer. It
    answers with the shared fields it reads and one export per written field, mapped as :attr:`imports`;
    the written fields are linked onto them (``SimState.link``). Callers decide when to :meth:`publish`; each
    exchange refreshes the fields the peer reads and orders later work after the peer's. After an exchange
    fails, other than by interruption, the bridge refuses further exchanges and closes without contacting
    the peer.
    """

    def __init__(
        self,
        simulator: BaseSimulator,
        allocator: SharedStateAllocator,
        *,
        endpoint: str,
        writes: Sequence[str] = (),
        meta: Mapping[str, Any] | None = None,
        startup_timeout_ms: int = STARTUP_TIMEOUT_MS,
        exchange_timeout_ms: int = EXCHANGE_TIMEOUT_MS,
    ) -> None:
        import torch
        import warp as wp

        self.simulator, self.allocator = simulator, allocator
        self.exchange_timeout_ms = exchange_timeout_ms
        state = simulator.state
        for name in sorted(allocator.fields):
            state.get(name)
        fields = {
            name: FieldSpec(name, tuple(value.shape), str(value.dtype).removeprefix("torch."))
            for name in state.names()
            if isinstance(value := state.get(name), torch.Tensor)
        }
        if unknown := sorted(set(writes) - set(fields)):
            raise ValueError(f"The peer cannot write {unknown}: the head has no such fields")
        shared = tuple(name for name in fields if name in allocator.allocations and name not in writes)
        if not shared:
            raise ValueError("The allocator shares no state; install it before simulator state is first read")
        self.reads = shared
        self.control = ControlChannel(endpoint=endpoint, timeout_ms=startup_timeout_ms)
        self.ready = wp.Event(device=simulator.sim_device, interprocess=True)
        self.sequence = 0
        self.closed = self.failed = False
        try:
            reply = self.control.exchange(
                MessageType.OPEN_REQUEST,
                fields=[asdict(spec) for spec in fields.values()],
                handles={spec.name: spec.handle for spec in allocator.specs(self.refresh())},
                writes=list(writes),
                ready_handle=self.ready.ipc_handle().hex(),
                gpu_uuid=gpu_uuid(simulator.sim_device),
                head_pid=os.getpid(),
                meta=as_received(dict(meta or {})),
            )
            if reply["type"] != MessageType.OPEN_REPLY:
                raise RuntimeError("Peer did not accept state buffers")
            self.done = wp.event_from_ipc_handle(bytes.fromhex(reply["done_handle"]), device=simulator.sim_device)
            if unknown := sorted(set(reply["reads"]) - set(shared)):
                raise RuntimeError(f"Peer reads state fields {unknown} the head does not share")
            self.reads = tuple(reply["reads"])
            specs = [BufferSpec.from_message(raw) for raw in reply["exports"]]
            exported = {spec.name: FieldSpec(spec.name, spec.shape, spec.dtype) for spec in specs}
            if exported != (wanted := {name: fields[name] for name in writes}):
                raise RuntimeError(f"Peer exported {exported}, the head asked it to write {wanted}")
            self._imports = MappedBuffers(specs, device=simulator.sim_device)
            # The written fields are served from the peer's memory; its writes to read-only ones are recorded as
            # their producer's, and writable ones flush into the simulation.
            self._linked = list(writes)
            writable = set(simulator.state.writable_names())
            self._produced = [name for name in writes if name not in writable]
            self._imports.serve(simulator.state.link({name: self._imports.tensors[name] for name in self._linked}))
        except BaseException:
            self.control.close()
            raise

    @property
    def imports(self) -> dict[str, torch.Tensor]:
        """The peer's exports, by written field name, read in place; valid after an exchange until the next one.

        The same tensors throughout, read-only; each version moves once per exchange. The written SimState
        fields are linked onto them, so they serve the peer's memory as well.
        """
        return self._imports.tensors

    def refresh(self) -> dict[str, Any]:
        """Bring the caches the peer reads up to date, in place, and return them."""
        return {name: self.simulator.state.get(name) for name in self.reads}

    def _exchange(self, command: MessageType, **fields: Any) -> dict[str, Any]:
        """Send ``command`` and order later work on this stream after the peer's."""
        import torch
        import warp as wp

        stream = wp.stream_from_torch(torch.cuda.current_stream(self.simulator.sim_device))
        stream.record_event(self.ready)
        try:
            reply = self.control.exchange(command, **fields)
        except Exception:
            self.failed = True
            raise
        stream.wait_event(self.done)
        return reply

    def _phase(self) -> str | None:
        emitting = self.simulator.hooks.emitting
        return next(iter(emitting)).value if len(emitting) == 1 else None

    def publish(self) -> None:
        """Have the peer process the current state as the next snapshot, at the head's current time."""
        if self.failed:
            raise RuntimeError("An earlier co-simulation exchange failed")
        self._imports.verify_read_only()
        self.refresh()
        self.sequence += 1
        snapshot = Snapshot(self.sequence, self.simulator.time(), self._phase())
        reply = self._exchange(MessageType.EXCHANGE_REQUEST, snapshot=asdict(snapshot))
        if reply["type"] != MessageType.EXCHANGE_REPLY or reply["sequence"] != snapshot.sequence:
            raise RuntimeError("Peer completed the wrong exchange")
        with self.simulator.state.producing(self._produced):
            self._imports.record_exchange()
        if snapshot.sequence == 1:
            self.control.set_timeout(self.exchange_timeout_ms)

    def close(self) -> None:
        import torch

        if self.closed:
            return
        self.closed = True
        try:
            # Unmap imports before the peer may free them.
            torch.cuda.synchronize(self.simulator.sim_device)
            try:
                self._imports.verify_read_only()
            finally:
                self.simulator.state.unlink(self._linked)
                self._imports.release()
            if not self.failed:
                reply = self.control.exchange(MessageType.CLOSE_REQUEST)
                if reply["type"] != MessageType.CLOSE_REPLY:
                    raise RuntimeError("Peer did not release shared state")
        finally:
            self.control.close()


class PeerHandlerProtocol(Protocol):
    """A peer's side of the exchange; :class:`holosoma.cosim.base.PeerHandler` implements it."""

    device: str
    """CUDA device for imported state, exports and events."""

    @property
    def allocations(self) -> Mapping[str, Any]:
        """Shared allocations of the fields the peer writes, by name, once :meth:`open` returns."""

    def open(
        self, fields: Sequence[BufferSpec], writes: Sequence[FieldSpec], meta: Mapping[str, Any]
    ) -> tuple[str, ...]:
        """Accept the head, allocate :attr:`allocations` for ``writes`` and return the shared ``fields`` to read;
        raise to refuse."""

    def link(self, imports: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        """Link tensors of the handler's own onto the mapped head fields ``imports``; they then serve as ``state``."""

    def process(self, snapshot: Snapshot, state: Mapping[str, torch.Tensor]) -> None:
        """Process the head's ``state`` for ``snapshot`` and fill the exports.

        ``state`` holds the read fields, read-only and valid only during the call.
        """

    def unlink(self) -> None:
        """Move what :meth:`link` linked off the mapping; called before the mapping is released."""


class PeerBridge:
    """Map the head state ``handler`` reads and have ``handler`` process each snapshot.

    After the handshake it waits for exchanges as long as the head process is running, so the head and
    peer must share a PID namespace.
    """

    def __init__(
        self, handler: PeerHandlerProtocol, *, endpoint: str, startup_timeout_ms: int = STARTUP_TIMEOUT_MS
    ) -> None:
        self.handler = handler
        self.control = ControlChannel(endpoint=endpoint, peer=True, timeout_ms=startup_timeout_ms)

    def run(self) -> None:
        import torch
        import warp as wp

        device = self.handler.device
        mapped: MappedBuffers | None = None
        ready = done = None
        head_pid: int | None = None
        sequence = 0
        stream = wp.stream_from_torch(torch.cuda.current_stream(device))

        def receive() -> dict[str, Any]:
            if head_pid is not None:
                while not self.control.socket.poll(HEAD_POLL_MS):
                    if not process_alive(head_pid):
                        raise RuntimeError(f"The co-simulation head (pid {head_pid}) exited without closing")
            return self.control.receive()

        try:
            while True:
                message = receive()
                try:
                    command = message["type"]
                    if command == MessageType.OPEN_REQUEST and mapped is None:
                        if (head_gpu := message["gpu_uuid"]) != (own_gpu := gpu_uuid(device)):
                            raise ValueError(f"Head GPU {head_gpu} differs from peer GPU {own_gpu}")
                        fields = {raw["name"]: FieldSpec.from_message(raw) for raw in message["fields"]}
                        shared = [
                            BufferSpec(name, fields[name].shape, handle, fields[name].dtype)
                            for name, handle in message["handles"].items()
                        ]
                        writes = [fields[name] for name in message["writes"]]
                        reads = self.handler.open(shared, writes, message["meta"])
                        if unknown := sorted(set(reads) - {spec.name for spec in shared}):
                            raise ValueError(f"Peer reads state fields {unknown} the head does not share")
                        mapped = MappedBuffers([spec for spec in shared if spec.name in reads], device=device)
                        mapped.serve(self.handler.link(mapped.tensors))
                        ready = wp.event_from_ipc_handle(bytes.fromhex(message["ready_handle"]), device=device)
                        done = wp.Event(device=device, interprocess=True)
                        head_pid = int(message["head_pid"])
                        self.control.socket.send_json(
                            {
                                "type": MessageType.OPEN_REPLY,
                                "done_handle": done.ipc_handle().hex(),
                                "exports": [
                                    asdict(share(name, allocation))
                                    for name, allocation in self.handler.allocations.items()
                                ],
                                "reads": list(reads),
                            }
                        )
                    elif command == MessageType.EXCHANGE_REQUEST and mapped is not None:
                        snapshot = Snapshot(**message["snapshot"])
                        if snapshot.sequence != sequence + 1:
                            raise ValueError("Snapshots must be consecutive")
                        stream.wait_event(ready)
                        mapped.record_exchange()
                        self.handler.process(snapshot, mapped.tensors)
                        mapped.verify_read_only()
                        stream.record_event(done)
                        sequence = snapshot.sequence
                        self.control.socket.send_json({"type": MessageType.EXCHANGE_REPLY, "sequence": sequence})
                    elif command == MessageType.CLOSE_REQUEST and mapped is not None:
                        torch.cuda.synchronize(device)
                        mapped.verify_read_only()
                        self.handler.unlink()
                        mapped.release()
                        mapped = None
                        self.control.socket.send_json({"type": MessageType.CLOSE_REPLY})
                        return
                    else:
                        raise ValueError(f"Unexpected bridge command: {command}")
                except Exception as error:
                    self.control.socket.send_json({"type": MessageType.ERROR, "message": str(error)})
                    raise
        except BaseException:
            if mapped is not None:
                with contextlib.suppress(Exception):  # Keep the original failure.
                    torch.cuda.synchronize(device)
                    self.handler.unlink()
                    mapped.release()
            raise
        finally:
            self.control.close()
