"""Optional lockstep bridge using only native Holosoma state/capture APIs."""

from __future__ import annotations

import contextlib
import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from holosoma.cosim.cuda import CacheAllocator, MappedBuffers

if TYPE_CHECKING:
    from holosoma.envs.render_task import RenderTask
    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator

# Bounds each exchange after the first frame; startup and the first frame use the startup timeout.
FRAME_TIMEOUT_MS = 30000


@dataclass(frozen=True)
class Frame:
    sequence: int
    physics_step: int
    sim_time: float

    def __post_init__(self) -> None:
        if self.sequence < 1 or self.physics_step < 0 or not math.isfinite(self.sim_time) or self.sim_time < 0:
            raise ValueError("Frame sequence, physics step and simulation time must be valid")


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


def state_fields(sim: BaseSimulator) -> tuple[str, ...]:
    """The source's non-empty writable unified-state fields."""
    return tuple(name for name in sim.state.writable_names() if sim.state.get(name).numel())


class Publisher:
    """Publish after native capture and hold cache storage until completion."""

    def __init__(
        self,
        sim: BaseSimulator,
        allocator: CacheAllocator,
        *,
        endpoint: str,
        fields: tuple[str, ...] | None = None,
    ) -> None:
        import warp as wp

        from holosoma.simulator.base_simulator.hooks import Phase

        self.sim, self.allocator = sim, allocator
        self.fields = state_fields(sim) if fields is None else fields
        if missing := sorted(set(self.fields) - allocator.fields):
            raise ValueError(f"CacheAllocator does not own published state fields {missing}")
        manager = sim.sensor_manager
        if not manager.cameras:
            raise ValueError("The source publishes when its cameras are due; configure at least one camera")
        self.control = ControlChannel(endpoint=endpoint)
        self.ready = wp.Event(device=sim.sim_device, interprocess=True)
        self.sequence = self.physics_step = 0
        self.closed = False
        try:
            reply = self.control.exchange(
                "OPEN",
                buffers=[asdict(spec) for spec in allocator.exports(self.values())],
                ready_handle=self.ready.ipc_handle().hex(),
                gpu_uuid=gpu_uuid(sim.sim_device),
                dof_names=list(sim.dof_names),
                object_names=list(sim.object_state_names),
                env_origins=sim.scene.env_origins.cpu().tolist(),
            )
            if reply["type"] != "OPENED":
                raise RuntimeError("Receiver did not accept state buffers")
            self.done = wp.event_from_ipc_handle(bytes.fromhex(reply["done_handle"]), device=sim.sim_device)
        except BaseException:
            self.control.close()
            raise
        self.step_hook = sim.hooks.add(Phase.POST_STEP, self._stepped, name="cosim.physics_step")
        self.capture_hook = sim.hooks.add(
            Phase.FRAME_END,
            self._capture,
            name="cosim.publish",
            after=sim.sensor_render_hook,
        )
        self.close_hook = sim.hooks.add(Phase.CLOSE, self.close, name="cosim.close")

    def values(self) -> dict[str, Any]:
        return {name: self.sim.state.get(name) for name in self.fields}

    def _stepped(self) -> None:
        self.physics_step += 1

    def _capture(self) -> None:
        import torch
        import warp as wp

        manager = self.sim.sensor_manager
        if not manager.last_due:
            return
        self.values()  # Refresh the same persistent caches, without replacing storage.
        torch.cuda.synchronize(self.sim.sim_device)
        stream = wp.stream_from_torch(torch.cuda.current_stream(self.sim.sim_device))
        stream.record_event(self.ready)
        self.sequence += 1
        frame = Frame(self.sequence, self.physics_step, self.sim.time())
        reply = self.control.exchange("FRAME_AVAILABLE", frame=asdict(frame))
        if reply["type"] != "RENDER_COMPLETE" or reply["sequence"] != frame.sequence:
            raise RuntimeError("Receiver completed the wrong frame")
        if frame.sequence == 1:
            self.control.set_timeout(FRAME_TIMEOUT_MS)
        stream.wait_event(self.done)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.step_hook.remove()
        self.capture_hook.remove()
        try:
            reply = self.control.exchange("CLOSE")
            if reply["type"] != "UNMAPPED":
                raise RuntimeError("Receiver did not release shared state")
        finally:
            self.control.close()


class Receiver:
    """Apply shared state to a render task and render it at the source's clock."""

    def __init__(self, task: RenderTask, *, endpoint: str) -> None:
        self.task = task
        self.sim = task.simulator
        self.control = ControlChannel(endpoint=endpoint, receiver=True)

    def run(self) -> None:
        import torch
        import warp as wp

        mapped = None
        ready = done = None
        sequence = 0
        stream = wp.stream_from_torch(torch.cuda.current_stream(self.sim.sim_device))
        try:
            while True:
                message = self.control.receive()
                try:
                    command = message["type"]
                    if command == "OPEN" and mapped is None:
                        fields = {spec["name"] for spec in message["buffers"]}
                        if unsupported := sorted(fields - set(self.sim.state.writable_names())):
                            raise ValueError(f"Receiver cannot apply state fields {unsupported}")
                        if tuple(message["dof_names"]) != tuple(self.sim.dof_names):
                            raise ValueError("Joint ordering differs between simulators")
                        if tuple(message["object_names"]) != self.sim.object_state_names:
                            raise ValueError("Object ordering differs between simulators")
                        if message["env_origins"] != self.sim.scene.env_origins.cpu().tolist():
                            raise ValueError("Environment placement differs between simulators")
                        if (source_gpu := message["gpu_uuid"]) != (own_gpu := gpu_uuid(self.sim.sim_device)):
                            raise ValueError(f"Source GPU {source_gpu} differs from receiver GPU {own_gpu}")
                        mapped = MappedBuffers(message["buffers"], device=self.sim.sim_device)
                        ready = wp.event_from_ipc_handle(
                            bytes.fromhex(message["ready_handle"]),
                            device=self.sim.sim_device,
                        )
                        done = wp.Event(device=self.sim.sim_device, interprocess=True)
                        self.control.socket.send_json({"type": "OPENED", "done_handle": done.ipc_handle().hex()})
                    elif command == "FRAME_AVAILABLE" and mapped is not None:
                        frame = Frame(**message["frame"])
                        if frame.sequence != sequence + 1:
                            raise ValueError("Frames must be consecutive")
                        stream.wait_event(ready)
                        torch.cuda.synchronize(self.sim.sim_device)
                        self.sim.apply_state(mapped.tensors)
                        self.task.render_frame(sim_time=frame.sim_time)
                        mapped.verify_read_only()
                        stream.record_event(done)
                        sequence = frame.sequence
                        self.control.socket.send_json({"type": "RENDER_COMPLETE", "sequence": sequence})
                        if sequence == 1:
                            self.control.set_timeout(FRAME_TIMEOUT_MS)
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
