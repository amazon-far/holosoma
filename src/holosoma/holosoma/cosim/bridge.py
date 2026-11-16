"""Optional lockstep bridge using only native Holosoma state/capture APIs."""

from __future__ import annotations

import contextlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from holosoma.cosim.cuda import BufferSpec, CacheAllocator, MappedBuffers, allocate_shared, share

if TYPE_CHECKING:
    import torch

    from holosoma.envs.render_task import RenderTask
    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator
    from holosoma.simulator.shared.sensor_manager import CameraRecord

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
        self.awaiting_reply = False
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
        if self.awaiting_reply:
            # A request interrupted before its reply, e.g. by a signal, leaves REQ waiting for that reply.
            self.receive()
            self.awaiting_reply = False
        self.socket.send_json(dict(type=command, **fields))
        self.awaiting_reply = True
        reply = self.receive()
        self.awaiting_reply = False
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


def _plain(value: Any) -> Any:
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


def buffer_spec(raw: Mapping[str, Any]) -> BufferSpec:
    return BufferSpec(**{**raw, "shape": tuple(raw["shape"])})


def scene_layout(sim: BaseSimulator) -> dict[str, Any]:
    """The joint, object and environment layout both sides' state must share."""
    layout: dict[str, Any] = _plain(
        {
            "dof_names": list(sim.dof_names),
            "object_names": list(sim.object_state_names),
            "env_origins": sim.scene.env_origins.cpu().tolist(),
        }
    )
    return layout


def image_name(camera: str, data_type: str) -> str:
    return f"{camera}/{data_type}"


def camera_requests(record: CameraRecord, num_envs: int) -> list[BufferRequest]:
    """One request per output of ``record``: rgb ``uint8 [E, H, W, 3]``, depth ``float32 [E, H, W, 1]``."""
    config = record.config
    meta = _plain(
        {
            "mount": asdict(config.mount),
            "vertical_fov": config.vertical_fov,
            "near": config.near,
            "far": config.far,
            "depth_clipping_behavior": config.depth_clipping_behavior,
        }
    )
    return [
        BufferRequest(
            image_name(record.name, data_type),
            (num_envs, config.height, config.width, 3 if data_type == "rgb" else 1),
            "uint8" if data_type == "rgb" else "float32",
            meta,
        )
        for data_type in config.data_types
    ]


class Publisher:
    """Publish source state for rendering and hold cache storage until completion.

    Without remote cameras, a frame is published after native capture whenever a camera is due. With remote
    cameras, this is their remote renderer: a frame is published when one is due, and the receiver's images
    become their buffers.
    """

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
        remote = manager.remote_cameras
        self.control = ControlChannel(endpoint=endpoint)
        self.ready = wp.Event(device=sim.sim_device, interprocess=True)
        self.sequence = self.physics_step = 0
        self.closed = False
        try:
            requests = [request for record in remote for request in camera_requests(record, manager.num_envs)]
            reply = self.control.exchange(
                "OPEN",
                exports=[asdict(spec) for spec in allocator.exports(self.values())],
                requests=[asdict(request) for request in requests],
                ready_handle=self.ready.ipc_handle().hex(),
                gpu_uuid=gpu_uuid(sim.sim_device),
                scene=scene_layout(sim),
            )
            if reply["type"] != "OPENED":
                raise RuntimeError("Receiver did not accept state buffers")
            self.done = wp.event_from_ipc_handle(bytes.fromhex(reply["done_handle"]), device=sim.sim_device)
            exported = {spec.name: (spec.shape, spec.dtype) for spec in map(buffer_spec, reply["exports"])}
            requested = {request.name: (request.shape, request.dtype) for request in requests}
            if exported != requested:
                raise RuntimeError(f"Receiver exported {exported}, requested {requested}")
            self.images = MappedBuffers(reply["exports"], device=sim.sim_device)
        except BaseException:
            self.control.close()
            raise
        self.step_hook = sim.hooks.add(Phase.POST_STEP, self._stepped, name="cosim.physics_step")
        self.capture_hook = None
        if remote:
            manager.set_remote_renderer(self)
        else:
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
        if self.sim.sensor_manager.last_due:
            self._publish()

    def _exchange(self, command: str, **fields: Any) -> dict[str, Any]:
        """Share current state, send ``command`` and order later work on this stream after the receiver's."""
        import torch
        import warp as wp

        self.values()  # Refresh the same persistent caches, without replacing storage.
        torch.cuda.synchronize(self.sim.sim_device)
        stream = wp.stream_from_torch(torch.cuda.current_stream(self.sim.sim_device))
        stream.record_event(self.ready)
        reply = self.control.exchange(command, **fields)
        stream.wait_event(self.done)
        return reply

    def _publish(self) -> None:
        self.sequence += 1
        frame = Frame(self.sequence, self.physics_step, self.sim.time())
        reply = self._exchange("FRAME_AVAILABLE", frame=asdict(frame))
        if reply["type"] != "RENDER_COMPLETE" or reply["sequence"] != frame.sequence:
            raise RuntimeError("Receiver completed the wrong frame")
        if frame.sequence == 1:
            self.control.set_timeout(FRAME_TIMEOUT_MS)

    def render_remote(self, cameras: list[CameraRecord]) -> None:
        """Publish a frame and set ``cameras``' buffers to copies of the receiver's images."""
        self._publish()
        for record in cameras:
            for data_type in record.config.data_types:
                record.set_buffer(data_type, self.images.tensors[image_name(record.name, data_type)].clone())

    def rerender_remote(self, cameras: list[CameraRecord], env_ids: torch.Tensor) -> None:
        """Have the receiver render current state at the current time, and take its ``env_ids`` rows."""
        reply = self._exchange("RERENDER", sim_time=self.sim.time())
        if reply["type"] != "RERENDERED":
            raise RuntimeError("Receiver did not re-render")
        for record in cameras:
            for data_type in record.config.data_types:
                image = self.images.tensors[image_name(record.name, data_type)]
                if data_type in record.buffers:
                    record.set_buffer_rows(data_type, env_ids, image[env_ids])
                else:
                    record.set_buffer(data_type, image.clone())

    def close(self) -> None:
        import torch

        if self.closed:
            return
        self.closed = True
        self.step_hook.remove()
        if self.capture_hook is None:
            self.sim.sensor_manager.set_remote_renderer(None)
        else:
            self.capture_hook.remove()
        try:
            # Unmap images before the receiver may free them.
            torch.cuda.synchronize(self.sim.sim_device)
            self.images.release()
            reply = self.control.exchange("CLOSE")
            if reply["type"] != "UNMAPPED":
                raise RuntimeError("Receiver did not release shared state")
        finally:
            self.control.close()


@dataclass(frozen=True)
class Image:
    """One exported remote camera output; ``owner`` holds the shared allocation ``tensor`` views."""

    record: CameraRecord
    data_type: str
    owner: Any
    tensor: torch.Tensor


class Receiver:
    """Apply shared state to a render task, render it at the source's clock and export remote camera images."""

    def __init__(self, task: RenderTask, *, endpoint: str) -> None:
        self.task = task
        self.sim = task.simulator
        self.control = ControlChannel(endpoint=endpoint, receiver=True)

    def _check_state(self, exports: list[BufferSpec]) -> None:
        """Check that every exported state field is writable here, with this task's layout."""
        state = self.sim.state
        if unsupported := sorted({spec.name for spec in exports} - set(state.writable_names())):
            raise ValueError(f"Receiver cannot apply state fields {unsupported}")
        for spec in exports:
            own = state.get(spec.name)
            if (spec.shape, spec.dtype) != (tuple(own.shape), str(own.dtype).removeprefix("torch.")):
                raise ValueError(
                    f"State field {spec.name!r} differs between simulators: "
                    f"source {spec.dtype} {spec.shape}, receiver {own.dtype} {tuple(own.shape)}"
                )

    def _allocate_images(self, requests: list[BufferRequest]) -> dict[str, Image]:
        """Check the source's requests against this task's cameras and allocate one export per request."""
        import warp as wp

        manager = self.sim.sensor_manager
        images = {}
        for request in requests:
            camera, _, data_type = request.name.rpartition("/")
            own = {}
            if manager.has_camera(camera):
                own = {output.name: output for output in camera_requests(manager.get(camera), self.task.num_envs)}
            if request.name not in own:
                raise ValueError(f"Receiver has no camera output {request.name!r} for the source's request")
            if request != own[request.name]:
                raise ValueError(
                    f"Camera output {request.name!r} differs between simulators: "
                    f"source {request}, receiver {own[request.name]}"
                )
            owner = allocate_shared(request.shape, request.dtype, self.sim.sim_device)
            images[request.name] = Image(manager.get(camera), data_type, owner, wp.to_torch(owner))
        return images

    def run(self) -> None:
        import torch
        import warp as wp

        mapped: MappedBuffers | None = None
        ready = done = None
        images: dict[str, Image] = {}
        sequence = 0
        stream = wp.stream_from_torch(torch.cuda.current_stream(self.sim.sim_device))

        def render(sim_time: float, *, repeat: bool = False) -> None:
            assert mapped is not None
            stream.wait_event(ready)
            torch.cuda.synchronize(self.sim.sim_device)
            self.sim.apply_state(mapped.tensors)
            self.task.render_frame(sim_time=sim_time, repeat=repeat)
            mapped.verify_read_only()
            for image in images.values():
                image.tensor.copy_(image.record.buffers[image.data_type])
            stream.record_event(done)

        try:
            while True:
                message = self.control.receive()
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
                        self._check_state([buffer_spec(raw) for raw in message["exports"]])
                        images = self._allocate_images([BufferRequest.from_message(raw) for raw in message["requests"]])
                        mapped = MappedBuffers(message["exports"], device=self.sim.sim_device)
                        ready = wp.event_from_ipc_handle(
                            bytes.fromhex(message["ready_handle"]),
                            device=self.sim.sim_device,
                        )
                        done = wp.Event(device=self.sim.sim_device, interprocess=True)
                        specs = [asdict(share(name, image.owner)) for name, image in images.items()]
                        self.control.socket.send_json(
                            {"type": "OPENED", "done_handle": done.ipc_handle().hex(), "exports": specs}
                        )
                    elif command == "FRAME_AVAILABLE" and mapped is not None:
                        frame = Frame(**message["frame"])
                        if frame.sequence != sequence + 1:
                            raise ValueError("Frames must be consecutive")
                        render(frame.sim_time)
                        sequence = frame.sequence
                        self.control.socket.send_json({"type": "RENDER_COMPLETE", "sequence": sequence})
                        if sequence == 1:
                            self.control.set_timeout(FRAME_TIMEOUT_MS)
                    elif command == "RERENDER" and mapped is not None:
                        render(message["sim_time"], repeat=True)
                        self.control.socket.send_json({"type": "RERENDERED"})
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
