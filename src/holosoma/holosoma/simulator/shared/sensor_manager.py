"""Backend-neutral mounted-sensor registry, cadence, and output buffering."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

from holosoma.config_types.frequency import resolve_decimation
from holosoma.config_types.sensor import CameraSensorConfig
from holosoma.config_types.value_types import UNIT_QUATERNION_NORM_TOLERANCE
from holosoma.simulator.base_simulator.state import SimState
from holosoma.simulator.shared.lidar_sensor import (
    LidarPatternProvider,
    ResolvedRayPattern,
)
from holosoma.simulator.shared.range_clipping import clip_sensor_ranges
from holosoma.utils.safe_torch_import import torch

if TYPE_CHECKING:
    from holosoma.config_types.sensor import LidarSensorConfig, SensorConfig

SensorRecordT = TypeVar("SensorRecordT", bound="SensorRecord")


@dataclass
class SensorRecord:
    """Bookkeeping shared by every mounted sensor type.

    Each output is one fixed buffer, ``buffers[output]``, served as the read-less SimState field
    :meth:`field_name` (linked onto it) and written in place every frame inside :meth:`writing`.
    """

    name: str
    config: SensorConfig
    buffers: dict[str, torch.Tensor] = field(default_factory=dict)
    step_counter: int = -1
    effective_decimation: int = 1
    rerender: Callable[[torch.Tensor], None] | None = None
    """Re-render only the given env ids into the existing buffers, or ``None`` if the backend
    can only render every env. Called after resets/teleports (``SensorManager.rerender``)."""
    mount_poses: torch.Tensor | None = None
    """Runtime offsets ``[N, 7]`` in XYZ + XYZW order, independent of the immutable config."""
    write_mount_pose: Callable[[torch.Tensor, torch.Tensor], None] | None = None
    """Backend writer taking environment ids and validated XYZ + XYZW offsets."""
    state: SimState = field(default_factory=SimState, repr=False)
    """The state that serves the outputs."""
    written: set[str] = field(default_factory=set)
    """Outputs written since the sensor was last due."""
    rendered: set[str] = field(default_factory=set)
    """Outputs written at least once."""
    _device_cache: dict[tuple[str, str], torch.Tensor] = field(default_factory=dict)

    kind: ClassVar[str] = "sensor"

    def field_name(self, output: str) -> str:
        """The SimState field serving ``output``."""
        return f"{self.kind}/{self.name}/{output}"

    def allocate(self, output: str, shape: tuple[int, ...], dtype: torch.dtype, device: str) -> None:
        """Allocate ``output``'s fixed buffer, zero-filled, and serve it as :meth:`field_name`."""
        name = self.field_name(output)
        with torch.inference_mode(False):
            frame = torch.zeros(shape, dtype=dtype, device=device)
        self.state.register(name, None, validation_groups=(), invalidates=(), dtype=dtype)
        self.buffers[output] = self.state.link({name: frame})[name]

    @contextmanager
    def writing(self, output: str, env_ids: torch.Tensor | None = None) -> Iterator[torch.Tensor]:
        """Write a new frame of ``output`` (its ``env_ids`` rows, or all) into the yielded buffer, in place."""
        with self.state.producing(self.field_name(output)):
            yield self.buffers[output]
            self._finish(output, env_ids)
        self.mark_written(output)

    def _finish(self, output: str, env_ids: torch.Tensor | None) -> None:
        """Post-process a frame just written into ``output``'s buffer."""

    def mark_written(self, output: str) -> None:
        """Record a new frame in ``output``'s buffer written by someone else, e.g. a co-simulation peer."""
        self.written.add(output)
        self.rendered.add(output)
        self._drop_device_cache(output)

    def set_buffer(self, output: str, tensor: torch.Tensor) -> None:
        """Copy a finished frame into ``output``'s buffer."""
        with self.writing(output) as buffer:
            if tensor.shape != buffer.shape:
                raise ValueError(f"{self.name} {output} must be {tuple(buffer.shape)}, got {tuple(tensor.shape)}")
            buffer.copy_(tensor)

    def set_buffer_rows(self, output: str, env_ids: torch.Tensor, rows: torch.Tensor) -> None:
        """Copy frames for ``env_ids`` into ``output``'s buffer, e.g. after a partial re-render."""
        with self.writing(output, env_ids) as buffer:
            buffer[env_ids] = rows

    def _drop_device_cache(self, output: str) -> None:
        for key in [key for key in self._device_cache if key[0] == output]:
            del self._device_cache[key]

    def buffer_on(self, output: str, device: torch.device | str | None = None) -> torch.Tensor:
        """``output``'s buffer, or with ``device``, a snapshot of the current frame there (cached per frame)."""
        buf = self.buffers[output]
        if device is None:
            return buf
        dev = torch.device(device)
        key = (output, str(dev))
        cached = self._device_cache.get(key)
        if cached is None:
            cached = buf.clone() if buf.device == dev else buf.to(dev)
            self._device_cache[key] = cached
        return cached


@dataclass
class CameraRecord(SensorRecord):
    config: CameraSensorConfig

    kind: ClassVar[str] = "camera"

    def allocate_outputs(self, num_envs: int, device: str) -> None:
        """Allocate rgb ``uint8 [E, H, W, 3]`` and depth ``float32 [E, H, W, 1]`` for the configured types."""
        config = self.config
        for data_type in config.data_types:
            rgb = data_type == "rgb"
            shape = (num_envs, config.height, config.width, 3 if rgb else 1)
            self.allocate(data_type, shape, torch.uint8 if rgb else torch.float32, device)

    def _finish(self, output: str, env_ids: torch.Tensor | None) -> None:
        """Apply the configured depth no-return semantics in place."""
        if output != "depth":
            return
        buffer = self.buffers[output]
        near, far, behavior = self.config.near, self.config.far, self.config.depth_clipping_behavior
        if env_ids is None:
            clip_sensor_ranges(buffer, near=near, far=far, behavior=behavior, out=buffer)
        else:
            buffer[env_ids] = clip_sensor_ranges(buffer[env_ids], near=near, far=far, behavior=behavior).ranges


@dataclass
class LidarRecord(SensorRecord):
    config: LidarSensorConfig

    kind: ClassVar[str] = "lidar"
    pattern_provider: LidarPatternProvider | None = None
    backend_cache: dict[str, Any] = field(default_factory=dict)
    """Backend-owned fixed-shape query buffers and compiled work for this LiDAR."""

    @property
    def pattern_shape(self) -> tuple[int, int]:
        if self.pattern_provider is None:
            raise RuntimeError(f"LiDAR '{self.name}' has no pattern provider.")
        return self.pattern_provider.height, self.pattern_provider.width

    def pattern_at(self, sim_time: float) -> ResolvedRayPattern:
        if self.pattern_provider is None:
            raise RuntimeError(f"LiDAR '{self.name}' has no pattern provider.")
        return self.pattern_provider.pattern_at(sim_time)


def _unit_quaternions(quaternions: torch.Tensor) -> torch.Tensor:
    """Normalise ``[..., 4]`` quaternions; ones within float32 rounding of unit length are returned unchanged."""
    norms = torch.linalg.vector_norm(quaternions, dim=-1, keepdim=True)
    unit = (norms - 1.0).abs() <= 4 * torch.finfo(torch.float32).eps
    return quaternions / torch.where(unit, torch.ones_like(norms), norms)


class SensorManager:
    """Cross-backend sensor registry and runtime mounts, available even without configured sensors."""

    def __init__(self, device: str, control_hz: float, *, num_envs: int = 1, state: SimState | None = None) -> None:
        self.device = device
        self.state = state if state is not None else SimState()
        """The state that serves sensor outputs, as ``camera/<name>/<output>`` and ``lidar/<name>/<output>``."""
        self.control_hz = control_hz
        self.num_envs = num_envs
        self._cameras: dict[str, CameraRecord] = {}
        self._lidars: dict[str, LidarRecord] = {}
        self._order: list[str] = []
        self._last_camera_due: set[str] = set()
        self._last_lidar_due: set[str] = set()
        self.on_mount_pose_set: Callable[[str], None] | None = None
        """Called with the sensor's name after :meth:`set_mount_pose` stores new offsets."""

    def _initial_mount_poses(self, config: SensorConfig) -> torch.Tensor:
        mount = config.mount
        w, x, y, z = mount.orientation
        pose = torch.tensor([*mount.position, x, y, z, w], dtype=torch.float32, device=self.device)
        pose[3:] = _unit_quaternions(pose[3:])
        return pose.repeat(self.num_envs, 1)

    def _assert_name_available(self, name: str) -> None:
        if name in self._cameras or name in self._lidars:
            raise ValueError(f"Sensor '{name}' already registered.")

    def register_camera(self, name: str, config: CameraSensorConfig) -> CameraRecord:
        self._assert_name_available(name)
        effective = resolve_decimation(
            config.update_decimation, self.control_hz, field=f"camera '{name}' update_decimation"
        )
        record = CameraRecord(
            name=name,
            config=config,
            effective_decimation=effective,
            mount_poses=self._initial_mount_poses(config),
            state=self.state,
        )
        record.allocate_outputs(self.num_envs, self.device)
        self._cameras[name] = record
        self._order.append(name)
        return record

    def register_lidar(
        self, name: str, config: LidarSensorConfig, *, outputs: Sequence[str] = ("ranges", "points")
    ) -> LidarRecord:
        """Register a LiDAR with fixed buffers for ``outputs``: ``ranges`` float32 ``[E, R]``, ``points``
        float32 ``[E, R, 3]`` and ``geom_ids`` int32 ``[E, R]``, for backends that report hit geometry."""
        self._assert_name_available(name)
        effective = resolve_decimation(
            config.update_decimation, self.control_hz, field=f"LiDAR '{name}' update_decimation"
        )
        provider_cls = config.pattern.get_provider_cls()
        pattern_provider = provider_cls(
            config.pattern,
            device=self.device,
            publish_hz=self.control_hz / effective,
        )
        record = LidarRecord(
            name=name,
            config=config,
            effective_decimation=effective,
            pattern_provider=pattern_provider,
            mount_poses=self._initial_mount_poses(config),
            state=self.state,
        )
        rays = pattern_provider.ray_count
        shapes = {
            "ranges": ((self.num_envs, rays), torch.float32),
            "points": ((self.num_envs, rays, 3), torch.float32),
            "geom_ids": ((self.num_envs, rays), torch.int32),
        }
        for output in outputs:
            record.allocate(output, *shapes[output], self.device)
        self._lidars[name] = record
        self._order.append(name)
        return record

    def has_camera(self, name: str) -> bool:
        return name in self._cameras

    def has_lidar(self, name: str) -> bool:
        return name in self._lidars

    def get(self, name: str) -> CameraRecord:
        if name not in self._cameras:
            raise KeyError(f"No camera named '{name}'. Registered cameras: {sorted(self._cameras)}.")
        return self._cameras[name]

    def get_lidar(self, name: str) -> LidarRecord:
        if name not in self._lidars:
            raise KeyError(f"No LiDAR named '{name}'. Registered LiDARs: {sorted(self._lidars)}.")
        return self._lidars[name]

    def get_sensor(self, name: str) -> SensorRecord:
        if name in self._cameras:
            return self._cameras[name]
        if name in self._lidars:
            return self._lidars[name]
        raise KeyError(f"No sensor named '{name}'. Registered sensors: {self._order}.")

    def _mount_env_ids(self, env_ids: torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device)
        ids = torch.as_tensor(env_ids, device=self.device)
        if ids.ndim != 1 or ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError("env_ids must be a one-dimensional integer tensor.")
        ids = ids.to(torch.long)
        if torch.any((ids < 0) | (ids >= self.num_envs)):
            raise ValueError(f"env_ids must be in [0, {self.num_envs}).")
        return ids

    def get_mount_pose(self, name: str, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Return a copy of the selected mount poses as ``[N, 7]`` XYZ + XYZW.

        Offsets are relative to the configured parent, or the per-environment frame for world mounts.
        ``env_ids=None`` selects all environments. Mount configuration quaternions use WXYZ.
        """
        record = self.get_sensor(name)
        if record.mount_poses is None:
            raise RuntimeError(f"Sensor '{name}' has no runtime mount state.")
        return record.mount_poses[self._mount_env_ids(env_ids)].clone()

    def set_mount_pose(
        self, name: str, poses: torch.Tensor, env_ids: torch.Tensor | None = None, *, notify: bool = True
    ) -> None:
        """Update camera/LiDAR poses relative to their configured mount.

        Use XYZ + XYZW with unit quaternions: a single ``[7]`` pose broadcasts, while ``[N, 7]``
        sets each selected environment. ``env_ids=None`` selects all environments.
        Offsets persist across resets; sensor outputs reflect the change on their next render/capture.
        A remote camera only stores the offsets, for the plugin that supplies its images. ``notify=False``
        skips :attr:`on_mount_pose_set`. Call outside CUDA graph capture.
        """
        record = self.get_sensor(name)
        ids = self._mount_env_ids(env_ids)
        if ids.unique().numel() != ids.numel():
            raise ValueError("env_ids must not contain duplicates when setting sensor mount poses.")
        remote = isinstance(record, CameraRecord) and bool(record.config.remote)
        if record.mount_poses is None or (record.write_mount_pose is None and not remote):
            raise NotImplementedError(f"Sensor '{name}' has no runtime mount-pose writer.")
        offsets = torch.as_tensor(poses, dtype=torch.float32, device=self.device).detach().clone()
        if offsets.shape == (7,):
            offsets = offsets.expand(len(ids), -1).clone()
        if offsets.shape != (len(ids), 7):
            raise ValueError(f"Sensor mount poses must have shape (7,) or ({len(ids)}, 7), got {offsets.shape}.")
        if not torch.isfinite(offsets).all():
            raise ValueError("Sensor mount poses must contain only finite values.")
        norms = torch.linalg.vector_norm(offsets[:, 3:], dim=-1, keepdim=True)
        if torch.any(torch.abs(norms - 1.0) > UNIT_QUATERNION_NORM_TOLERANCE):
            raise ValueError("Sensor mount orientation must be a unit quaternion in XYZW order.")
        offsets[:, 3:] = _unit_quaternions(offsets[:, 3:])
        if len(ids) == 0:
            return
        if record.write_mount_pose is not None:
            record.write_mount_pose(ids, offsets)
        record.mount_poses[ids] = offsets
        if notify and self.on_mount_pose_set is not None:
            self.on_mount_pose_set(name)

    @property
    def names(self) -> list[str]:
        return list(self._order)

    @property
    def camera_names(self) -> list[str]:
        return list(self._cameras)

    @property
    def lidar_names(self) -> list[str]:
        return list(self._lidars)

    @property
    def cameras(self) -> list[CameraRecord]:
        return list(self._cameras.values())

    @property
    def lidars(self) -> list[LidarRecord]:
        return list(self._lidars.values())

    @staticmethod
    def _collect_due(records: Sequence[SensorRecordT]) -> list[SensorRecordT]:
        due = []
        for record in records:
            record.step_counter += 1
            if record.step_counter % record.effective_decimation == 0:
                record.written.clear()
                due.append(record)
        return due

    def collect_due(self) -> list[CameraRecord]:
        """Advance camera cadence (compatibility name retained for camera backends)."""
        due = self._collect_due(self.cameras)
        self._last_camera_due = {record.name for record in due}
        return due

    def collect_lidars_due(self) -> list[LidarRecord]:
        due = self._collect_due(self.lidars)
        self._last_lidar_due = {record.name for record in due}
        return due

    def rerender(self, env_ids: torch.Tensor) -> None:
        """Re-render ``env_ids`` for every sensor that supports partial rendering.

        Sensors without a ``rerender`` callback keep their last frame, so for one control step
        these envs observe the state from before their reset or teleport."""
        records: list[SensorRecord] = [*self.cameras, *self.lidars]
        for record in records:
            if record.rerender is not None:
                record.rerender(env_ids)

    @property
    def last_due(self) -> set[str]:
        """Camera names rendered by the latest camera collection."""
        return self._last_camera_due

    @property
    def last_lidar_due(self) -> set[str]:
        return self._last_lidar_due

    def frames_produced(self, name: str) -> int:
        record: SensorRecord
        if name in self._cameras:
            record = self._cameras[name]
        elif name in self._lidars:
            record = self._lidars[name]
        else:
            raise KeyError(f"No sensor named '{name}'. Registered sensors: {self._order}.")
        return record.step_counter // record.effective_decimation + 1
