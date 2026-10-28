"""Backend-neutral mounted-sensor registry, cadence, and output buffering."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

from holosoma.config_types.frequency import resolve_decimation
from holosoma.config_types.sensor import CameraSensorConfig, WarpDepthCameraConfig
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
    """Bookkeeping shared by every mounted sensor type."""

    name: str
    config: SensorConfig
    buffers: dict[str, torch.Tensor] = field(default_factory=dict)
    step_counter: int = -1
    effective_decimation: int = 1
    rerender: Callable[[torch.Tensor], None] | None = None
    """Re-render only the given env ids into the existing buffers, or ``None`` if the backend
    can only render every env. Called after resets/teleports (``SensorManager.rerender``)."""
    _device_cache: dict[tuple[str, str], torch.Tensor] = field(default_factory=dict)

    def set_buffer(self, output: str, tensor: torch.Tensor) -> None:
        self.buffers[output] = tensor
        self._drop_device_cache(output)

    def set_buffer_rows(self, output: str, env_ids: torch.Tensor, rows: torch.Tensor) -> None:
        """Overwrite the ``env_ids`` rows of an existing output, e.g. after a partial re-render."""
        buffer = self.buffers[output].clone()
        buffer[env_ids] = rows
        self.buffers[output] = buffer
        self._drop_device_cache(output)

    def _drop_device_cache(self, output: str) -> None:
        for key in [key for key in self._device_cache if key[0] == output]:
            del self._device_cache[key]

    def buffer_on(self, output: str, device: torch.device | str | None = None) -> torch.Tensor:
        buf = self.buffers[output]
        if device is None:
            return buf
        dev = torch.device(device)
        if buf.device == dev:
            return buf
        key = (output, str(dev))
        cached = self._device_cache.get(key)
        if cached is None:
            cached = buf.to(dev)
            self._device_cache[key] = cached
        return cached


def _clip_depth(config: object, tensor: torch.Tensor) -> torch.Tensor:
    """Apply a camera's configured depth no-return semantics (records without a camera config store as-is)."""
    if not isinstance(config, (CameraSensorConfig, WarpDepthCameraConfig)):
        return tensor
    return clip_sensor_ranges(
        tensor,
        near=config.near,
        far=config.far,
        behavior=config.depth_clipping_behavior,
    ).ranges


@dataclass
class CameraRecord(SensorRecord):
    """A camera its backend renders natively."""

    config: CameraSensorConfig

    def set_buffer(self, output: str, tensor: torch.Tensor) -> None:
        """Store a camera output, applying configured depth no-return semantics at the shared boundary."""
        super().set_buffer(output, _clip_depth(self.config, tensor) if output == "depth" else tensor)

    def set_buffer_rows(self, output: str, env_ids: torch.Tensor, rows: torch.Tensor) -> None:
        """Overwrite camera output rows, applying the same depth no-return semantics as :meth:`set_buffer`."""
        super().set_buffer_rows(output, env_ids, _clip_depth(self.config, rows) if output == "depth" else rows)


@dataclass
class SharedCameraRecord(SensorRecord):
    """A camera rendered by shared, backend-neutral code (see ``SensorManager.register_shared_camera``)."""

    config: WarpDepthCameraConfig
    camera: Any = None
    """The object that renders this camera, e.g. for randomization terms that configure it."""

    def set_buffer(self, output: str, tensor: torch.Tensor) -> None:
        super().set_buffer(output, _clip_depth(self.config, tensor) if output == "depth" else tensor)

    def set_buffer_rows(self, output: str, env_ids: torch.Tensor, rows: torch.Tensor) -> None:
        super().set_buffer_rows(output, env_ids, _clip_depth(self.config, rows) if output == "depth" else rows)


@dataclass
class LidarRecord(SensorRecord):
    config: LidarSensorConfig
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


class SensorManager:
    """Registry and independent update cadence for cameras and LiDARs."""

    def __init__(self, device: str, control_hz: float) -> None:
        self.device = device
        self.control_hz = control_hz
        self._cameras: dict[str, CameraRecord] = {}
        # Cameras rendered by shared, backend-neutral code rather than by the backend itself.
        self._shared_cameras: dict[str, SharedCameraRecord] = {}
        self._shared_renderers: dict[str, Callable[[SharedCameraRecord, torch.Tensor | None], None]] = {}
        self._lidars: dict[str, LidarRecord] = {}
        self._order: list[str] = []
        self._last_camera_due: set[str] = set()
        self._last_lidar_due: set[str] = set()

    def _assert_name_available(self, name: str) -> None:
        if name in self._cameras or name in self._shared_cameras or name in self._lidars:
            raise ValueError(f"Sensor '{name}' already registered.")

    def register_camera(self, name: str, config: CameraSensorConfig) -> CameraRecord:
        self._assert_name_available(name)
        effective = resolve_decimation(
            config.update_decimation, self.control_hz, field=f"camera '{name}' update_decimation"
        )
        record = CameraRecord(name=name, config=config, effective_decimation=effective)
        self._cameras[name] = record
        self._order.append(name)
        return record

    def register_shared_camera(
        self,
        name: str,
        config: WarpDepthCameraConfig,
        render: Callable[[SharedCameraRecord, torch.Tensor | None], None],
    ) -> SharedCameraRecord:
        """Register a camera that shared code renders on every backend.

        ``render(record, env_ids)`` renders every env (``env_ids=None``) or only ``env_ids`` into
        ``record``. It runs from :meth:`render_shared` on the camera's cadence and, for envs reset
        or teleported since the last render, from :meth:`rerender`. Backends' own render loops never see it:
        :meth:`collect_due` returns only backend-rendered cameras."""
        self._assert_name_available(name)
        effective = resolve_decimation(
            config.update_decimation, self.control_hz, field=f"camera '{name}' update_decimation"
        )
        record = SharedCameraRecord(name=name, config=config, effective_decimation=effective)
        record.rerender = lambda env_ids: render(record, env_ids)
        self._shared_cameras[name] = record
        self._shared_renderers[name] = render
        self._order.append(name)
        return record

    def register_lidar(self, name: str, config: LidarSensorConfig) -> LidarRecord:
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
        )
        self._lidars[name] = record
        self._order.append(name)
        return record

    def has_camera(self, name: str) -> bool:
        return name in self._cameras or name in self._shared_cameras

    def has_lidar(self, name: str) -> bool:
        return name in self._lidars

    def get(self, name: str) -> CameraRecord | SharedCameraRecord:
        if name in self._shared_cameras:
            return self._shared_cameras[name]
        if name not in self._cameras:
            raise KeyError(f"No camera named '{name}'. Registered cameras: {self.camera_names}.")
        return self._cameras[name]

    def get_lidar(self, name: str) -> LidarRecord:
        if name not in self._lidars:
            raise KeyError(f"No LiDAR named '{name}'. Registered LiDARs: {sorted(self._lidars)}.")
        return self._lidars[name]

    @property
    def names(self) -> list[str]:
        return list(self._order)

    @property
    def camera_names(self) -> list[str]:
        return [*self._cameras, *self._shared_cameras]

    @property
    def lidar_names(self) -> list[str]:
        return list(self._lidars)

    @property
    def cameras(self) -> list[CameraRecord]:
        """Backend-rendered cameras; shared-code cameras are in :attr:`shared_cameras`."""
        return list(self._cameras.values())

    @property
    def shared_cameras(self) -> list[SharedCameraRecord]:
        return list(self._shared_cameras.values())

    @property
    def lidars(self) -> list[LidarRecord]:
        return list(self._lidars.values())

    @staticmethod
    def _collect_due(records: Sequence[SensorRecordT]) -> list[SensorRecordT]:
        due = []
        for record in records:
            record.step_counter += 1
            if record.step_counter % record.effective_decimation == 0:
                due.append(record)
        return due

    def collect_due(self) -> list[CameraRecord]:
        """Advance the cadence of backend-rendered cameras and return those due this step."""
        due = self._collect_due(self.cameras)
        self._last_camera_due = {record.name for record in due}
        return due

    def render_shared(self) -> None:
        """Render every shared-code camera due this control step (see :meth:`register_shared_camera`)."""
        for record in self._collect_due(self.shared_cameras):
            self._shared_renderers[record.name](record, None)
            self._last_camera_due.add(record.name)

    def collect_lidars_due(self) -> list[LidarRecord]:
        due = self._collect_due(self.lidars)
        self._last_lidar_due = {record.name for record in due}
        return due

    def rerender(self, env_ids: torch.Tensor) -> None:
        """Re-render ``env_ids`` for every sensor that supports partial rendering.

        Sensors without a ``rerender`` callback keep their last frame, so for one control step
        these envs observe the state from before their reset or teleport."""
        records: list[SensorRecord] = [*self.cameras, *self.shared_cameras, *self.lidars]
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
