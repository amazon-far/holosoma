"""Shared, backend-agnostic camera-sensor substrate.

Holds the :class:`SensorManager`, a read-only camera registry. It owns one
:class:`CameraRuntime` per configured camera and the decimation lifecycle
(:meth:`SensorManager.collect_due`).

The camera optical frame is -Z forward, +Y up: a camera at an identity mount quaternion looks down
its own -Z. IsaacGym converts the mount orientation to its own camera basis at its boundary.

See ``BaseSimulator.get_camera_data`` for the full public data format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from holosoma.config_types.frequency import resolve_decimation
from holosoma.utils.safe_torch_import import torch

if TYPE_CHECKING:
    from holosoma.config_types.sensor import CameraSensorConfig


@dataclass
class CameraRuntime:
    """Per-camera runtime record held by :class:`SensorManager`.

    Backend-agnostic bookkeeping only; holds no native camera handle. Each backend resolves its own
    native resource from a name-keyed map using ``name`` (the sensor key).
    """

    name: str
    """Sensor name (the --sensor dict key); the backend's native-resource lookup key."""

    config: CameraSensorConfig

    buffers: dict[str, torch.Tensor] = field(default_factory=dict)
    """Most recent render per data_type on the SIM device (e.g. {"rgb": [N,H,W,3] uint8}). Write via
    :meth:`set_buffer` (never assign directly) so the cross-device cache stays consistent."""

    step_counter: int = -1
    """Render counter for decimation gating (``step_counter % decimation == 0``). Advanced by
    :meth:`SensorManager.collect_due`; starts at -1 so the gate fires on the first render."""

    effective_decimation: int = 1
    """Control-step decimation as an int, resolved from ``config.update_decimation`` (which may be
    a frequency string) at registration."""

    _device_cache: dict[tuple[str, str], torch.Tensor] = field(default_factory=dict)
    """Cross-device copies of ``buffers``, keyed by ``(data_type, str(device))``. Populated lazily by
    :meth:`buffer_on` (e.g. the host copy every off-GPU consumer shares), cleared by :meth:`set_buffer`
    when a fresh render lands. Never a copy of the sim-device buffer itself (that is served directly)."""

    def set_buffer(self, data_type: str, tensor: torch.Tensor) -> None:
        """Store the latest render for ``data_type`` and drop any now-stale cross-device copies of it.

        The single write path for a rendered frame: invalidation is co-located with the mutation, so
        a cached host copy can never outlive the sim-device buffer it was derived from. Called by each
        backend's ``render_sensors`` (once per due camera per step) in place of a raw ``buffers[...] =``."""
        self.buffers[data_type] = tensor
        for key in [k for k in self._device_cache if k[0] == data_type]:
            del self._device_cache[key]

    def buffer_on(self, data_type: str, device: torch.device | str | None = None) -> torch.Tensor:
        """Return the full ``[N, ...]`` buffer for ``data_type`` on ``device`` (cross-device copies cached).

        ``device=None`` (or the sim device the buffer already lives on) returns the buffer directly —
        no copy, no cache entry. Any other device returns a cached ``.to(device)`` copy: the FIRST
        caller for a given ``(data_type, device)`` pays the transfer (one device->host sync for
        ``"cpu"``), every later caller this render reuses it. The cache is invalidated by
        :meth:`set_buffer` on the next render, so a slow (decimated) camera keeps serving cheap cached
        reads across the steps it does not re-render."""
        buf = self.buffers[data_type]
        if device is None:
            return buf
        dev = torch.device(device)
        if buf.device == dev:
            return buf
        key = (data_type, str(dev))
        cached = self._device_cache.get(key)
        if cached is None:
            cached = buf.to(dev)
            self._device_cache[key] = cached
        return cached


class SensorManager:
    """Read-only registry of mounted cameras for one simulator instance.

    Holds :class:`CameraRuntime` records keyed by sensor name and exposes name-keyed lookups. Owns
    no settable pose state.
    """

    def __init__(self, device: str, control_hz: float) -> None:
        self.device = device
        self.control_hz = control_hz
        """Control-step rate (fps/control_decimation), the base for per-camera update_decimation."""
        self._cameras: dict[str, CameraRuntime] = {}
        self._last_due: set[str] = set()
        """Names of cameras that rendered on the most recent ``collect_due`` call. Empty until the
        first ``collect_due``."""

    # ----- registration (called by the backend during its sensor setup) -----

    def register_camera(self, name: str, config: CameraSensorConfig) -> CameraRuntime:
        """Register one camera under ``name``. data_types are already validated by ``CameraSensorConfig``."""
        if name in self._cameras:
            raise ValueError(f"Camera '{name}' already registered.")
        eff = resolve_decimation(config.update_decimation, self.control_hz, field=f"camera '{name}' update_decimation")
        runtime = CameraRuntime(name=name, config=config, effective_decimation=eff)
        self._cameras[name] = runtime
        return runtime

    # ----- lookups -----

    def has_camera(self, name: str) -> bool:
        return name in self._cameras

    def get(self, name: str) -> CameraRuntime:
        if name not in self._cameras:
            raise KeyError(f"No camera named '{name}'. Registered cameras: {sorted(self._cameras)}.")
        return self._cameras[name]

    @property
    def names(self) -> list[str]:
        return list(self._cameras)

    @property
    def cameras(self) -> list[CameraRuntime]:
        return list(self._cameras.values())

    def collect_due(self) -> list[CameraRuntime]:
        """Advance every camera's render counter and return those due to render this step.

        Called once per control step by a backend's ``render_sensors`` before the native render.
        """
        due = []
        for runtime in self._cameras.values():
            runtime.step_counter += 1
            if runtime.step_counter % runtime.effective_decimation == 0:
                due.append(runtime)
        self._last_due = {rt.name for rt in due}
        return due

    @property
    def last_due(self) -> set[str]:
        """Camera names that rendered on the most recent ``collect_due`` (empty before the first)."""
        return self._last_due

    def frames_produced(self, name: str) -> int:
        """Number of frames a camera has rendered so far (0 before its first render)."""
        runtime = self.get(name)
        return runtime.step_counter // runtime.effective_decimation + 1  # -1 -> 0 (floor div)
