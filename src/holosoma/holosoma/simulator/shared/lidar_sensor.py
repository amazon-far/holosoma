"""Shared LiDAR ray-pattern providers in the canonical optical sensor frame."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

from holosoma.config_types.sensor import (
    CustomLidarRayPatternConfig,
    GridLidarRayPatternConfig,
)
from holosoma.utils.safe_torch_import import torch


def _is_plain_int(value: object) -> bool:
    return type(value) is int


@dataclass(frozen=True)
class ResolvedRayPattern:
    directions: torch.Tensor
    """Normalized ``[R, 3]`` directions in the canonical optical sensor frame."""

    height: int
    width: int


class LidarPatternProvider(ABC):
    """Fixed-shape, simulation-time-indexed optical-frame ray directions."""

    def __init__(self, *, device: torch.device | str, height: int, width: int) -> None:
        self.device = torch.device(device)
        self.height = height
        self.width = width

    @property
    def ray_count(self) -> int:
        return self.height * self.width

    @property
    @abstractmethod
    def time_varying(self) -> bool:
        pass

    @abstractmethod
    def directions_at(self, sim_time: float) -> torch.Tensor:
        """Return ordered, normalized ``[R,3]`` directions for one instantaneous snapshot."""

    def pattern_at(self, sim_time: float) -> ResolvedRayPattern:
        directions = self.directions_at(sim_time)
        if directions.shape != (self.ray_count, 3):
            raise RuntimeError(
                f"LiDAR pattern provider changed shape: expected {(self.ray_count, 3)}, got {directions.shape}."
            )
        if not torch.isfinite(directions).all():
            raise ValueError("LiDAR pattern provider returned non-finite ray directions.")
        norms = torch.linalg.vector_norm(directions, dim=1, keepdim=True)
        if torch.any(norms < 1e-8):
            raise ValueError("LiDAR pattern provider returned a zero ray direction.")
        if not torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5):
            directions = directions / norms
        return ResolvedRayPattern(directions=directions, height=self.height, width=self.width)


class _StaticLidarPatternProvider(LidarPatternProvider):
    def __init__(self, directions: torch.Tensor, *, height: int, width: int) -> None:
        super().__init__(device=directions.device, height=height, width=width)
        self._directions = directions

    @property
    def time_varying(self) -> bool:
        return False

    def directions_at(self, sim_time: float) -> torch.Tensor:
        return self._directions


class GridLidarPatternProvider(_StaticLidarPatternProvider):
    """Device-resident fixed grid of uniformly or explicitly sampled ray directions."""

    def __init__(
        self,
        config: GridLidarRayPatternConfig,
        *,
        device: torch.device | str,
        publish_hz: float,
    ) -> None:
        def axis_samples(angles: list[float] | None, fov: list[float], resolution: float) -> torch.Tensor:
            if angles is not None:
                return torch.tensor(angles, dtype=torch.float32, device=device)
            start, end = fov
            count = max(1, math.ceil((end - start) / resolution))
            return start + torch.arange(count, dtype=torch.float32, device=device) * resolution

        azimuth = axis_samples(config.horizontal_angles, config.horizontal_fov, config.horizontal_resolution)
        elevation = axis_samples(config.vertical_angles, config.vertical_fov, config.vertical_resolution)
        elevations, azimuths = torch.meshgrid(elevation, azimuth, indexing="ij")
        elevations = torch.deg2rad(elevations)
        azimuths = torch.deg2rad(azimuths)
        directions = torch.stack(
            [
                -torch.cos(elevations) * torch.sin(azimuths),
                torch.sin(elevations),
                -torch.cos(elevations) * torch.cos(azimuths),
            ],
            dim=-1,
        ).reshape(-1, 3)
        super().__init__(directions, height=len(elevation), width=len(azimuth))


class CustomLidarPatternProvider(_StaticLidarPatternProvider):
    """Device-resident fixed grid of normalized user-supplied ray directions."""

    def __init__(
        self,
        config: CustomLidarRayPatternConfig,
        *,
        device: torch.device | str,
        publish_hz: float,
    ) -> None:
        directions = torch.tensor(config.ray_directions, dtype=torch.float32, device=device)
        directions = directions / torch.linalg.vector_norm(directions, dim=1, keepdim=True)
        height, width = config.organized_shape or [1, len(config.ray_directions)]
        super().__init__(directions, height=height, width=width)


class CyclicLidarPatternProvider(LidarPatternProvider):
    """Fixed-size windows from an ordered ray sequence that repeats over simulation time.

    Extension providers load or generate the complete normalized ``sequence`` and choose the
    snapshot ``height`` and ``width``. Each measurement returns the next contiguous window,
    wrapping at the end. The provider does not model per-ray acquisition time or motion
    distortion: every returned window remains one instantaneous cloud.
    """

    def __init__(
        self,
        sequence: torch.Tensor,
        *,
        height: int,
        width: int,
        publish_hz: float,
        sequence_offset: int = 0,
    ) -> None:
        if not math.isfinite(publish_hz) or publish_hz <= 0.0:
            raise ValueError(f"LiDAR publish_hz must be finite and > 0, got {publish_hz}.")
        if not _is_plain_int(height) or not _is_plain_int(width):
            raise ValueError(f"LiDAR snapshot shape must use integer dimensions, got {(height, width)}.")
        if height <= 0 or width <= 0:
            raise ValueError(f"LiDAR snapshot shape must be positive, got {(height, width)}.")
        if sequence.ndim != 2 or sequence.shape[0] == 0 or sequence.shape[1] != 3:
            raise ValueError(f"LiDAR ray sequence must have shape [N, 3] with N > 0, got {tuple(sequence.shape)}.")
        if not _is_plain_int(sequence_offset) or sequence_offset < 0:
            raise ValueError(f"LiDAR sequence_offset must be >= 0, got {sequence_offset}.")
        sequence = sequence.to(dtype=torch.float32)
        if not torch.isfinite(sequence).all():
            raise ValueError("LiDAR ray sequence must contain only finite directions.")
        norms = torch.linalg.vector_norm(sequence, dim=1, keepdim=True)
        if torch.any(norms < 1e-8):
            raise ValueError("LiDAR ray sequence must not contain a zero direction.")
        super().__init__(device=sequence.device, height=height, width=width)
        self.publish_hz = publish_hz
        self.sequence_offset = sequence_offset
        self._sequence = sequence / norms
        self._time_origin = 0.0
        self._last_sim_time: float | None = None

    @property
    def time_varying(self) -> bool:
        return True

    def directions_at(self, sim_time: float) -> torch.Tensor:
        if not math.isfinite(sim_time) or sim_time < 0.0:
            raise ValueError(f"LiDAR simulation time must be finite and >= 0, got {sim_time}.")
        if self._last_sim_time is not None and sim_time < self._last_sim_time:
            self._time_origin = sim_time
        self._last_sim_time = sim_time
        elapsed = sim_time - self._time_origin
        elapsed_snapshots = elapsed * self.publish_hz
        nearest_snapshot = round(elapsed_snapshots)
        # MuJoCo Warp exposes its simulation clock as float32. Snap a few float32 ULPs around a
        # completed measurement boundary, while leaving materially early reads in the prior window.
        float32_tolerance = 4.0 * torch.finfo(torch.float32).eps * max(1.0, abs(elapsed_snapshots))
        if abs(elapsed_snapshots - nearest_snapshot) <= float32_tolerance:
            snapshot_index = nearest_snapshot
        else:
            snapshot_index = math.floor(elapsed_snapshots)
        start = self.sequence_offset + snapshot_index * self.ray_count
        indices = torch.arange(self.ray_count, device=self.device, dtype=torch.long)
        indices = torch.remainder(indices + start, self._sequence.shape[0])
        return self._sequence.index_select(0, indices)
