# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""LiDAR output finalization and sensor-frame point conversion."""

from __future__ import annotations

from typing import Protocol

from holosoma.config_types.sensor import LidarSensorConfig
from holosoma.simulator.shared.range_clipping import clip_sensor_ranges
from holosoma.utils.safe_torch_import import torch


class LidarReturnRecord(Protocol):
    """Minimal sensor-record surface shared by LiDAR backend adapters."""

    config: LidarSensorConfig

    def set_buffer(self, output: str, tensor: torch.Tensor) -> None:
        """Store one output tensor for the current scan."""


def points_from_ranges(directions: torch.Tensor, ranges: torch.Tensor) -> torch.Tensor:
    """Convert reported radial ranges into sensor-frame XYZ, preserving configured sentinels."""
    points = ranges.unsqueeze(-1) * directions.unsqueeze(0)
    return torch.where(
        torch.isfinite(ranges).unsqueeze(-1),
        points,
        torch.full_like(points, float("nan")),
    ).to(torch.float32)


def finalize_lidar_returns(
    sensor_record: LidarReturnRecord,
    directions: torch.Tensor,
    distances: torch.Tensor,
    *,
    geom_ids: torch.Tensor | None = None,
) -> None:
    """Apply range clipping and store backend-neutral output buffers for one LiDAR scan."""
    result = clip_sensor_ranges(
        distances,
        near=sensor_record.config.near,
        far=sensor_record.config.far,
        behavior=sensor_record.config.range_clipping_behavior,
    )
    sensor_record.set_buffer("ranges", result.ranges)
    sensor_record.set_buffer("points", points_from_ranges(directions, result.ranges))
    if geom_ids is not None:
        sensor_record.set_buffer(
            "geom_ids",
            torch.where(result.hit_mask, geom_ids, torch.full_like(geom_ids, -1)).to(torch.int32),
        )
