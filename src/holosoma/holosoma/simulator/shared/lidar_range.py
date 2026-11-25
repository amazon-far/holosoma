# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""LiDAR output finalization and sensor-frame point conversion."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Protocol

from holosoma.config_types.sensor import LidarSensorConfig
from holosoma.simulator.shared.range_clipping import clip_sensor_ranges
from holosoma.utils.safe_torch_import import torch


class LidarReturnRecord(Protocol):
    """Minimal sensor-record surface shared by LiDAR backend adapters."""

    config: LidarSensorConfig
    buffers: dict[str, torch.Tensor]
    scan_directions: torch.Tensor | None

    def writing(self, output: str, env_ids: torch.Tensor | None = None) -> AbstractContextManager[torch.Tensor]:
        """Write one output's buffer in place for the current scan."""

    def set_buffer_rows(self, output: str, env_ids: torch.Tensor, rows: torch.Tensor) -> None:
        """Overwrite the ``env_ids`` rows of one existing output."""


def points_from_ranges(directions: torch.Tensor, ranges: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Convert reported radial ranges into sensor-frame XYZ, preserving configured sentinels; into ``out`` if given."""
    if out is None:
        out = torch.empty((*ranges.shape, 3), dtype=torch.float32, device=ranges.device)
    torch.mul(ranges.unsqueeze(-1), directions.unsqueeze(0), out=out)
    return out.masked_fill_(~torch.isfinite(ranges).unsqueeze(-1), float("nan"))


def finalize_lidar_returns(
    sensor_record: LidarReturnRecord,
    directions: torch.Tensor,
    distances: torch.Tensor,
    *,
    geom_ids: torch.Tensor | None = None,
    env_ids: torch.Tensor | None = None,
) -> None:
    """Apply range clipping and write the backend-neutral outputs of one LiDAR scan into their buffers.

    With ``env_ids``, ``distances`` and ``geom_ids`` hold only those envs' rows, which overwrite the
    matching rows of the buffers."""
    if (geom_ids is not None) != ("geom_ids" in sensor_record.buffers):
        raise ValueError("LiDAR geom_ids must be reported exactly when registered as an output")
    config = sensor_record.config
    with sensor_record.writing("ranges", env_ids) as ranges:
        clipped = clip_sensor_ranges(
            distances.to(ranges.dtype),
            near=config.near,
            far=config.far,
            behavior=config.range_clipping_behavior,
            out=ranges if env_ids is None else None,
        )
        if env_ids is not None:
            ranges[env_ids] = clipped.ranges
    with sensor_record.writing("points", env_ids) as points:
        if env_ids is None:
            points_from_ranges(directions, ranges, out=points)
        else:
            points[env_ids] = points_from_ranges(directions, clipped.ranges)
    if geom_ids is not None:
        with sensor_record.writing("geom_ids", env_ids) as ids:
            hit_ids = torch.where(clipped.hit_mask, geom_ids.to(torch.int32), ids.new_tensor(-1))
            if env_ids is None:
                ids.copy_(hit_ids)
            else:
                ids[env_ids] = hit_ids
    if env_ids is None:
        sensor_record.scan_directions = directions
