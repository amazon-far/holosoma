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

    def set_buffer_rows(self, output: str, env_ids: torch.Tensor, rows: torch.Tensor) -> None:
        """Overwrite the ``env_ids`` rows of one existing output."""


def points_from_ranges(directions: torch.Tensor, ranges: torch.Tensor) -> torch.Tensor:
    """Convert reported radial ranges into sensor-frame XYZ, preserving configured sentinels."""
    points = ranges.unsqueeze(-1) * directions.unsqueeze(0)
    # Scalar sentinel: a full_like fill would allocate and write a second [..., 3] buffer.
    return torch.where(torch.isfinite(ranges).unsqueeze(-1), points, float("nan")).to(torch.float32)


def finalize_lidar_returns(
    sensor_record: LidarReturnRecord,
    directions: torch.Tensor,
    distances: torch.Tensor,
    *,
    geom_ids: torch.Tensor | None = None,
    env_ids: torch.Tensor | None = None,
) -> None:
    """Apply range clipping and store backend-neutral output buffers for one LiDAR scan.

    With ``env_ids``, ``distances`` and ``geom_ids`` hold only those envs' rows, which overwrite the
    matching rows of the existing buffers."""
    result = clip_sensor_ranges(
        distances,
        near=sensor_record.config.near,
        far=sensor_record.config.far,
        behavior=sensor_record.config.range_clipping_behavior,
    )
    outputs = {"ranges": result.ranges, "points": points_from_ranges(directions, result.ranges)}
    if geom_ids is not None:
        outputs["geom_ids"] = torch.where(result.hit_mask, geom_ids, -1).to(torch.int32)
    for output, tensor in outputs.items():
        if env_ids is None:
            sensor_record.set_buffer(output, tensor)
        else:
            sensor_record.set_buffer_rows(output, env_ids, tensor)
