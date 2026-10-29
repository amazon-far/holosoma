"""Randomization terms for robot-mounted sensors."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from holosoma.utils.sampler import DistributionLike, TermSampler


def setup_raycast_depth_camera_mount(
    env: Any,
    sampler: TermSampler,
    *,
    sensor: str,
    position_range: Sequence[DistributionLike],
    euler_deg_range: Sequence[DistributionLike],
    enabled: bool = False,
    **_: Any,
) -> None:
    """Perturb a ray-cast depth camera's mount per env, once at startup.

    ``position_range`` and ``euler_deg_range`` give one config range value (a ``[lo, hi]`` pair or a
    spec dict) per axis: ``[x, y, z]`` position offsets in meters and XYZ Euler offsets in degrees, in
    the mount-link frame (see ``RayCastDepthCamera.set_mount_offset``). ``sensor`` is the camera's key in
    the ``--sensor`` dict.
    """
    if not enabled:
        return
    if len(position_range) != 3 or len(euler_deg_range) != 3:
        raise ValueError(
            f"setup_raycast_depth_camera_mount needs one range per axis, got {len(position_range)} position and "
            f"{len(euler_deg_range)} rotation ranges."
        )
    camera = getattr(env.simulator, "raycast_cameras", {}).get(sensor)
    if camera is None:
        raise ValueError(
            f"setup_raycast_depth_camera_mount: no ray-cast depth camera named '{sensor}'. It needs the "
            f"Isaac Sim simulator and a camera with isaacsim.renderer='raycast'; other renderers and "
            f"backends don't support mount randomization."
        )

    # Six decorrelated per-env streams: position x/y/z are 0-2, rotations about x/y/z are 3-5.
    env_ids = torch.arange(env.num_envs, device="cpu")
    position = torch.stack(
        [sampler.draw(r, env_ids=env_ids, coords=(axis,), device=env.device) for axis, r in enumerate(position_range)],
        dim=-1,
    )
    euler_deg = torch.stack(
        [
            sampler.draw(r, env_ids=env_ids, coords=(3 + axis,), device=env.device)
            for axis, r in enumerate(euler_deg_range)
        ],
        dim=-1,
    )
    camera.set_mount_offset(position, torch.deg2rad(euler_deg))
