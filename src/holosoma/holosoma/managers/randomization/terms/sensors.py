"""Randomization terms for robot-mounted sensors."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from holosoma.managers.randomization.exceptions import RandomizerNotSupportedError
from holosoma.utils.rotations import quat_from_euler_xyz, quat_mul
from holosoma.utils.sampler import DistributionLike, TermSampler


def setup_camera_mount(
    env: Any,
    sampler: TermSampler,
    *,
    sensor: str,
    position_range: Sequence[DistributionLike],
    euler_deg_range: Sequence[DistributionLike],
    enabled: bool = False,
    **_: Any,
) -> None:
    """Perturb a camera's mount per env, once at startup, through ``SensorManager.set_mount_pose``.

    ``position_range`` and ``euler_deg_range`` give one config range value (a ``[lo, hi]`` pair or a
    spec dict) per axis: ``[x, y, z]`` position offsets in meters and XYZ Euler offsets in degrees, in
    the mount frame. The position offset is added to ``mount.position`` and the rotation is applied
    about the mount frame's axes before ``mount.orientation``. ``sensor`` is the camera's key in the
    ``--sensor`` dict.
    """
    if not enabled:
        return
    if len(position_range) != 3 or len(euler_deg_range) != 3:
        raise ValueError(
            f"setup_camera_mount needs one range per axis, got {len(position_range)} position and "
            f"{len(euler_deg_range)} rotation ranges."
        )
    manager = env.simulator.sensor_manager
    try:
        mount = manager.get_sensor(sensor).config.mount
    except KeyError as exc:
        raise ValueError(f"setup_camera_mount: no sensor named '{sensor}'.") from exc

    # Six decorrelated per-env streams: position x/y/z are 0-2, rotations about x/y/z are 3-5.
    env_ids = torch.arange(env.num_envs, device="cpu")
    offset = torch.stack(
        [sampler.draw(r, env_ids=env_ids, coords=(axis,), device=env.device) for axis, r in enumerate(position_range)],
        dim=-1,
    )
    euler = torch.deg2rad(
        torch.stack(
            [
                sampler.draw(r, env_ids=env_ids, coords=(3 + axis,), device=env.device)
                for axis, r in enumerate(euler_deg_range)
            ],
            dim=-1,
        )
    )
    w, x, y, z = mount.orientation
    base_quat = torch.tensor([x, y, z, w], dtype=torch.float32, device=env.device).expand(env.num_envs, 4)
    position = torch.tensor(mount.position, dtype=torch.float32, device=env.device) + offset
    orientation = quat_mul(
        quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2]), base_quat.contiguous(), w_last=True
    )
    try:
        manager.set_mount_pose(sensor, torch.cat([position, orientation], dim=-1))
    except NotImplementedError as exc:
        raise RandomizerNotSupportedError(
            f"setup_camera_mount does not support sensor '{sensor}' on {type(env.simulator).__name__}: {exc}"
        ) from exc
