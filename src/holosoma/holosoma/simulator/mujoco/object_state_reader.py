"""MJWarp object-root reads using Holosoma's setup-time actor addressing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import torch

from holosoma.simulator.mujoco.backends.base import mj_to_holosoma_quat
from holosoma.utils.rotations import quat_rotate


@dataclass(frozen=True)
class ObjectReadPlan:
    """Fixed object addressing and reusable scratch; :meth:`read` returns the same output tensor each call.

    Rebuild the simulator for topology changes.
    """

    qpos: torch.Tensor
    qvel: torch.Tensor
    xpos: torch.Tensor
    xquat: torch.Tensor
    free_columns: torch.Tensor
    static_columns: torch.Tensor
    qpos_columns: torch.Tensor
    qvel_columns: torch.Tensor
    static_body_ids: torch.Tensor
    free_pose: torch.Tensor
    free_velocity: torch.Tensor
    free_states: torch.Tensor
    static_states: torch.Tensor
    output: torch.Tensor

    @classmethod
    def build(
        cls,
        names: tuple[str, ...],
        object_addrs: Mapping[str, Mapping[str, int]],
        static_object_body_ids: Mapping[str, int],
        qpos: torch.Tensor,
        qvel: torch.Tensor,
        xpos: torch.Tensor,
        xquat: torch.Tensor,
    ) -> ObjectReadPlan:
        if len(set(names)) != len(names) or "robot" in names:
            raise ValueError("ObjectReadPlan requires unique non-robot actor names")
        envs = qpos.shape[0]
        if (
            qpos.ndim != 2
            or qvel.ndim != 2
            or xpos.ndim != 3
            or xquat.ndim != 3
            or qvel.shape[0] != envs
            or xpos.shape[0] != envs
            or xquat.shape != (*xpos.shape[:2], 4)
            or xpos.shape[2] != 3
        ):
            raise ValueError("Inconsistent batched native object-state shapes")
        if any(t.device != qpos.device or t.dtype != torch.float32 for t in (qpos, qvel, xpos, xquat)):
            raise ValueError("Native object-state arrays must share float32 device storage")

        free_columns: list[int] = []
        static_columns: list[int] = []
        qpos_columns: list[int] = []
        qvel_columns: list[int] = []
        body_ids: list[int] = []
        for column, name in enumerate(names):
            if name in static_object_body_ids:
                body_id = static_object_body_ids[name]
                if not 0 <= body_id < xpos.shape[1]:
                    raise ValueError(f"Invalid static body ID for {name}: {body_id}")
                static_columns.append(column)
                body_ids.append(body_id)
            else:
                address = object_addrs[name]
                pos, vel = address["qpos_addr"], address["qvel_addr"]
                if not (0 <= pos <= qpos.shape[1] - 7 and 0 <= vel <= qvel.shape[1] - 6):
                    raise ValueError(f"Invalid freejoint addressing for {name}")
                free_columns.append(column)
                qpos_columns.extend(range(pos, pos + 7))
                qvel_columns.extend(range(vel, vel + 6))

        def indices(values: list[int]) -> torch.Tensor:
            return torch.tensor(values, dtype=torch.long, device=qpos.device)

        free_count, static_count = len(free_columns), len(static_columns)
        # Static rows' velocity columns are zeroed once and never written.
        return cls(
            qpos,
            qvel,
            xpos,
            xquat,
            indices(free_columns),
            indices(static_columns),
            indices(qpos_columns),
            indices(qvel_columns),
            indices(body_ids),
            qpos.new_empty(envs, free_count * 7),
            qvel.new_empty(envs, free_count * 6),
            qpos.new_empty(envs, free_count, 13),
            qpos.new_zeros(envs, static_count, 13),
            qpos.new_empty(envs, len(names), 13),
        )

    def read(self) -> torch.Tensor:
        """Return ``[E, O, 13]`` object roots in the per-actor getter's world-frame convention.

        Jointless objects use derived world body poses, which the caller must have brought up to date.
        """
        envs = self.qpos.shape[0]
        free_count = self.free_columns.numel()
        if free_count:
            torch.index_select(self.qpos, 1, self.qpos_columns, out=self.free_pose)
            torch.index_select(self.qvel, 1, self.qvel_columns, out=self.free_velocity)
            pose = self.free_pose.view(envs, free_count, 7)
            velocity = self.free_velocity.view(envs, free_count, 6)
            quaternion = cast("torch.Tensor", mj_to_holosoma_quat(pose[..., 3:7]))
            # Native angular velocity is body-local; unified state is world-frame.
            self.free_states[..., :3].copy_(pose[..., :3])
            self.free_states[..., 3:7].copy_(quaternion)
            self.free_states[..., 7:10].copy_(velocity[..., :3])
            self.free_states[..., 10:13].copy_(
                quat_rotate(
                    quaternion.reshape(-1, 4),
                    velocity[..., 3:6].reshape(-1, 3),
                    w_last=True,
                ).view(envs, free_count, 3)
            )
            self.output.index_copy_(1, self.free_columns, self.free_states)
        if self.static_columns.numel():
            self.static_states[..., :3].copy_(torch.index_select(self.xpos, 1, self.static_body_ids))
            self.static_states[..., 3:7].copy_(
                cast("torch.Tensor", mj_to_holosoma_quat(torch.index_select(self.xquat, 1, self.static_body_ids)))
            )
            self.output.index_copy_(1, self.static_columns, self.static_states)
        return self.output
