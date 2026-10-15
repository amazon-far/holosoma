"""Rollout storage for :class:`Distillation` (pure DAgger).

Parallels :class:`holosoma.agents.distillation_ppo.rollout_storage.RolloutStorageDistillation`
but drops every PPO-only field: no ``critic_obs``, no ``values``, ``returns``,
``advantages``, ``actions_log_prob``, ``action_mean``, ``action_sigma``, and
no ``rewards``. Pure behavior-cloning needs only student/teacher obs, the
depth image (for a depth student), the (student) action taken, the (teacher)
action label, and the ``dones`` flag for episode resets.
"""

from __future__ import annotations

from typing import Iterator

import torch

from holosoma.agents.modules.data_utils import RolloutStorage


class RolloutStorageDagger(RolloutStorage):
    """Storage for pure-DAgger distillation.

    Parameters
    ----------
    num_envs : int
    num_transitions_per_env : int
    actor_obs_dim : int
        Flat dimension of the student's proprio observation.
    teacher_obs_dim : int
        Flat dimension of the teacher's privileged observation.
    depth_shape : tuple[int, ...] | None
        Spatial shape of the depth image (e.g. ``(H, W)``), or ``None`` for a
        student without depth input.
    num_actions : int
    device : str
    """

    def __init__(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        actor_obs_dim: int,
        teacher_obs_dim: int,
        depth_shape: tuple[int, ...] | None,
        num_actions: int,
        device: str = "cpu",
    ):
        super().__init__(num_envs=num_envs, num_transitions_per_env=num_transitions_per_env, device=device)

        self.actor_obs_dim = actor_obs_dim
        self.teacher_obs_dim = teacher_obs_dim
        self.depth_shape = tuple(depth_shape) if depth_shape is not None else None
        self.num_actions = num_actions

        self.register("actor_obs", shape=(actor_obs_dim,), dtype=torch.float)
        self.register("teacher_obs", shape=(teacher_obs_dim,), dtype=torch.float)
        if self.depth_shape is not None:
            self.register("depth_obs", shape=self.depth_shape, dtype=torch.float)
        self.register("actions", shape=(num_actions,), dtype=torch.float)
        self.register("teacher_actions", shape=(num_actions,), dtype=torch.float)
        self.register("teacher_valid", shape=(1,), dtype=torch.bool)
        self.register("dones", shape=(1,), dtype=torch.bool)

    def add(self, **data: torch.Tensor) -> None:
        """Strict add — raise on unknown keys (see the PPO-version docstring).

        Silent key-drop on typos is too easy to miss; this guards against e.g.
        ``action_means`` instead of ``actions`` going to /dev/null.
        """
        unknown = [k for k in data if k not in self._buffers]
        if unknown:
            known = sorted(self._buffers.keys())
            raise KeyError(f"RolloutStorageDagger.add() got unknown keys {unknown}. Registered buffers: {known}.")
        super().add(**data)

    def mini_batch_generator(
        self, num_mini_batches: int, num_epochs: int = 8, reshuffle_each_epoch: bool = True
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Same as :meth:`RolloutStorage.mini_batch_generator`, but reshuffles every
        epoch by default, as the reference distillation implementation does."""
        return super().mini_batch_generator(num_mini_batches, num_epochs, reshuffle_each_epoch=reshuffle_each_epoch)
