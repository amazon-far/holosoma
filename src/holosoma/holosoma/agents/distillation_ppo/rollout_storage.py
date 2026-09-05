"""Rollout storage for :class:`DistillationPPO`.

Thin wrapper over :class:`holosoma.agents.modules.data_utils.RolloutStorage`
that pre-registers the extra fields required by the student/teacher/critic
distillation pipeline:

- ``teacher_obs``: privileged observation consumed by the frozen teacher MLP.
- ``depth_obs``: raw depth image (H, W) consumed by the student's CNN backbone.
- ``teacher_actions``: DAgger targets produced by ``policy.teacher_act``.

All other fields (``actor_obs``, ``critic_obs``, ``actions``, ``rewards``,
``dones``, ``values``, ``returns``, ``advantages``, ``actions_log_prob``,
``action_mean``, ``action_sigma``) are registered exactly as the PPO rollout
storage does, so the :class:`Minibatch`-like generator output stays compatible.
"""

from __future__ import annotations

import torch

from holosoma.agents.modules.data_utils import RolloutStorage


class RolloutStorageDistillation(RolloutStorage):
    """Rollout storage that pre-registers all fields the distillation update
    needs.

    Parameters
    ----------
    num_envs : int
        Number of parallel environments.
    num_transitions_per_env : int
        Number of rollout steps collected per environment between updates.
    actor_obs_dim : int
        Flat dimension of the student's proprio observation.
    teacher_obs_dim : int
        Flat dimension of the teacher's privileged observation.
    critic_obs_dim : int
        Flat dimension of the critic's privileged observation.
    depth_shape : tuple[int, ...]
        Spatial shape of the depth image (e.g. ``(H, W)``).
    num_actions : int
        Action dimension.
    device : str
        Device to allocate buffers on.
    """

    def __init__(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        actor_obs_dim: int,
        teacher_obs_dim: int,
        critic_obs_dim: int,
        depth_shape: tuple[int, ...],
        num_actions: int,
        device: str = "cpu",
    ):
        super().__init__(num_envs=num_envs, num_transitions_per_env=num_transitions_per_env, device=device)

        self.actor_obs_dim = actor_obs_dim
        self.teacher_obs_dim = teacher_obs_dim
        self.critic_obs_dim = critic_obs_dim
        self.depth_shape = tuple(depth_shape)
        self.num_actions = num_actions

        # Register all per-step buffers the DistillationPPO update() loop reads.
        self.register("actor_obs", shape=(actor_obs_dim,), dtype=torch.float)
        self.register("teacher_obs", shape=(teacher_obs_dim,), dtype=torch.float)
        self.register("critic_obs", shape=(critic_obs_dim,), dtype=torch.float)
        self.register("depth_obs", shape=self.depth_shape, dtype=torch.float)

        self.register("actions", shape=(num_actions,), dtype=torch.float)
        self.register("teacher_actions", shape=(num_actions,), dtype=torch.float)

        # 1-D scalar fields stored as (1,) to match holosoma PPO's conventions.
        self.register("rewards", shape=(1,), dtype=torch.float)
        self.register("dones", shape=(1,), dtype=torch.bool)
        self.register("values", shape=(1,), dtype=torch.float)
        self.register("returns", shape=(1,), dtype=torch.float)
        self.register("advantages", shape=(1,), dtype=torch.float)
        self.register("actions_log_prob", shape=(1,), dtype=torch.float)

        self.register("action_mean", shape=(num_actions,), dtype=torch.float)
        self.register("action_sigma", shape=(num_actions,), dtype=torch.float)

        # Per-sample PPO gate in [0, 1]: how much of the PPO/DAgger split this
        # transition is subject to. 1.0 reproduces the scalar-``ppo_coef``
        # behaviour exactly; 0.0 makes the transition pure DAgger. Filled by
        # ``DistillationPPO._get_ppo_gate``, which defaults to all-ones, so this
        # is inert unless an application overrides it.
        self.register("ppo_gate", shape=(1,), dtype=torch.float)

    def add(self, **data):
        """Strict variant of :meth:`RolloutStorage.add`.

        Base ``RolloutStorage.add`` silently drops any key that isn't pre-registered.
        For distillation training we rely on a specific key set (``teacher_obs``,
        ``depth_obs``, ``teacher_actions``, …), and a typo like ``action_means``
        instead of ``action_mean`` would silently zero out a buffer that the
        update loop later reads — leading to a plausible-looking but wrong
        gradient. Raise on unknown keys instead.
        """
        unknown = [k for k in data if k not in self._buffers]
        if unknown:
            known = sorted(self._buffers.keys())
            raise KeyError(
                f"RolloutStorageDistillation.add() got unknown keys {unknown}. "
                f"Registered buffers: {known}."
            )
        super().add(**data)
