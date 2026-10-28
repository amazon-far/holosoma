"""Generalized advantage estimation shared by the on-policy agents."""

from __future__ import annotations

from holosoma.utils.safe_torch_import import torch


def compute_gae_returns(
    last_values: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    rewards: torch.Tensor,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    """GAE returns for a ``[num_steps, num_envs, 1]`` rollout; advantages are ``returns - values``."""
    advantage: torch.Tensor | float = 0.0
    returns = torch.zeros_like(values)
    num_steps = returns.shape[0]
    for step in reversed(range(num_steps)):
        next_values = last_values if step == num_steps - 1 else values[step + 1]
        next_is_not_terminal = 1.0 - dones[step].float()
        delta = rewards[step] + next_is_not_terminal * gamma * next_values - values[step]
        advantage = delta + next_is_not_terminal * gamma * lam * advantage
        returns[step] = advantage + values[step]
    return returns


def normalize_advantages(advantages: torch.Tensor, world_size: int | None = None) -> torch.Tensor:
    """Standardize advantages to zero mean and unit std.

    With ``world_size``, the statistics are pooled over all ranks with an all-reduce. Each rank
    must hold the same number of samples, as every rank's full rollout does.
    """
    if world_size is None:
        return (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    local_stats = torch.stack([advantages.mean(), (advantages**2).mean()])
    torch.distributed.all_reduce(local_stats, op=torch.distributed.ReduceOp.SUM)
    global_mean = local_stats[0] / world_size
    global_sq_mean = local_stats[1] / world_size
    global_std = torch.sqrt(global_sq_mean - global_mean**2 + 1e-8)
    return (advantages - global_mean) / global_std
