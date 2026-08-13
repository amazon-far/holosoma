"""Small helpers shared by the randomization term modules (locomotion + objects)."""

from __future__ import annotations

from typing import Any, Sequence

import torch

# PhysX caps the number of unique materials per scene, so IsaacSim material DR fills a fixed bucket
# table and assigns each shape a bucket. 64 buckets keep the quantile staircase close to continuous.
_MATERIAL_NUM_BUCKETS = 64


def _ensure_env_ids_tensor(env: Any, env_ids: torch.Tensor | Sequence[int] | None) -> torch.Tensor:
    """Convert environment indices to a tensor on the correct device."""
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=env.device, dtype=torch.long)
    return torch.as_tensor(list(env_ids), device=env.device, dtype=torch.long)
