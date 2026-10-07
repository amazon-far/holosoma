"""Common termination term helpers."""

from __future__ import annotations

from holosoma.utils.safe_torch_import import torch


def timeout_exceeded(env, **_) -> torch.Tensor:
    """Terminate environments that exceeded the maximum episode length."""
    # Match IsaacLab/FAR semantics: the episode times out on the first step that
    # reaches the configured limit, not one step later.
    return env.episode_length_buf >= env.max_episode_length
