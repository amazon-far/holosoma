"""Common termination term helpers."""

from __future__ import annotations

from typing import Any

from holosoma.utils.safe_torch_import import torch


def timeout_exceeded(env: Any, **_: Any) -> torch.Tensor:
    """Terminate environments that exceeded the maximum episode length."""
    return env.episode_length_buf > env.max_episode_length
