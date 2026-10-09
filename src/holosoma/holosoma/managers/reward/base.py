"""Base classes and protocols for reward terms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from holosoma.utils.safe_torch_import import torch

if TYPE_CHECKING:
    from holosoma.config_types.reward import RewardTermCfg


class RewardTermBase(ABC):
    """Base class for stateful reward terms.

    This class provides the interface for reward terms that need to maintain
    internal state (e.g., tracking metrics over time, curriculum values).
    For simple stateless rewards, use plain functions instead.

    The reward term is responsible for computing a raw reward value (before
    weighting and dt scaling, which is handled by the manager).
    """

    def __init__(self, cfg: RewardTermCfg, env: Any):
        """Initialize reward term.

        Args:
            cfg: Configuration for this reward term
            env: Environment instance (typically a ``BaseTask`` subclass)
        """
        self.cfg = cfg
        self.env = env

    @abstractmethod
    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset internal state for specified environments.

        Args:
            env_ids: Environment IDs to reset. If None, reset all environments.
        """

    @abstractmethod
    def __call__(self, env: Any, **kwargs: Any) -> torch.Tensor:
        """Compute reward.

        Args:
            env: Environment instance
            **kwargs: Additional parameters from config

        Returns:
            Reward tensor of shape [num_envs] (raw, unweighted value)
        """
