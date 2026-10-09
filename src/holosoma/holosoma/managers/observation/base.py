"""Base classes and protocols for observation terms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from holosoma.utils.safe_torch_import import torch

if TYPE_CHECKING:
    from holosoma.config_types.observation import ObsTermCfg


class ObservationTermBase(ABC):
    """Base class for stateful observation terms.

    This class provides the interface for observation terms that need to maintain
    internal state (e.g., history buffers, filters). For simple stateless observations,
    use plain functions instead.

    Note: Currently not used in basic locomotion implementation, but provided for
    future extensibility.
    """

    def __init__(self, cfg: ObsTermCfg, env: Any):
        """Initialize observation term.

        Args:
            cfg: Configuration for this observation term
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
        """Compute observation.

        Args:
            env: Environment instance
            **kwargs: Additional parameters from config

        Returns:
            Observation tensor of shape [num_envs, obs_dim]
        """
