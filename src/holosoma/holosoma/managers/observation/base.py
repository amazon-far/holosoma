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

    The manager calls a term once per frame for all envs, and may additionally call it for the
    envs about to reset to observe their terminal state.
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
    def __call__(self, env: Any, env_ids: torch.Tensor | None = None, **kwargs: Any) -> torch.Tensor:
        """Compute the observation of ``env_ids`` and advance only their internal state.

        Envs not in ``env_ids`` must keep their state untouched. An empty ``env_ids`` is how the
        manager measures the output shape: return ``[0, *obs_shape]`` and change nothing.

        Args:
            env: Environment instance
            env_ids: Environment IDs to compute and advance. If None, all environments.
            **kwargs: Additional parameters from config

        Returns:
            Observation tensor of shape [num_envs, *obs_shape], or [len(env_ids), *obs_shape]
        """
