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

    The observation manager reads a term in one of three ways:

    - :meth:`__call__` once per control step, for the observations the policy acts on. It may
      advance internal state (shift a buffer, sample noise).
    - :meth:`peek` for any other read of the current state, such as the final observations of
      envs about to reset. It must have no side effects: no state change and no RNG draws.
    - :meth:`output_shape` to size buffers and networks. It must not advance internal state
      and may skip computing the observation.

    A term whose ``__call__`` advances state must override :meth:`peek`.
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
        """Compute this control step's observation, advancing internal state if any.

        Args:
            env: Environment instance
            **kwargs: Additional parameters from config

        Returns:
            Observation tensor of shape [num_envs, obs_dim]
        """

    def peek(self, env: Any, **kwargs: Any) -> torch.Tensor:
        """Compute the observation of the current state, consistent with the last step read, without side effects.

        A peek changes no internal state and draws no random numbers, so any number of peeks
        leaves the next :meth:`__call__` unaffected. A term whose step read samples randomness
        (noise, augmentation, a resampled delay) cannot reproduce it under that rule; its peek
        may leave that randomness out, and the term documents what it omits.

        The default calls :meth:`__call__`, which is correct only for terms whose
        ``__call__`` has no side effects. Terms that advance state per call must override it.

        Args:
            env: Environment instance
            **kwargs: Additional parameters from config

        Returns:
            Observation tensor of shape [num_envs, ...], the same shape :meth:`__call__` returns
        """
        return self(env, **kwargs)

    def output_shape(self, env: Any, **kwargs: Any) -> tuple[int, ...]:
        """Shape of one env's observation, i.e. the output shape without the env axis.

        The default reads it from :meth:`peek`. Override to report the shape without
        computing an observation.

        Args:
            env: Environment instance
            **kwargs: Additional parameters from config
        """
        return tuple(self.peek(env, **kwargs).shape[1:])
