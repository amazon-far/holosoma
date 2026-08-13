"""Base classes for randomization manager terms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor

    from holosoma.config_types.randomization import RandomizationTermCfg
    from holosoma.managers.randomization.manager import RandomizationManager
    from holosoma.utils.sampler import TermSampler


class RandomizationTermBase(ABC):
    """Base class for stateful randomization hooks.

    Subclass this to apply custom domain randomization during environment setup, reset, or step.
    Each lifecycle method receives a :class:`~holosoma.utils.sampler.TermSampler` bound by the
    manager to this term's stable name, lifecycle stage, and the per-env episode counter.

    Draw EVERY random value from the provided ``sampler``. Never use a raw generator (``torch.rand``,
    ``numpy.random``, the ``random`` module, etc.): only sampler draws are reproducible per
    (term, env, episode) and independent of the order or set of other active terms.
    """

    def __init__(self, cfg: RandomizationTermCfg, env: Any):
        self.cfg = cfg
        self.env = env
        self.manager: RandomizationManager | None = None

    @abstractmethod
    def setup(self, sampler: TermSampler) -> None:
        """Called once during environment setup."""

    @abstractmethod
    def reset(self, env_ids: Tensor | None, sampler: TermSampler) -> None:
        """Called when specific environments reset."""

    @abstractmethod
    def step(self, sampler: TermSampler) -> None:
        """Called every simulation step (if configured)."""
