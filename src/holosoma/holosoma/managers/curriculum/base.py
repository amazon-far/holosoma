"""Base classes for curriculum terms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from holosoma.config_types.curriculum import CurriculumTermCfg


class CurriculumTermBase(ABC):
    """Base class for stateful curriculum terms.

    Curriculum terms can maintain state across episodes and adapt based on
    training progress (e.g., episode lengths, success rates). This could involve:
    - Scaling reward term weights over time
    - Adjusting environment parameters
    - Modifying observation/action ranges
    - Any adaptive curriculum logic

    Each curriculum term manages its own state and lifecycle hooks.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: Any):
        """Initialize curriculum term.

        Parameters
        ----------
        cfg : CurriculumTermCfg
            Configuration for this curriculum term
        env : Any
            Environment instance (typically BaseTask or subclass)
        """
        self.cfg = cfg
        self.env = env

    @abstractmethod
    def setup(self) -> None:
        """Setup hook called once during environment initialization.

        Override this to perform one-time setup operations like:
        - Initializing state variables
        - Storing initial environment parameters
        - Configuring adaptive schedules
        """

    @abstractmethod
    def reset(self, env_ids: Any) -> None:
        """Reset hook called when environments are reset.

        Override this to perform per-episode reset operations.

        Parameters
        ----------
        env_ids : torch.Tensor or array-like
            Environment IDs being reset
        """

    @abstractmethod
    def step(self) -> None:
        """Step hook called every simulation step.

        Override this to perform per-step curriculum updates like:
        - Clamping parameters
        - Monitoring progress
        - Triggering adaptive changes
        """
