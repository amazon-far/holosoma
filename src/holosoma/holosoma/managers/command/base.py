"""Base classes for command manager terms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor

    from holosoma.config_types.command import CommandTermCfg
    from holosoma.managers.command.manager import CommandManager


class CommandTermBase(ABC):
    """Base class for stateful command terms.

    Command terms can keep internal state and execute logic during the
    environment lifecycle. Subclasses should implement all lifecycle hooks.
    """

    def __init__(self, cfg: CommandTermCfg, env: Any):
        self.cfg = cfg
        self.env = env
        self.manager: CommandManager | None = None

    @abstractmethod
    def setup(self) -> None:
        """Setup hook called once during environment initialization."""

    @abstractmethod
    def reset(self, env_ids: Tensor | None) -> None:
        """Reset hook called whenever environments reset.

        Args:
            env_ids: Tensor of environment ids to reset, or None to reset all.
        """

    @abstractmethod
    def step(self) -> None:
        """Per-step hook called during simulation rollout."""
