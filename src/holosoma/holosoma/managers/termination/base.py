"""Base classes and protocols for termination terms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from holosoma.utils.safe_torch_import import torch

if TYPE_CHECKING:
    from holosoma.config_types.termination import TerminationTermCfg


class TerminationTermBase(ABC):
    """Base class for stateful termination terms."""

    def __init__(self, cfg: TerminationTermCfg, env: Any) -> None:
        self.cfg = cfg
        self.env = env

    @abstractmethod
    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset internal state for specified environments."""

    @abstractmethod
    def __call__(self, env: Any, **kwargs: Any) -> torch.Tensor:
        """Evaluate termination condition."""
