"""Base class for workflow tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import Self

    from holosoma.config_types.experiment import HolosomaConfig


class Task(ABC):
    """A workflow: parses its config from CLI tokens, then builds and drives its environment."""

    name: str
    """Registry name selected by ``task:<name>``."""

    def __init__(self, config: HolosomaConfig, args: list[str] | None = None) -> None:
        self.config = config
        self.args = list(args or [])
        """The final CLI tokens the config was parsed from, for the run's record."""

    @classmethod
    @abstractmethod
    def from_args(cls, args: list[str]) -> Self:
        """Parse ``args`` (CLI tokens without the ``task:`` selection) into a task."""

    @abstractmethod
    def run(self) -> None:
        """Run the workflow."""

    @classmethod
    def with_selection(cls, args: list[str]) -> list[str]:
        """``args`` with ``task:<name>`` added, after a leading ``exp:`` root selection if any."""
        if args and args[0].startswith("exp:"):
            return [args[0], f"task:{cls.name}", *args[1:]]
        return [f"task:{cls.name}", *args]
