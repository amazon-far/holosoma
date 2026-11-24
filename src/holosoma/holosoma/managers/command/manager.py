from __future__ import annotations

from typing import TYPE_CHECKING, Any

from omegaconf import DictConfig, OmegaConf

from holosoma.config_types.command import CommandManagerCfg
from holosoma.managers.staged_terms import StagedTerms

from .base import CommandTermBase

if TYPE_CHECKING:
    from torch import Tensor


def _resolve_command_cfg(command_params: dict[str, Any] | DictConfig | None) -> DictConfig:
    """Create a DictConfig wrapper around the command parameters.

    Parameters
    ----------
    command_params : dict[str, Any] or DictConfig or None
        Raw parameter mapping or existing DictConfig describing command term
        settings.

    Returns
    -------
    DictConfig
        Configuration object that supports attribute-style access.
    """
    if isinstance(command_params, DictConfig):
        return command_params
    return OmegaConf.create(command_params or {})


class CommandManager:
    """Coordinate command-related hooks across setup, reset, and step phases.

    The command manager evaluates the configured command terms during the
    environment lifecycle. Terms can be simple functions or subclasses of
    :class:`CommandTermBase` that expose ``setup``/``reset``/``step`` hooks.
    This keeps command sampling and gait state management centralized and
    ensures environments do not own redundant buffers.

    Parameters
    ----------
    cfg : CommandManagerCfg
        Command manager configuration specifying the term registry.
    env : Any
        Environment instance (usually a ``BaseTask`` subclass).
    device : str
        Device identifier used for tensors created by the terms.
    """

    def __init__(self, cfg: CommandManagerCfg, env: Any, device: str):
        self.cfg = cfg
        self.env = env
        self.device = device
        self.logger = getattr(env, "logger", None)
        self.command_cfg = _resolve_command_cfg(getattr(self.cfg, "params", None))

        self.staged_terms = StagedTerms(cfg, env, term_base=CommandTermBase, context="command term", owner=self)
        # Read by downstream code; keep the name.
        self._state_terms = self.staged_terms.instances

    def setup(self) -> None:
        self.staged_terms.run("setup", self.env)

    def reset(self, env_ids: Tensor | None) -> None:
        self.staged_terms.run("reset", self.env, env_ids)

    def step(self) -> None:
        self.staged_terms.run("step", self.env)

    def get_state(self, term_name: str) -> CommandTermBase | None:
        """Retrieve a stateful command term by name.

        Parameters
        ----------
        term_name : str
            Name of the command term.

        Returns
        -------
        CommandTermBase or None
            Stateful command term instance if it exists, otherwise ``None``.
        """
        return self._state_terms.get(term_name)

    @property
    def commands(self) -> Tensor:
        """Expose the locomotion command buffer owned by the locomotion command term.

        Returns
        -------
        torch.Tensor
            Reference to the locomotion command tensor maintained by the term.
        """
        state = self.get_state("locomotion_command")
        if state is None or not hasattr(state, "commands") or state.commands is None:
            raise AttributeError("Locomotion command state is not available on the command manager.")
        return state.commands
