from __future__ import annotations

import importlib
from typing import Any

from omegaconf import DictConfig, OmegaConf

from holosoma.config_types.command import CommandManagerCfg, CommandTermCfg

from .base import CommandTermBase


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

#负责协调与命令相关的钩子在环境生命周期中的设置、重置和步骤阶段。
# 命令管理器在环境生命周期中评估配置的命令术语。术语可以是简单的函数或CommandTermBase的子类，后者公开setup/reset/step钩子。这保持了命令采样和步态状态管理的集中，并确保环境不拥有冗余的缓冲区。
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

        # Storage for resolved stateless functions
        self._setup_funcs: dict[str, Any] = {}
        self._reset_funcs: dict[str, Any] = {}
        self._step_funcs: dict[str, Any] = {}

        # Storage for stateful class-based hooks and their lifecycle bindings
        self._class_entries: list[dict[str, Any]] = []
        self._state_terms: dict[str, CommandTermBase] = {}

        # Storage for term names and configs (preserve original ordering)
        self._setup_names: list[str] = []
        self._reset_names: list[str] = []
        self._step_names: list[str] = []

        self._setup_cfgs: list[CommandTermCfg] = []
        self._reset_cfgs: list[CommandTermCfg] = []
        self._step_cfgs: list[CommandTermCfg] = []

        # Initialize terms
        self._initialize_terms()

    def _initialize_terms(self) -> None:
        """Resolve command term implementations and register their lifecycle hooks."""
        for term_name, term_cfg in self.cfg.setup_terms.items():
            resolved = self._resolve_function(term_cfg.func)
            if isinstance(resolved, type):
                if not issubclass(resolved, CommandTermBase):
                    raise TypeError(f"Class-based command term '{term_name}' must inherit from CommandTermBase.")
                self._register_class_term(term_name, resolved, term_cfg, "setup")
            else:
                self._setup_funcs[term_name] = resolved
            self._setup_names.append(term_name)
            self._setup_cfgs.append(term_cfg)

        for term_name, term_cfg in self.cfg.reset_terms.items():
            resolved = self._resolve_function(term_cfg.func)
            if isinstance(resolved, type):
                if not issubclass(resolved, CommandTermBase):
                    raise TypeError(f"Class-based command term '{term_name}' must inherit from CommandTermBase.")
                self._register_class_term(term_name, resolved, term_cfg, "reset")
            else:
                self._reset_funcs[term_name] = resolved
            self._reset_names.append(term_name)
            self._reset_cfgs.append(term_cfg)

        for term_name, term_cfg in self.cfg.step_terms.items():
            resolved = self._resolve_function(term_cfg.func)
            if isinstance(resolved, type):
                if not issubclass(resolved, CommandTermBase):
                    raise TypeError(f"Class-based command term '{term_name}' must inherit from CommandTermBase.")
                self._register_class_term(term_name, resolved, term_cfg, "step")
            else:
                self._step_funcs[term_name] = resolved
            self._step_names.append(term_name)
            self._step_cfgs.append(term_cfg)

    def _resolve_function(self, func: Any | str) -> Any:
        """Resolve a callable or class from a string specification.

        Parameters
        ----------
        func : Any or str
            Function reference or string path like ``"module:function"``.

        Returns
        -------
        Any
            Resolved callable or class.

        Raises
        ------
        ValueError
            If the string path is malformed or the target cannot be imported.
        """
        if isinstance(func, str):
            if ":" not in func:
                raise ValueError(f"Function string must be in format 'module:function', got: {func}")
            module_path, func_name = func.split(":", 1)
            try:
                module = importlib.import_module(module_path)
                return getattr(module, func_name)
            except (ImportError, AttributeError) as exc:
                raise ValueError(f"Failed to import command function '{func}': {exc}") from exc
        return func

    def setup(self) -> None:
        for entry in self._class_entries:
            if "setup" in entry["stages"]:
                entry["instance"].setup()

        for term_name, term_cfg in zip(self._setup_names, self._setup_cfgs):
            if term_name in self._setup_funcs:
                func = self._setup_funcs[term_name]
                func(self.env, **term_cfg.params)

    def reset(self, env_ids) -> None:
        for entry in self._class_entries:
            if "reset" in entry["stages"]:
                entry["instance"].reset(env_ids)

        for term_name, term_cfg in zip(self._reset_names, self._reset_cfgs):
            if term_name in self._reset_funcs:
                func = self._reset_funcs[term_name]
                func(self.env, env_ids, **term_cfg.params)

    def step(self) -> None:
        for entry in self._class_entries:
            if "step" in entry["stages"]:
                entry["instance"].step()

        for term_name, term_cfg in zip(self._step_names, self._step_cfgs):
            if term_name in self._step_funcs:
                func = self._step_funcs[term_name]
                func(self.env, **term_cfg.params)

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
    def commands(self):
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

    def _register_class_term(
        self, term_name: str, term_class: type[CommandTermBase], term_cfg: CommandTermCfg, stage: str
    ) -> None:
        """Instantiate or update a class-based command term for the given lifecycle stage.

        Parameters
        ----------
        term_name : str
            Registry name of the command term.
        term_class : type[CommandTermBase]
            Command term implementation class.
        term_cfg : CommandTermCfg
            Configuration for the command term.
        stage : str
            Lifecycle stage being registered (``"setup"``, ``"reset"``, or ``"step"``).
        """
        for entry in self._class_entries:
            if entry["name"] == term_name and isinstance(entry["instance"], term_class):
                entry["stages"].add(stage)
                self._state_terms.setdefault(term_name, entry["instance"])
                return

        instance = term_class(term_cfg, self.env)
        instance.manager = self
        self._class_entries.append({"name": term_name, "instance": instance, "stages": {stage}})
        self._state_terms[term_name] = instance
