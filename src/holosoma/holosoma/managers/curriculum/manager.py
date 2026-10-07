"""Curriculum manager coordinating curriculum hooks."""

from __future__ import annotations

from typing import Any, ItemsView

from holosoma.config_types.curriculum import CurriculumManagerCfg, CurriculumTermCfg
from holosoma.managers.utils import resolve_callable

from .base import CurriculumTermBase


class CurriculumManager:
    """Drive curriculum terms at setup, reset, and step boundaries.

    Curriculum terms adjust difficulty or scaling factors as training
    progresses. The manager resolves the configured terms, instantiates
    stateful ones, and triggers their lifecycle hooks automatically so that
    environments remain free of ad-hoc curriculum state.

    Parameters
    ----------
    cfg : CurriculumManagerCfg
        Curriculum manager configuration specifying terms and parameters.
    env : Any
        Environment instance operated on by curriculum terms.
    device : str
        Device identifier used by stateful terms.
    """

    def __init__(self, cfg: CurriculumManagerCfg, env: Any, device: str):
        self.cfg = cfg
        self.env = env
        self.device = device
        self.logger = getattr(env, "logger", None)

        # Storage for resolved functions (stateless)
        self._setup_funcs: dict[str, Any] = {}
        self._reset_funcs: dict[str, Any] = {}
        self._step_funcs: dict[str, Any] = {}

        # Storage for term instances (stateful) - tracked separately for automatic lifecycle calls
        self._class_instances: list[CurriculumTermBase] = []
        self._class_terms: dict[str, CurriculumTermBase] = {}

        # Storage for term names and configs
        self._setup_names: list[str] = []
        self._reset_names: list[str] = []
        self._step_names: list[str] = []

        self._setup_cfgs: list[CurriculumTermCfg] = []
        self._reset_cfgs: list[CurriculumTermCfg] = []
        self._step_cfgs: list[CurriculumTermCfg] = []

        # Initialize terms
        self._initialize_terms()

    def _initialize_terms(self) -> None:
        """Initialize curriculum terms and resolve their functions/classes.

        Notes
        -----
        Similar to IsaacLab, class-based terms are instantiated once and tracked
        separately so their lifecycle methods (``setup``/``reset``/``step``) can
        be called automatically.
        """
        for term_name, term_cfg in self.cfg.setup_terms.items():
            resolved = resolve_callable(term_cfg.func, context="curriculum term")

            # Check if it's a class (stateful) or function (stateless)
            if isinstance(resolved, type) and issubclass(resolved, CurriculumTermBase):
                # Stateful term - instantiate (or reuse existing) and track for automatic lifecycle calls
                instance = self._class_terms.get(term_name)
                if instance is None:
                    instance = resolved(term_cfg, self.env)
                    self._class_terms[term_name] = instance
                    self._class_instances.append(instance)
            else:
                # Stateless function
                self._setup_funcs[term_name] = resolved

            self._setup_names.append(term_name)
            self._setup_cfgs.append(term_cfg)

        for term_name, term_cfg in self.cfg.reset_terms.items():
            resolved = resolve_callable(term_cfg.func, context="curriculum term")

            if isinstance(resolved, type) and issubclass(resolved, CurriculumTermBase):
                instance = self._class_terms.get(term_name)
                if instance is None:
                    instance = resolved(term_cfg, self.env)
                    self._class_terms[term_name] = instance
                    self._class_instances.append(instance)
            else:
                self._reset_funcs[term_name] = resolved

            self._reset_names.append(term_name)
            self._reset_cfgs.append(term_cfg)

        for term_name, term_cfg in self.cfg.step_terms.items():
            resolved = resolve_callable(term_cfg.func, context="curriculum term")

            if isinstance(resolved, type) and issubclass(resolved, CurriculumTermBase):
                instance = self._class_terms.get(term_name)
                if instance is None:
                    instance = resolved(term_cfg, self.env)
                    self._class_terms[term_name] = instance
                    self._class_instances.append(instance)
            else:
                self._step_funcs[term_name] = resolved

            self._step_names.append(term_name)
            self._step_cfgs.append(term_cfg)

    def setup(self) -> None:
        """Run setup hooks.

        Calls setup() on all class-based curriculum terms, then executes setup functions.
        """
        self.env._manager_curriculum_cfg = self.cfg

        # First, call setup() on all class-based terms (IsaacLab pattern)
        for instance in self._class_instances:
            instance.setup()

        # Then execute stateless setup functions
        for term_name, term_cfg in zip(self._setup_names, self._setup_cfgs):
            if term_name in self._setup_funcs:
                func = self._setup_funcs[term_name]
                func(self.env, **term_cfg.params)

    def reset(self, env_ids: Any) -> None:
        """Run reset hooks.

        Automatically calls ``reset()`` on all class-based curriculum terms, then executes reset functions.

        Parameters
        ----------
        env_ids : Any
            Environment identifiers selected for reset (matches environment API expectations).
        """
        # Automatically call reset() on all class-based terms (IsaacLab pattern)
        for instance in self._class_instances:
            instance.reset(env_ids)

        # Then execute stateless reset functions
        for term_name, term_cfg in zip(self._reset_names, self._reset_cfgs):
            if term_name in self._reset_funcs:
                func = self._reset_funcs[term_name]
                func(self.env, env_ids, **term_cfg.params)

    def step(self) -> None:
        """Run step hooks.

        Automatically calls ``step()`` on all class-based curriculum terms, then executes step functions.
        """
        # Automatically call step() on all class-based terms (IsaacLab pattern)
        for instance in self._class_instances:
            instance.step()

        # Then execute stateless step functions
        for term_name, term_cfg in zip(self._step_names, self._step_cfgs):
            if term_name in self._step_funcs:
                func = self._step_funcs[term_name]
                func(self.env, **term_cfg.params)

    def get_term(self, name: str) -> CurriculumTermBase | None:
        """Return the instantiated curriculum term by name, if available."""
        return self._class_terms.get(name)

    def iter_terms(self) -> ItemsView[str, CurriculumTermBase]:
        """Iterate over registered class-based curriculum terms."""
        return self._class_terms.items()
