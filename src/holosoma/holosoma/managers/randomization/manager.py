"""Randomization manager coordinating startup and episodic randomization hooks."""

from __future__ import annotations

import importlib
from typing import Any

from holosoma.config_types.randomization import RandomizationManagerCfg, RandomizationTermCfg
from holosoma.managers.randomization.exceptions import RandomizerNotSupportedError
from holosoma.utils.sampler import STAGE_RESET, STAGE_SETUP, STAGE_STEP, TermSampler

from .base import RandomizationTermBase


class RandomizationManager:
    """Apply domain-randomization hooks during setup, reset, and rollout.

    The manager loads the configured randomization terms, instantiates
    stateful classes, and invokes their lifecycle methods so that random
    perturbations (pushes, friction, masses, sensor noise, etc.) stay
    centralized. Stateless functions and stateful terms can be mixed freely.

    Parameters
    ----------
    cfg : RandomizationManagerCfg
        Randomization manager configuration describing the active terms.
    env : Any
        Environment instance whose state is randomized.
    device : str
        Device identifier passed to stateful terms.
    """

    def __init__(self, cfg: RandomizationManagerCfg, env: Any, device: str):
        self.cfg = cfg
        self.env = env
        self.device = device
        self.logger = getattr(env, "logger", None)

        # Storage for resolved stateless functions
        self._setup_funcs: dict[str, Any] = {}
        self._reset_funcs: dict[str, Any] = {}
        self._step_funcs: dict[str, Any] = {}

        # Storage for class-based hooks with their lifecycle bindings
        self._class_entries: list[dict[str, Any]] = []
        self._state_terms: dict[str, RandomizationTermBase] = {}

        # Storage for term names and configs
        self._setup_names: list[str] = []
        self._reset_names: list[str] = []
        self._step_names: list[str] = []

        self._setup_cfgs: list[RandomizationTermCfg] = []
        self._reset_cfgs: list[RandomizationTermCfg] = []
        self._step_cfgs: list[RandomizationTermCfg] = []

        # Track failed randomizers to avoid re-attempting them
        self._failed_randomizers: set[str] = set()

        # Initialize terms with filtering
        self._initialize_terms()

    def _initialize_terms(self) -> None:
        """Initialize randomization terms and resolve their functions/classes."""
        # Setup terms
        for term_name, term_cfg in self.cfg.setup_terms.items():
            resolved = self._resolve_function(term_cfg.func)
            if isinstance(resolved, type):
                if not issubclass(resolved, RandomizationTermBase):
                    raise TypeError(
                        f"Class-based randomization term '{term_name}' must inherit from RandomizationTermBase."
                    )
                self._register_class_term(term_name, resolved, term_cfg, "setup")
            else:
                self._setup_funcs[term_name] = resolved
            self._setup_names.append(term_name)
            self._setup_cfgs.append(term_cfg)

        # Reset terms
        for term_name, term_cfg in self.cfg.reset_terms.items():
            resolved = self._resolve_function(term_cfg.func)
            if isinstance(resolved, type):
                if not issubclass(resolved, RandomizationTermBase):
                    raise TypeError(
                        f"Class-based randomization term '{term_name}' must inherit from RandomizationTermBase."
                    )
                self._register_class_term(term_name, resolved, term_cfg, "reset")
            else:
                self._reset_funcs[term_name] = resolved
            self._reset_names.append(term_name)
            self._reset_cfgs.append(term_cfg)

        # Step terms
        for term_name, term_cfg in self.cfg.step_terms.items():
            resolved = self._resolve_function(term_cfg.func)
            if isinstance(resolved, type):
                if not issubclass(resolved, RandomizationTermBase):
                    raise TypeError(
                        f"Class-based randomization term '{term_name}' must inherit from RandomizationTermBase."
                    )
                self._register_class_term(term_name, resolved, term_cfg, "step")
            else:
                self._step_funcs[term_name] = resolved
            self._step_names.append(term_name)
            self._step_cfgs.append(term_cfg)

    def _register_class_term(
        self,
        term_name: str,
        term_class: type[RandomizationTermBase],
        term_cfg: RandomizationTermCfg,
        stage: str,
    ) -> None:
        """Instantiate or update a class-based randomization hook for the given stage.

        Parameters
        ----------
        term_name : str
            Registry name of the randomization term.
        term_class : type[RandomizationTermBase]
            Randomization term implementation class.
        term_cfg : RandomizationTermCfg
            Configuration for the randomization term.
        stage : str
            Lifecycle stage being registered (``"setup"``, ``"reset"``, or ``"step"``).
        """
        for entry in self._class_entries:
            if entry["name"] != term_name:
                continue
            # Same registry name across stages must be the SAME class — reuse the one instance so a
            # multi-stage term shares state. A name collision with a DIFFERENT class is a config error
            # (it would otherwise silently create a duplicate entry while _state_terms kept only one).
            if type(entry["instance"]) is not term_class:
                raise ValueError(
                    f"Randomization term '{term_name}' is registered with conflicting classes "
                    f"{type(entry['instance']).__name__} and {term_class.__name__}."
                )
            entry["stages"].add(stage)
            self._state_terms.setdefault(term_name, entry["instance"])
            return

        instance = term_class(term_cfg, self.env)
        instance.manager = self
        self._class_entries.append({"name": term_name, "instance": instance, "stages": {stage}})
        self._state_terms[term_name] = instance

    def get_state(self, term_name: str) -> RandomizationTermBase | None:
        """Return a stateful randomization term by name.

        Parameters
        ----------
        term_name : str
            Name of the randomization term.

        Returns
        -------
        RandomizationTermBase or None
            Stateful term instance if found, otherwise ``None``.
        """
        return self._state_terms.get(term_name)

    def _resolve_function(self, func: Any | str) -> Any:
        """Resolve a randomization callable or class from a string specification.

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
                raise ValueError(f"Failed to import randomization function '{func}': {exc}") from exc
        return func

    def setup(self) -> None:
        """Run startup hooks."""
        # Run setup for class-based terms, filtering out unsupported ones
        for entry in self._class_entries:
            if "setup" in entry["stages"]:
                try:
                    entry["instance"].setup(
                        TermSampler.bind(self.env.dr_base_seed, entry["name"], STAGE_SETUP, self.env.dr_episode_count)
                    )
                except RandomizerNotSupportedError:
                    if self.cfg.ignore_unsupported:
                        # Mark as failed to skip in reset() and step()
                        self._failed_randomizers.add(entry["name"])
                    else:
                        raise

        # Run setup for function-based terms
        for term_name, term_cfg in zip(self._setup_names, self._setup_cfgs):
            if term_name in self._setup_funcs:
                func = self._setup_funcs[term_name]
                try:
                    func(
                        self.env,
                        sampler=TermSampler.bind(
                            self.env.dr_base_seed, term_name, STAGE_SETUP, self.env.dr_episode_count
                        ),
                        **term_cfg.params,
                    )
                except RandomizerNotSupportedError:
                    if self.cfg.ignore_unsupported:
                        self._failed_randomizers.add(term_name)
                    else:
                        raise

        # Note: Do NOT call refresh_sim_tensors() here for IsaacGym during startup.
        # For manager-based environments, setup() is called BEFORE prepare_sim(), so tensors
        # haven't been acquired yet. The tensors will be properly initialized in prepare_sim().
        # For IsaacSim, we still need to write data and refresh.
        if type(self.env.simulator).__name__ == "IsaacSim":
            self.env.simulator.scene.write_data_to_sim()
            self.env.simulator.refresh_sim_tensors()

    def reset(self, env_ids) -> None:
        """Run episodic hooks during environment reset.

        Parameters
        ----------
        env_ids : Any
            Environment identifiers selected for reset (matches environment API expectations).
        """
        for entry in self._class_entries:
            if "reset" in entry["stages"] and entry["name"] not in self._failed_randomizers:
                entry["instance"].reset(
                    env_ids,
                    TermSampler.bind(self.env.dr_base_seed, entry["name"], STAGE_RESET, self.env.dr_episode_count),
                )

        for term_name, term_cfg in zip(self._reset_names, self._reset_cfgs):
            if term_name in self._reset_funcs and term_name not in self._failed_randomizers:
                func = self._reset_funcs[term_name]
                # Don't catch unsupported here because setup() has already checked
                func(
                    self.env,
                    env_ids,
                    sampler=TermSampler.bind(self.env.dr_base_seed, term_name, STAGE_RESET, self.env.dr_episode_count),
                    **term_cfg.params,
                )

    def step(self) -> None:
        """Run per-step hooks (if any are configured)."""
        for entry in self._class_entries:
            if "step" in entry["stages"] and entry["name"] not in self._failed_randomizers:
                entry["instance"].step(
                    TermSampler.bind(self.env.dr_base_seed, entry["name"], STAGE_STEP, self.env.dr_episode_count)
                )

        for term_name, term_cfg in zip(self._step_names, self._step_cfgs):
            if term_name in self._step_funcs and term_name not in self._failed_randomizers:
                func = self._step_funcs[term_name]
                func(
                    self.env,
                    sampler=TermSampler.bind(self.env.dr_base_seed, term_name, STAGE_STEP, self.env.dr_episode_count),
                    **term_cfg.params,
                )
