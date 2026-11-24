"""Randomization manager coordinating startup and episodic randomization hooks."""

from __future__ import annotations

from collections.abc import ItemsView, Iterator
from contextlib import contextmanager
from typing import Any

from holosoma.config_types.randomization import RandomizationManagerCfg
from holosoma.managers.randomization.exceptions import RandomizerNotSupportedError
from holosoma.managers.staged_terms import Stage, StagedTerms
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

        self.staged_terms = StagedTerms(
            cfg, env, term_base=RandomizationTermBase, context="randomization term", owner=self
        )

        # Terms a simulator does not support, skipped after setup when ``ignore_unsupported`` is set
        self._failed_randomizers: set[str] = set()

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
        return self.get_term(term_name)

    def get_term(self, name: str) -> RandomizationTermBase | None:
        """Return the class term registered as ``name``, or ``None``."""
        return self.staged_terms.instances.get(name)

    def iter_terms(self) -> ItemsView[str, RandomizationTermBase]:
        """Iterate over ``(name, instance)`` for the class terms."""
        return self.staged_terms.instances.items()

    def _sampler(self, term_name: str, stage: int) -> TermSampler:
        return TermSampler.bind(self.env.dr_base_seed, term_name, stage, self.env.episode_count)

    @contextmanager
    def _skip_if_unsupported(self, term_name: str) -> Iterator[None]:
        """Record ``term_name`` as failed instead of raising when ``ignore_unsupported`` is set."""
        try:
            yield
        except RandomizerNotSupportedError:
            if not self.cfg.ignore_unsupported:
                raise
            self._failed_randomizers.add(term_name)

    def setup(self) -> None:
        """Run startup hooks."""
        for name, instance in self.staged_terms.classes("setup"):
            with self._skip_if_unsupported(name):
                instance.setup(self._sampler(name, STAGE_SETUP))
        for name, func, term_cfg in self.staged_terms.functions("setup"):
            with self._skip_if_unsupported(name):
                func(self.env, sampler=self._sampler(name, STAGE_SETUP), **term_cfg.params)

        self.env.simulator.forward_kinematics()

    def reset(self, env_ids: Any) -> None:
        """Run episodic hooks during environment reset.

        Parameters
        ----------
        env_ids : Any
            Environment identifiers selected for reset (matches environment API expectations).
        """
        self._run("reset", STAGE_RESET, env_ids)

    def step(self) -> None:
        """Run per-step hooks (if any are configured)."""
        self._run("step", STAGE_STEP)

    def _run(self, stage: Stage, sampler_stage: int, *env_ids: Any) -> None:
        """Run ``stage``'s terms that setup did not mark unsupported, each with its own sampler."""
        for name, instance in self.staged_terms.classes(stage):
            if name not in self._failed_randomizers:
                getattr(instance, stage)(*env_ids, self._sampler(name, sampler_stage))
        for name, func, term_cfg in self.staged_terms.functions(stage):
            if name not in self._failed_randomizers:
                func(self.env, *env_ids, sampler=self._sampler(name, sampler_stage), **term_cfg.params)
