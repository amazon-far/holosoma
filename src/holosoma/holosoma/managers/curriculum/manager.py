"""Curriculum manager coordinating curriculum hooks."""

from __future__ import annotations

from collections.abc import ItemsView, Mapping
from typing import Any

from loguru import logger

from holosoma.config_types.curriculum import CurriculumManagerCfg
from holosoma.hooks import Phase
from holosoma.managers.staged_terms import StagedTerms
from holosoma.utils.safe_torch_import import torch

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

        self.staged_terms = StagedTerms(
            cfg,
            env,
            term_base=CurriculumTermBase,
            context="curriculum term",
            owner=self,
            class_terms_run_every_stage=True,
        )

    def install(self) -> None:
        """Set up the terms and drive them from the task's hooks.

        ``reset`` runs on every ``Phase.RESET`` and ``step`` on every ``Phase.FRAME_BEGIN``; the task
        also collects this manager's term metrics.
        """
        self.setup()
        self.env.hooks.add(Phase.RESET, self.reset, name="curriculum.reset")
        self.env.hooks.add(Phase.FRAME_BEGIN, self.step, name="curriculum.step")
        self.env.add_metric_source(self)

    def setup(self) -> None:
        """Run setup hooks: ``setup()`` on every class term, then the setup functions."""
        self.staged_terms.run("setup", self.env)

    def reset(self, env_ids: Any) -> None:
        """Run reset hooks: ``reset(env_ids)`` on every class term, then the reset functions."""
        self.staged_terms.run("reset", self.env, env_ids)

    def step(self) -> None:
        """Run step hooks: ``step()`` on every class term, then the step functions."""
        self.staged_terms.run("step", self.env)

    def get_term(self, name: str) -> CurriculumTermBase | None:
        """Return the instantiated curriculum term by name, if available."""
        return self.staged_terms.instances.get(name)

    def iter_terms(self) -> ItemsView[str, CurriculumTermBase]:
        """Iterate over registered class-based curriculum terms."""
        return self.staged_terms.instances.items()

    def state_dict(self) -> dict[str, dict[str, Any]]:
        """Collect checkpoint state from class-based terms by name; terms with no state are left out."""
        return {name: s for name, term in self.staged_terms.instances.items() if (s := term.state_dict())}

    def load_state_dict(self, state: Mapping[str, dict[str, Any]]) -> None:
        """Restore term state returned by :meth:`state_dict`."""
        for name, term_state in state.items():
            term = self.staged_terms.instances.get(name)
            if term is None:
                logger.warning(f"Checkpoint has curriculum state for unknown term '{name}'; ignoring it.")
                continue
            term.load_state_dict(term_state)

    def load_checkpoint_state(self, env_state: Mapping[str, Any]) -> None:
        """Restore the state saved under ``env_state["curriculum"]``, or under the legacy top-level keys."""
        from holosoma.managers.curriculum.terms.locomotion import convert_legacy_checkpoint_state

        if "curriculum" not in env_state:
            env_state = convert_legacy_checkpoint_state(self, dict(env_state))
        state = env_state.get("curriculum")
        if not state:
            return
        self.load_state_dict(state)
        for name in state:
            term = self.get_term(name)
            if term is not None:
                term.after_checkpoint_load()

    def synchronize(self, *, device: str, world_size: int) -> None:
        """Broadcast rank 0's curriculum state to every rank."""
        if world_size <= 1:
            return
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return
        # Every rank builds the same terms from the same config, so all ranks agree on skipping. A term
        # whose state is empty depending on runtime data would deadlock here.
        state = self.state_dict()
        if not state:
            return
        if not getattr(self, "_logged_sync", False):
            logger.info(f"Multi-GPU curriculum synchronization enabled for terms: {sorted(state)}")
            self._logged_sync = True
        payload = [state]
        torch.distributed.broadcast_object_list(payload, src=0, device=torch.device(device))
        if torch.distributed.get_rank() != 0:
            self.load_state_dict(payload[0])
