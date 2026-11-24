"""Setup, reset and step terms shared by the command, randomization and curriculum managers."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from holosoma.managers.utils import resolve_callable

Stage = Literal["setup", "reset", "step"]
STAGES: tuple[Stage, ...] = ("setup", "reset", "step")


@dataclass
class _ClassTerm:
    instance: Any
    stages: set[Stage] = field(default_factory=set)


class StagedTerms:
    """Terms configured under ``setup_terms``, ``reset_terms`` and ``step_terms``.

    Each term is a function or a subclass of ``term_base``. A class listed under several stages is one
    shared instance. In every stage, class terms run first, in the order they were first listed, then
    function terms in config order.

    Parameters
    ----------
    cfg : Any
        Manager config with ``setup_terms``, ``reset_terms`` and ``step_terms`` mappings.
    env : Any
        Task passed to class-term constructors.
    term_base : type
        Base class every class term must inherit from.
    context : str
        Term kind named in error messages, e.g. ``"command term"``.
    owner : Any
        Manager assigned to each class term's ``manager`` attribute.
    class_terms_run_every_stage : bool, optional
        Run each class term in all three stages, not only the stages it is listed under.

    Raises
    ------
    TypeError
        If a class term does not inherit from ``term_base``.
    ValueError
        If one name is listed with different implementations.
    """

    def __init__(
        self,
        cfg: Any,
        env: Any,
        *,
        term_base: type[Any],
        context: str,
        owner: Any,
        class_terms_run_every_stage: bool = False,
    ) -> None:
        self._class_terms: dict[str, _ClassTerm] = {}
        self._functions: dict[Stage, list[tuple[str, Callable[..., Any], Any]]] = {stage: [] for stage in STAGES}
        self._function_names: set[str] = set()
        self._run_every_stage = class_terms_run_every_stage
        self.instances: dict[str, Any] = {}
        """Class terms by name, in the order they were first listed."""

        for stage in STAGES:
            for name, term_cfg in getattr(cfg, f"{stage}_terms").items():
                resolved = resolve_callable(term_cfg.func, context=context)
                if isinstance(resolved, type):
                    self._add_class(name, resolved, term_cfg, stage, env, term_base, context, owner)
                else:
                    self._add_function(name, resolved, term_cfg, stage, context)

    def _add_class(
        self,
        name: str,
        term_class: type[Any],
        term_cfg: Any,
        stage: Stage,
        env: Any,
        term_base: type[Any],
        context: str,
        owner: Any,
    ) -> None:
        if not issubclass(term_class, term_base):
            raise TypeError(f"Class-based {context} '{name}' must inherit from {term_base.__name__}.")
        if name in self._function_names:
            raise ValueError(f"{context.capitalize()} '{name}' is listed as both a function and a class.")
        entry = self._class_terms.get(name)
        if entry is None:
            instance: Any = term_class(term_cfg, env)
            instance.manager = owner
            entry = self._class_terms[name] = _ClassTerm(instance)
            self.instances[name] = instance
        elif type(entry.instance) is not term_class:
            raise ValueError(
                f"{context.capitalize()} '{name}' is listed with conflicting classes "
                f"{type(entry.instance).__name__} and {term_class.__name__}."
            )
        entry.stages.add(stage)

    def _add_function(self, name: str, func: Callable[..., Any], term_cfg: Any, stage: Stage, context: str) -> None:
        if name in self._class_terms:
            raise ValueError(f"{context.capitalize()} '{name}' is listed as both a function and a class.")
        self._function_names.add(name)
        self._functions[stage].append((name, func, term_cfg))

    def classes(self, stage: Stage) -> Iterator[tuple[str, Any]]:
        """Yield ``(name, instance)`` for the class terms that run in ``stage``."""
        for name, entry in self._class_terms.items():
            if self._run_every_stage or stage in entry.stages:
                yield name, entry.instance

    def functions(self, stage: Stage) -> list[tuple[str, Callable[..., Any], Any]]:
        """Return ``(name, function, term_cfg)`` for the function terms listed under ``stage``."""
        return self._functions[stage]

    def names(self, stage: Stage) -> list[str]:
        """Return the names listed under ``stage``: class terms first, then functions."""
        return [name for name, _ in self.classes(stage)] + [name for name, _, _ in self.functions(stage)]

    def run(self, stage: Stage, env: Any, env_ids: Any = None) -> None:
        """Run ``stage`` with the plain calling convention.

        Class terms get ``setup()``, ``reset(env_ids)`` or ``step()``; functions get ``func(env, **params)``,
        with ``env_ids`` after ``env`` in the reset stage.
        """
        for _, instance in self.classes(stage):
            if stage == "reset":
                instance.reset(env_ids)
            else:
                getattr(instance, stage)()
        for _, func, term_cfg in self.functions(stage):
            params: Mapping[str, Any] = term_cfg.params or {}
            if stage == "reset":
                func(env, env_ids, **params)
            else:
                func(env, **params)
