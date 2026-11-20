"""Recipe: a training run described as data, in one readable place.

A ``Recipe`` is an experiment config plus an ordered set of phases. Each phase
holds the values it sets (addressed by :mod:`holosoma.recipes.keys`) and the
ramps that move a value WITHIN that phase. Recipes are built by composing small
functions, so a variant reads as its parent plus the lines that differ.

Precedence, one rule, no exceptions:

    experiment config  <  recipe.set_all  <  phase values  <  CLI --set

Phases are addressed by short key ("A", "B", "C", "DE", "F", "TELEOP"), never
by list position, so inserting a phase cannot silently move a setting onto a
different phase.

``resolve()`` produces the ``RenewPhaseConfig`` list the existing pipeline
already knows how to run; ``describe()`` prints what every phase ends up with,
and where each value came from, without running anything.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from holosoma.agents.early_stop import EarlyStopCfg
from holosoma.agents.sigma_scheduler import SigmaScheduleEntry, SigmaSchedulerCfg
from holosoma.recipes.keys import (
    COMMAND_ATTR,
    REWARD_PARAM,
    REWARD_WEIGHT,
    Key,
    key as lookup_key,
)
from holosoma.training_pipeline_renew import RenewPhaseConfig

__all__ = ["Ramp", "Phase", "Recipe"]


# Units that describe a VALUE TYPE rather than a physical unit: not printed.
_UNITLESS = {"dimensionless", "bool", "path", "enum", "callable-path", "obs-term"}

_TARGET_FIELD = {
    REWARD_PARAM: "params",
    REWARD_WEIGHT: "weight",
    COMMAND_ATTR: "command",
}


@dataclass
class Ramp:
    """A value moved linearly within one phase, as a fraction of that phase.

    Holds ``start`` until ``from_pct`` of the phase has elapsed, interpolates
    to ``end`` at ``until_pct``, and holds ``end`` afterwards. Progress is
    measured in training iterations of the phase, so it follows an early stop.
    """

    key: Key
    start: float
    end: float
    from_pct: float = 0.0
    until_pct: float = 1.0

    def __post_init__(self) -> None:
        if self.key.kind not in _TARGET_FIELD:
            raise ValueError(
                f"{self.key.name} cannot be ramped (kind {self.key.kind}); only "
                "reward params, reward weights and command attributes can."
            )
        if not 0.0 <= self.from_pct < self.until_pct <= 1.0:
            raise ValueError(
                f"ramp on {self.key.name} needs 0 <= from_pct < until_pct <= 1, "
                f"got {self.from_pct} -> {self.until_pct}"
            )
        for v in (self.start, self.end):
            self.key.check(v)

    def entry(self) -> SigmaScheduleEntry:
        return SigmaScheduleEntry(
            term_name=self.key.term,
            param_name=self.key.field if self.key.kind != REWARD_WEIGHT else "",
            start_value=self.start,
            end_value=self.end,
            ramp_start_pct=self.from_pct,
            ramp_end_pct=self.until_pct,
            target_field=_TARGET_FIELD[self.key.kind],
        )

    def describe(self) -> str:
        span = f"{self.from_pct:.0%}-{self.until_pct:.0%}"
        return f"{self.start} -> {self.end} over {span}"


@dataclass
class Phase:
    """One phase of a recipe: what it trains, for how long, and what it ramps."""

    key: str
    """Short handle used to address this phase, e.g. "DE"."""

    name: str
    """Full phase name. Also the pipeline's log and directory name."""

    iterations: int
    values: dict[str, Any] = field(default_factory=dict)
    ramps: list[Ramp] = field(default_factory=list)
    early_stop: EarlyStopCfg | None = None
    source: dict[str, str] = field(default_factory=dict)
    """Key name -> which recipe layer last set it, for describe()."""

    def set(self, key_name: str, value: Any, *, by: str = "phase") -> "Phase":
        k = lookup_key(key_name)
        k.check(value)
        self.values[key_name] = value
        self.source[key_name] = by
        return self

    def ramp(
        self, key_name: str, start: float, end: float,
        *, from_pct: float = 0.0, until_pct: float = 1.0, by: str = "phase",
    ) -> "Phase":
        """Ramp a value within this phase. Does NOT set the phase's own value:
        set it explicitly when the phase should start from something other than
        whatever the config already has."""
        k = lookup_key(key_name)
        self.ramps = [r for r in self.ramps if r.key.name != k.name]
        self.ramps.append(Ramp(k, start, end, from_pct, until_pct))
        self.source[f"{key_name} (ramp)"] = by
        return self

    def unramp(self, key_name: str) -> "Phase":
        self.ramps = [r for r in self.ramps if r.key.name != key_name]
        self.source.pop(f"{key_name} (ramp)", None)
        return self

    def unset(self, key_name: str) -> "Phase":
        self.values.pop(key_name, None)
        self.source.pop(key_name, None)
        return self


class Recipe:
    """An experiment plus its phases. Build one by composing small functions."""

    def __init__(self, exp: str, phases: list[Phase], everywhere: dict[str, Any] | None = None):
        self.exp = exp
        self._phases: dict[str, Phase] = {p.key: p for p in phases}
        self.everywhere: dict[str, Any] = dict(everywhere or {})
        self._everywhere_source: dict[str, str] = {}

    # ------------------------------------------------------------ construction
    def copy(self) -> "Recipe":
        out = Recipe(self.exp, [deepcopy(p) for p in self._phases.values()], deepcopy(self.everywhere))
        out._everywhere_source = dict(self._everywhere_source)
        return out

    def with_exp(self, exp: str) -> "Recipe":
        self.exp = exp
        return self

    @property
    def phase_keys(self) -> list[str]:
        return list(self._phases)

    def phase(self, phase_key: str) -> Phase:
        try:
            return self._phases[phase_key]
        except KeyError:
            raise KeyError(
                f"unknown phase {phase_key!r}; this recipe has {self.phase_keys}"
            ) from None

    def has_phase(self, phase_key: str) -> bool:
        return phase_key in self._phases

    def keep_phases(self, *phase_keys: str) -> "Recipe":
        """Drop every other phase. Each phase is built from the experiment plus
        its own values, so a kept phase resolves exactly as it did in the full
        recipe; only the checkpoint it would have started from is gone."""
        for k in phase_keys:
            self.phase(k)
        self._phases = {k: p for k, p in self._phases.items() if k in phase_keys}
        return self

    def add_phase(self, phase: Phase) -> Phase:
        if phase.key in self._phases:
            raise ValueError(f"phase {phase.key!r} already exists")
        self._phases[phase.key] = phase
        return phase

    def derive_phase(
        self, src_key: str, new_key: str, name: str, iterations: int,
        *, keep_ramps: bool = False, keep_early_stop: bool = False,
    ) -> Phase:
        """Add a phase that STARTS as a copy of another phase's values.

        Used for the teleop phase, which is Phase F with a different dataset.
        Copying by construction is what keeps a value from being annealed in
        one phase and silently reverting in the next.
        """
        src = self.phase(src_key)
        new = Phase(
            key=new_key,
            name=name,
            iterations=iterations,
            values=deepcopy(src.values),
            ramps=deepcopy(src.ramps) if keep_ramps else [],
            early_stop=deepcopy(src.early_stop) if keep_early_stop else None,
            source={k: f"from {src_key}" for k in src.values},
        )
        return self.add_phase(new)

    # -------------------------------------------------------------- set values
    def set(self, phase_key: str, key_name: str, value: Any, *, by: str = "recipe") -> "Recipe":
        self.phase(phase_key).set(key_name, value, by=by)
        return self

    def set_all(self, key_name: str, value: Any, *, by: str = "set_all") -> "Recipe":
        """Set a value for every phase, including ones added later."""
        lookup_key(key_name).check(value)
        self.everywhere[key_name] = value
        self._everywhere_source[key_name] = by
        return self

    def set_per_phase(self, key_name: str, by_phase: dict[str, Any], *, by: str = "recipe") -> "Recipe":
        """Set one key across several phases: {phase_key: value}."""
        for phase_key, value in by_phase.items():
            self.phase(phase_key).set(key_name, value, by=by)
        return self

    def ramp(
        self, phase_key: str, key_name: str, start: float, end: float,
        *, from_pct: float = 0.0, until_pct: float = 1.0, by: str = "recipe",
    ) -> "Recipe":
        self.phase(phase_key).ramp(
            key_name, start, end, from_pct=from_pct, until_pct=until_pct, by=by
        )
        return self

    def set_iterations(self, by_phase: dict[str, int]) -> "Recipe":
        for phase_key, n in by_phase.items():
            self.phase(phase_key).iterations = n
        return self

    # ---------------------------------------------------------------- resolve
    def resolved_values(self, phase_key: str) -> dict[str, Any]:
        """Final key -> value for one phase, applying the precedence rule."""
        phase = self.phase(phase_key)
        return {**self.everywhere, **phase.values}

    def resolve(self) -> list[RenewPhaseConfig]:
        """Build the phase list the existing pipeline runs."""
        out: list[RenewPhaseConfig] = []
        for phase in self._phases.values():
            overrides: dict[str, Any] = {}
            for key_name, value in self.resolved_values(phase.key).items():
                overrides[lookup_key(key_name).path] = value
            schedule = None
            if phase.ramps:
                schedule = SigmaSchedulerCfg(
                    entries=[r.entry() for r in phase.ramps],
                    phase_start_step=0,
                    phase_total_iterations=phase.iterations,
                )
            out.append(
                RenewPhaseConfig(
                    name=phase.name,
                    num_iterations=phase.iterations,
                    overrides=overrides,
                    early_stop=deepcopy(phase.early_stop),
                    sigma_schedule=schedule,
                )
            )
        return out

    # --------------------------------------------------------------- describe
    def describe(self, phase_key: str | None = None, *, docs: bool = False) -> str:
        """Human-readable dump: every value, its source layer, and its ramps."""
        keys = [phase_key] if phase_key else self.phase_keys
        lines = [f"recipe exp: {self.exp}"]
        for pk in keys:
            phase = self.phase(pk)
            lines.append("")
            title = phase.name if phase.name.startswith(f"Phase {pk}") else f"Phase {pk}: {phase.name}"
            lines.append(f"{title}  ({phase.iterations} iterations)")
            if phase.early_stop is not None:
                es = phase.early_stop
                lines.append(
                    f"  early stop: {es.metric} {es.direction} {es.threshold} "
                    f"(patience {es.patience}, min {es.min_iterations})"
                )
            ramp_by_key = {r.key.name: r for r in phase.ramps}
            values = self.resolved_values(pk)
            width = max((len(k) for k in {*values, *ramp_by_key}), default=0)
            for key_name in sorted({*values, *ramp_by_key}):
                k = lookup_key(key_name)
                if key_name in ramp_by_key:
                    shown = ramp_by_key[key_name].describe()
                    src = phase.source.get(f"{key_name} (ramp)", "recipe") + " ramp"
                else:
                    shown = repr(values[key_name])
                    src = phase.source.get(
                        key_name, self._everywhere_source.get(key_name, "set_all")
                    )
                if len(shown) > 72:
                    shown = shown[:69] + "..."
                unit = "" if k.unit in _UNITLESS else f" {k.unit}"
                lines.append(f"  {key_name:<{width}}  {shown}{unit}   [{src}]")
                if docs:
                    for doc_line in k.doc.strip().splitlines():
                        lines.append(f"      | {doc_line}")
        return "\n".join(lines)
