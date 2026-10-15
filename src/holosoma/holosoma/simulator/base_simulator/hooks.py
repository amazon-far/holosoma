from __future__ import annotations

import heapq
import inspect
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Sequence, overload

from loguru import logger
from typing_extensions import Literal, Self

from holosoma.config_types.frequency import DecimationLike, is_frequency_string, resolve_decimation


class Phase(str, Enum):
    """Lifecycle points emitted by the active simulator loop.

    ``payload`` names the positional args ``emit`` forwards to callbacks. ``periodic`` marks phases
    that fire on a fixed clock (per-substep or per-frame ticks) — only these accept a frequency-string
    rate (``"30Hz"``) for :meth:`HookRegistry.add`'s ``every``. Episode events accept an int decimation
    ("every Nth event") but reject frequency strings (no clock to resolve against); CLOSE always
    fires once and rejects any decimation.

    ``FRAME_*`` bracket the outer tick (once per frame); ``*_STEP`` bracket each physics substep. The
    names are engine-agnostic — they don't assume what drives the outer tick (a policy, a bridge, ...).
    """

    payload: tuple[str, ...]
    periodic: bool

    FRAME_BEGIN = ("frame_begin", (), True)
    PRE_STEP = ("pre_step", (), True)
    POST_STEP = ("post_step", (), True)
    FRAME_END = ("frame_end", (), True)
    EPISODE_END = ("episode_end", ("env_id",), False)
    EPISODE_START = ("episode_start", ("env_id",), False)
    CLOSE = ("close", (), False)

    def __new__(cls, value: str, payload: tuple[str, ...], periodic: bool) -> Self:
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.payload = payload
        obj.periodic = periodic
        return obj


# Callback shapes keyed to each phase's payload, so :meth:`HookRegistry.add`'s typed overloads
# reject an arity mismatch (e.g. a one-arg callback on a zero-payload phase) at type-check time.
# A callback with only-defaulted params still satisfies the zero-arg alias, matching the runtime
# ``inspect``-based check in ``add``.
FrameCallback = Callable[[], Any]
"""Callback for a zero-payload periodic phase (``FRAME_BEGIN``/``PRE_STEP``/``POST_STEP``/``FRAME_END``)."""
EpisodeCallback = Callable[[int], Any]
"""Callback for an episode phase (``EPISODE_START``/``EPISODE_END``); receives ``env_id``."""
CloseCallback = Callable[[], Any]
"""Callback for ``CLOSE`` (zero payload, fires once at teardown)."""


class HookRegistryError(RuntimeError):
    """Raised when the hook registry is misused."""


class HookOrderError(HookRegistryError):
    """Raised when a phase's ``after``/``before`` constraints cannot be ordered (a cycle)."""


class HookCloseError(RuntimeError):
    """Raised when a close hook fails."""

    def __init__(self, failures: list[tuple[str, Exception]]) -> None:
        self.failures = failures
        details = ", ".join(f"{name}: {exc!r}" for name, exc in failures)
        super().__init__(f"{len(failures)} close hook(s) failed: {details}")


@dataclass
class _HookRecord:
    phase: Phase
    callback: Callable[..., Any]
    name: str
    every: int = 1
    """Resolved emission decimation: the callback runs once per ``every`` emissions of its phase."""
    after: tuple[str, ...] = ()
    """Names of same-phase hooks this one must run after."""
    before: tuple[str, ...] = ()
    """Names of same-phase hooks this one must run before."""
    _counter: int = 0
    """Emissions counted toward the next fire; only advances while enabled (disabled hooks leave the snapshot)."""
    enabled: bool = True
    removed: bool = False

    def due(self) -> bool:
        """Advance the per-hook counter and report whether this emission should fire the callback."""
        self._counter += 1
        if self._counter >= self.every:
            self._counter = 0
            return True
        return False


class HookHandle:
    """Identity handle returned by hook registration."""

    def __init__(self, registry: HookRegistry, record: _HookRecord) -> None:
        self._registry = registry
        self._record = record

    @property
    def name(self) -> str:
        return self._record.name

    @property
    def phase(self) -> Phase:
        return self._record.phase

    @property
    def enabled(self) -> bool:
        return self._record.enabled and not self._record.removed

    @property
    def every(self) -> int:
        """Current resolved emission decimation (see :meth:`set_every`)."""
        return self._record.every

    def enable(self) -> None:
        self._registry._enable(self._record)

    def disable(self) -> None:
        self._registry._disable(self._record)

    def set_every(self, every: DecimationLike) -> None:
        """Change this hook's cadence live (int decimation or frequency string), resetting its counter.

        Same rules as ``every`` at registration. Safe to call while other phases emit, but not from
        inside this hook's own phase (mutating a phase mid-emit raises)."""
        self._registry._set_every(self._record, every)

    def remove(self) -> None:
        self._registry._remove(self._record)


class HookRegistry:
    """Small lifecycle hook registry: registration order, refined by declared ``after``/``before``.

    Within a phase, hooks run in registration order unless they declare an ordering constraint; the
    resolved order is computed lazily on :meth:`emit` (see :meth:`_resolve_order`) and cached until a
    registration change invalidates it.
    """

    def __init__(self, base_rates: Mapping[Phase, float] | None = None) -> None:
        self._records: dict[Phase, list[_HookRecord]] = {phase: [] for phase in Phase}
        self._snapshots: dict[Phase, tuple[_HookRecord, ...]] = dict.fromkeys(Phase, ())
        self._stale: set[Phase] = set()
        self._warned_missing: set[tuple[Phase, str]] = set()
        self._emitting: set[Phase] = set()
        self._closed = False
        self._base_rates: dict[Phase, float] = dict(base_rates or {})

    @overload
    def add(
        self,
        phase: Literal[Phase.FRAME_BEGIN, Phase.PRE_STEP, Phase.POST_STEP, Phase.FRAME_END],
        callback: FrameCallback,
        *,
        name: str | None = None,
        every: DecimationLike = 1,
        after: Sequence[str] = (),
        before: Sequence[str] = (),
    ) -> HookHandle: ...

    @overload
    def add(
        self,
        phase: Literal[Phase.EPISODE_END, Phase.EPISODE_START],
        callback: EpisodeCallback,
        *,
        name: str | None = None,
        every: DecimationLike = 1,
        after: Sequence[str] = (),
        before: Sequence[str] = (),
    ) -> HookHandle: ...

    @overload
    def add(
        self,
        phase: Literal[Phase.CLOSE],
        callback: CloseCallback,
        *,
        name: str | None = None,
        every: DecimationLike = 1,
        after: Sequence[str] = (),
        before: Sequence[str] = (),
    ) -> HookHandle: ...

    def add(
        self,
        phase: Phase,
        callback: Callable[..., Any],
        *,
        name: str | None = None,
        every: DecimationLike = 1,
        after: Sequence[str] = (),
        before: Sequence[str] = (),
    ) -> HookHandle:
        """Register a hook callback for a lifecycle phase.

        The callback's signature must accept the phase's payload (see :class:`Phase`): zero args for
        the periodic ``FRAME_*``/``*_STEP`` phases and ``CLOSE``, one ``env_id`` for the episode
        phases. The typed overloads reject a mismatch at type-check time; :meth:`_check_arity` also
        checks it at registration, so a wrong-arity callback fails loudly at ``add`` rather than with a
        ``TypeError`` on the first ``emit``. A parameter with a default still satisfies a phase that
        passes fewer args (e.g. ``capture_frame(env_id: int = 0)`` on the zero-payload ``FRAME_END``).

        ``every`` sub-samples emissions: an int decimation runs the callback once per ``every``
        emissions; a frequency string (``"30Hz"``, ``">30Hz"``, ``"<30Hz"``) is resolved against the
        phase's base tick rate into that decimation. Frequency strings require a periodic phase and a
        known base rate; event phases (episode/close) accept only int decimations. Default ``1`` fires
        every emission.

        ``after`` / ``before`` order this hook against other hooks *on the same phase*, by name: each
        name matches every hook registered under it (names need not be unique), and lookup is
        phase-scoped — a name registered only on another phase does not exist from here and is
        reported once as a missing constraint, not an edge. Hooks left unconstrained keep their
        registration slot. ``CLOSE`` resolves the same way and then runs it backwards, so a hook that
        declares ``after`` tears down before the hook it follows.
        """
        self._assert_mutable(phase)
        resolved_name = name or self._default_name(callback)
        self._check_arity(phase, callback, resolved_name)
        record = _HookRecord(
            phase=phase,
            callback=callback,
            name=resolved_name,
            every=self._resolve_every(phase, every, name),
            after=self._normalize_deps(after, "after", resolved_name, phase),
            before=self._normalize_deps(before, "before", resolved_name, phase),
        )
        self._records[phase].append(record)
        self._invalidate(phase)
        return HookHandle(self, record)

    @staticmethod
    def _normalize_deps(deps: Sequence[str], field_name: str, name: str, phase: Phase) -> tuple[str, ...]:
        """Freeze a constraint list, rejecting a bare string (which would order against its letters)."""
        if isinstance(deps, str):
            raise HookRegistryError(
                f"hook {name!r} on phase {phase.value!r}: {field_name}={deps!r} is a string; "
                f"pass a sequence of hook names, e.g. {field_name}=[{deps!r}]."
            )
        return tuple(deps)

    @staticmethod
    def _check_arity(phase: Phase, callback: Callable[..., Any], name: str) -> None:
        """Fail at registration if ``callback`` can't take the args ``emit`` will pass for ``phase``.

        ``emit`` calls ``callback(*payload)`` with ``len(phase.payload)`` positional args. A callback is
        compatible when it requires no more than that many positional params (extras must have defaults)
        and can accept that many (via fixed params or ``*args``). Built-ins and C callables whose
        signature ``inspect`` can't read are left to runtime.
        """
        try:
            params = list(inspect.signature(callback).parameters.values())
        except (TypeError, ValueError):
            return  # Signature not introspectable (e.g. some C callables); defer to runtime.

        n = len(phase.payload)
        positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        required = sum(1 for p in positional if p.default is p.empty)
        has_varargs = any(p.kind is p.VAR_POSITIONAL for p in params)
        max_positional = len(positional)

        if required > n:
            raise HookRegistryError(
                f"hook {name!r} on phase {phase.value!r}: callback requires {required} positional "
                f"arg(s) but the phase emits {n} ({', '.join(phase.payload) or 'none'}). "
                f"Give the extra param(s) a default or drop them."
            )
        if not has_varargs and n > max_positional:
            raise HookRegistryError(
                f"hook {name!r} on phase {phase.value!r}: phase emits {n} arg(s) "
                f"({', '.join(phase.payload)}) but the callback accepts at most {max_positional}. "
                f"Accept the payload (add the param or *args)."
            )

    def _resolve_every(self, phase: Phase, every: DecimationLike, name: str | None) -> int:
        """Turn a hook's ``every`` into an int decimation, resolving frequency strings vs the base rate."""
        field = f"hook {name or '<callback>'!r} on phase {phase.value!r} 'every'"
        if phase is Phase.CLOSE and every != 1:
            raise HookRegistryError(f"{field}: CLOSE fires once at teardown; decimating it would skip cleanup.")
        if is_frequency_string(every):
            if not phase.periodic:
                raise HookRegistryError(
                    f"{field}: phase {phase.value!r} is not periodic; use an int decimation, not {every!r}."
                )
            base_hz = self._base_rates.get(phase)
            if base_hz is None:
                raise HookRegistryError(
                    f"{field}: no base rate registered for periodic phase {phase.value!r}; "
                    f"cannot resolve frequency {every!r}. Pass base_rates to HookRegistry, or use an int."
                )
            return resolve_decimation(every, base_hz, field=field, log=True)
        return resolve_decimation(every, base_hz=1.0, field=field)

    def emit(self, phase: Phase, *args: Any) -> None:
        """Run due hooks in resolved order; CLOSE runs once, backwards, and stops on failure."""
        self._validate_payload(phase, args)

        if phase is Phase.CLOSE:
            if self._closed:
                return

            if phase in self._emitting:
                raise HookRegistryError("Recursive close hook emission is not supported")

            self._closed = True
            self._emitting.add(phase)
            try:
                for record in reversed(self._close_order()):
                    try:
                        record.callback()
                    except Exception as exc:  # noqa: PERF203 - preserve the failing owner's name.
                        raise HookCloseError([(record.name, exc)]) from exc
            finally:
                self._emitting.remove(phase)
            return

        if phase in self._emitting:
            raise HookRegistryError(f"Recursive hook emission is not supported for phase {phase.value!r}")

        order = self._order(phase)
        self._emitting.add(phase)
        try:
            for record in order:
                if record.due():
                    record.callback(*args)
        finally:
            self._emitting.remove(phase)

    def names(self, phase: Phase) -> tuple[str, ...]:
        """Names of the hooks that would fire on ``phase`` now, in the order they would run.

        Resolves the phase's constraints like :meth:`emit` does, so ``CLOSE`` reads back-to-front.
        Decimated hooks are included regardless of where their counter sits.
        """
        order = self._close_order() if phase is Phase.CLOSE else self._order(phase)
        names = tuple(record.name for record in order)
        return tuple(reversed(names)) if phase is Phase.CLOSE else names

    @staticmethod
    def _validate_payload(phase: Phase, args: tuple[Any, ...]) -> None:
        expected = phase.payload
        if len(args) == len(expected):
            return
        expected_text = ", ".join(expected) or "no arguments"
        raise HookRegistryError(f"Phase {phase.value!r} expects {expected_text}")

    def _enable(self, record: _HookRecord) -> None:
        if record.removed or record.enabled:
            return
        self._assert_mutable(record.phase)
        record.enabled = True
        self._invalidate(record.phase)

    def _disable(self, record: _HookRecord) -> None:
        if record.removed or not record.enabled:
            return
        self._assert_mutable(record.phase)
        record.enabled = False
        self._invalidate(record.phase)

    def _set_every(self, record: _HookRecord, every: DecimationLike) -> None:
        if record.removed:
            raise HookRegistryError(f"Cannot set cadence on removed hook {record.name!r}.")
        self._assert_mutable(record.phase)
        record.every = self._resolve_every(record.phase, every, record.name)
        record._counter = 0

    def _remove(self, record: _HookRecord) -> None:
        if record.removed:
            return
        self._assert_mutable(record.phase)
        record.enabled = False
        record.removed = True
        self._invalidate(record.phase)

    def _invalidate(self, phase: Phase) -> None:
        """Mark a phase's resolved order out of date; :meth:`_order` recomputes it on the next emit."""
        self._stale.add(phase)

    def _order(self, phase: Phase) -> tuple[_HookRecord, ...]:
        """Return the phase's emission order, resolving it first if a registration change invalidated it."""
        if phase in self._stale:
            self._snapshots[phase] = self._resolve_order(phase)
            self._stale.discard(phase)
        return self._snapshots[phase]

    def _close_order(self) -> tuple[_HookRecord, ...]:
        """CLOSE's order, falling back to registration order if its constraints don't resolve.

        A cycle must not cost every hook its teardown (viewers, ROS nodes, open video files), and the
        same cycle raises from any other phase's emit, so here it is logged and the constraints dropped.
        """
        try:
            return self._order(Phase.CLOSE)
        except HookOrderError as exc:
            logger.error(f"{exc} Tearing down in registration order instead.")
            self._stale.discard(Phase.CLOSE)
            self._snapshots[Phase.CLOSE] = tuple(self._live(Phase.CLOSE))
            return self._snapshots[Phase.CLOSE]

    def _live(self, phase: Phase) -> list[_HookRecord]:
        """The phase's hooks that would fire now, in registration order."""
        return [record for record in self._records[phase] if record.enabled and not record.removed]

    def _resolve_order(self, phase: Phase) -> tuple[_HookRecord, ...]:
        """Topologically sort a phase's live hooks by their ``after``/``before`` constraints.

        Stable: among hooks whose constraints are already satisfied, the earliest-registered runs
        first, so an unconstrained phase emits exactly in registration order. A constraint naming a
        hook that is registered on this phase but currently disabled is inert (it returns when the
        hook is re-enabled); one naming no hook on this phase at all is dropped, with a warning logged
        once per name.
        """
        live = self._live(phase)
        registered = {record.name for record in self._records[phase] if not record.removed}
        positions: dict[str, list[int]] = defaultdict(list)
        for position, record in enumerate(live):
            positions[record.name].append(position)

        successors: dict[int, set[int]] = {position: set() for position in range(len(live))}
        blockers = [0] * len(live)

        def link(earlier: int, later: int) -> None:
            if earlier != later and later not in successors[earlier]:
                successors[earlier].add(later)
                blockers[later] += 1

        for position, record in enumerate(live):
            for dep in record.after:
                for target in self._targets(phase, dep, positions, registered):
                    link(target, position)
            for dep in record.before:
                for target in self._targets(phase, dep, positions, registered):
                    link(position, target)

        ready = [position for position in range(len(live)) if blockers[position] == 0]
        heapq.heapify(ready)
        order: list[_HookRecord] = []
        while ready:
            position = heapq.heappop(ready)
            order.append(live[position])
            for later in sorted(successors[position]):
                blockers[later] -= 1
                if blockers[later] == 0:
                    heapq.heappush(ready, later)

        if len(order) != len(live):
            cycle = sorted({live[position].name for position in range(len(live)) if blockers[position] > 0})
            raise HookOrderError(
                f"Hook ordering cycle on phase {phase.value!r} among: {', '.join(cycle)}. "
                f"Their 'after'/'before' constraints cannot all hold."
            )
        return tuple(order)

    def _targets(self, phase: Phase, dep: str, positions: Mapping[str, list[int]], registered: set[str]) -> list[int]:
        """Positions of the live hooks a constraint name refers to, warning once if it names none."""
        if dep in positions:
            return positions[dep]
        if dep not in registered and (phase, dep) not in self._warned_missing:
            self._warned_missing.add((phase, dep))
            logger.warning(
                f"Hook ordering constraint on phase {phase.value!r} names {dep!r}, which no hook on "
                f"that phase is registered under; ignoring it."
            )
        return []

    def _assert_mutable(self, phase: Phase) -> None:
        if phase in self._emitting:
            raise HookRegistryError(f"Cannot mutate hooks while emitting phase {phase.value!r}")

    @staticmethod
    def _default_name(callback: Callable[..., Any]) -> str:
        return getattr(callback, "__qualname__", repr(callback))
