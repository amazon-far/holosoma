from __future__ import annotations

import functools
import heapq
import inspect
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Sequence, overload

from loguru import logger
from typing_extensions import Literal, Self

from holosoma.config_types.frequency import DecimationLike, is_frequency_string, resolve_decimation

SENSOR_RENDER_HOOK = "sensors.render"
"""Name of the hook that refreshes the mounted-sensor buffers, for egress hooks to run ``after``.

Shared by both ends of that dependency because an unmatched name is dropped silently: renaming the
hook has to break its dependents rather than quietly stop ordering them.
"""


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


@dataclass(eq=False)
class _HookRecord:
    """One registered hook. Compared and hashed by identity, since two hooks are never interchangeable.

    ``after``/``before`` can hold other records, so identity comparison also keeps ``==`` and
    ``repr`` from recursing through a pair that references each other.
    """

    phase: Phase
    callback: Callable[..., Any]
    name: str
    every: int = 1
    """Resolved emission decimation: the callback runs once per ``every`` emissions of its phase."""
    after: tuple[str | _HookRecord, ...] = field(default=(), repr=False)
    """Same-phase hooks this one must run after: a name matches all its holders, a record just one."""
    before: tuple[str | _HookRecord, ...] = field(default=(), repr=False)
    """Same-phase hooks this one must run before: a name matches all its holders, a record just one."""
    _counter: int = 0
    """Emissions counted toward the next fire; only advances while this hook is enabled."""
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


def _dep_positions(
    dep: str | _HookRecord, positions: Mapping[str, list[int]], by_record: Mapping[_HookRecord, int]
) -> Sequence[int]:
    """Live positions a constraint refers to: every holder of a name, or the one hook a handle is for."""
    if isinstance(dep, str):
        return positions.get(dep, ())
    position = by_record.get(dep)
    return () if position is None else (position,)


class HookRegistry:
    """Small lifecycle hook registry: registration order, refined by declared ``after``/``before``.

    Within a phase, hooks run in registration order unless they declare an ordering constraint; the
    resolved order is computed lazily on :meth:`emit` (see :meth:`_resolve_order`) and cached until a
    registration change invalidates it.
    """

    def __init__(self, base_rates: Mapping[Phase, float] | None = None) -> None:
        self._records: dict[Phase, list[_HookRecord]] = {phase: [] for phase in Phase}
        self._resolved: dict[Phase, tuple[_HookRecord, ...]] = dict.fromkeys(Phase, ())
        self._stale: set[Phase] = set()
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
        after: Sequence[str | HookHandle] = (),
        before: Sequence[str | HookHandle] = (),
    ) -> HookHandle: ...

    @overload
    def add(
        self,
        phase: Literal[Phase.EPISODE_END, Phase.EPISODE_START],
        callback: EpisodeCallback,
        *,
        name: str | None = None,
        every: DecimationLike = 1,
        after: Sequence[str | HookHandle] = (),
        before: Sequence[str | HookHandle] = (),
    ) -> HookHandle: ...

    @overload
    def add(
        self,
        phase: Literal[Phase.CLOSE],
        callback: CloseCallback,
        *,
        name: str | None = None,
        every: DecimationLike = 1,
        after: Sequence[str | HookHandle] = (),
        before: Sequence[str | HookHandle] = (),
    ) -> HookHandle: ...

    def add(
        self,
        phase: Phase,
        callback: Callable[..., Any],
        *,
        name: str | None = None,
        every: DecimationLike = 1,
        after: Sequence[str | HookHandle] = (),
        before: Sequence[str | HookHandle] = (),
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

        ``name`` defaults to where the callback is defined (``module.QualName``; see
        :meth:`_default_name`), which is unique per place in the source without anyone coordinating.
        Pass one explicitly to give a hook a short public handle for others to depend on
        (``"sensors.render"``), which is also the only way to name a hook whose callback moves —
        a backend override, for instance, derives a different default per subclass.

        ``after`` / ``before`` order this hook against other hooks *on the same phase*. An entry is
        either a name, which matches **every** hook registered under it — so one reference covers a
        plugin selected in several slots — or a :class:`HookHandle`, which pins exactly the one hook
        it came from, for ordering two instances of the same callback against each other. Lookup is
        phase-scoped: a name registered only on another phase does not exist from here, and a handle
        for another phase is rejected outright. A name constraint is advisory — one that matches
        nothing is simply not applied, silently, since a plugin may name a hook belonging to another
        plugin that this run does not select. Disabling a hook costs it its turn to run but not its
        edges, so a chain survives one link switching off. Unconstrained hooks keep their order
        relative to each other.
        ``CLOSE`` resolves the same way and then runs it backwards, so a hook that declares ``after``
        tears down before the hook it follows.
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

    def _normalize_deps(
        self, deps: Sequence[str | HookHandle], field_name: str, name: str, phase: Phase
    ) -> tuple[str | _HookRecord, ...]:
        """Freeze a constraint list, resolving each handle to the single hook it refers to.

        Every rejection here is something that would otherwise order against nothing at all, which an
        unmatched *name* is allowed to do silently: a bare string would order against its own
        letters, a callback or plugin object matches no hook, and a handle for another phase or
        another registry cannot be reached from this one. Whoever holds a handle already knows where
        it came from, so those are mistakes rather than a run configuring things differently.
        """
        if isinstance(deps, str):
            raise HookRegistryError(
                f"hook {name!r} on phase {phase.value!r}: {field_name}={deps!r} is a string; "
                f"pass a sequence of hook names, e.g. {field_name}=[{deps!r}]."
            )
        resolved: list[str | _HookRecord] = []
        for dep in deps:
            if isinstance(dep, str):
                resolved.append(dep)
                continue
            if not isinstance(dep, HookHandle):
                raise HookRegistryError(
                    f"hook {name!r} on phase {phase.value!r}: {field_name} entry {dep!r} is a "
                    f"{type(dep).__name__}; pass a hook name or the HookHandle that add() returned."
                )
            target = dep._record
            if dep._registry is not self:
                raise HookRegistryError(
                    f"hook {name!r} on phase {phase.value!r}: {field_name} holds a handle to "
                    f"{target.name!r} from a different registry, whose hooks this one never emits."
                )
            if target.phase is not phase:
                raise HookRegistryError(
                    f"hook {name!r} on phase {phase.value!r}: {field_name} holds a handle to "
                    f"{target.name!r} on phase {target.phase.value!r}. Constraints apply within one "
                    f"phase; cross-phase order is the phase order itself."
                )
            resolved.append(target)
        return tuple(resolved)

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

        Resolves the phase's constraints like :meth:`emit` does, so ``CLOSE`` reads back-to-front and
        a phase whose constraints form a cycle raises :class:`HookOrderError` here too — the error
        names the hooks in the cycle, which is what you were after. Decimated hooks are included
        regardless of where their counter sits.
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
            self._resolved[phase] = self._resolve_order(phase)
            self._stale.discard(phase)
        return self._resolved[phase]

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
            self._resolved[Phase.CLOSE] = tuple(self._live(Phase.CLOSE))
            return self._resolved[Phase.CLOSE]

    def _structural(self, phase: Phase) -> list[_HookRecord]:
        """The phase's hooks that still exist, enabled or not, in registration order."""
        return [record for record in self._records[phase] if not record.removed]

    def _live(self, phase: Phase) -> list[_HookRecord]:
        """The phase's hooks that would fire now, in registration order."""
        return [record for record in self._structural(phase) if record.enabled]

    def _resolve_order(self, phase: Phase) -> tuple[_HookRecord, ...]:
        """Topologically sort a phase's live hooks by their ``after``/``before`` constraints.

        Stable: among hooks whose constraints are already satisfied, the earliest-registered runs
        first, so an unconstrained phase emits exactly in registration order, and hooks a constraint
        displaces keep their order relative to each other.

        A disabled hook stays in the graph and only loses its turn to run, so a chain outlives one
        of its links being switched off: in ``a -> b -> c``, disabling ``b`` still emits ``a`` before
        ``c``. A constraint naming a hook that is absent or removed matches nothing, contributes no
        edge, and is otherwise ignored; it applies on its own if that hook shows up later.
        """
        live = self._structural(phase)
        positions: dict[str, list[int]] = defaultdict(list)
        by_record: dict[_HookRecord, int] = {}
        for position, record in enumerate(live):
            positions[record.name].append(position)
            by_record[record] = position

        successors: list[set[int]] = [set() for _ in live]
        blockers = [0] * len(live)

        def link(earlier: int, later: int) -> None:
            if earlier != later and later not in successors[earlier]:
                successors[earlier].add(later)
                blockers[later] += 1

        for position, record in enumerate(live):
            for dep in record.after:
                for target in _dep_positions(dep, positions, by_record):
                    link(target, position)
            for dep in record.before:
                for target in _dep_positions(dep, positions, by_record):
                    link(position, target)

        ready = [position for position in range(len(live)) if blockers[position] == 0]
        heapq.heapify(ready)
        order: list[_HookRecord] = []
        while ready:
            position = heapq.heappop(ready)
            order.append(live[position])
            # Iteration order doesn't matter: the heap decides which of the unblocked runs next.
            for later in successors[position]:
                blockers[later] -= 1
                if blockers[later] == 0:
                    heapq.heappush(ready, later)

        if len(order) != len(live):
            cycle = sorted({live[position].name for position in range(len(live)) if blockers[position] > 0})
            raise HookOrderError(
                f"Hook ordering cycle on phase {phase.value!r} among: {', '.join(cycle)}. "
                f"Their 'after'/'before' constraints cannot all hold."
            )
        return tuple(record for record in order if record.enabled)

    def _assert_mutable(self, phase: Phase) -> None:
        if phase in self._emitting:
            raise HookRegistryError(f"Cannot mutate hooks while emitting phase {phase.value!r}")

    @staticmethod
    def _default_name(callback: Callable[..., Any]) -> str:
        """Derive a name from where the callback is defined: ``module.QualName``.

        ``partial`` and decorators are unwrapped and a bound method reduces to its function, so every
        instance of a class derives the same name — one place in the source, however many hooks —
        while a subclass that overrides the method derives its own. A lambda or a closure also gets
        its definition line appended, because their qualnames don't identify them on their own.
        """
        target: Any = callback
        while isinstance(target, functools.partial):
            target = target.func
        target = getattr(inspect.unwrap(target), "__func__", inspect.unwrap(target))
        qualname = getattr(target, "__qualname__", None)
        if qualname is None:
            return repr(callback)
        module = getattr(target, "__module__", None)
        derived = f"{module}.{qualname}" if module else qualname
        code = getattr(target, "__code__", None)
        if "<" in qualname and code is not None:
            derived = f"{derived}:{code.co_firstlineno}"
        return derived
