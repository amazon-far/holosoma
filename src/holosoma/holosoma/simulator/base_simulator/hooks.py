from __future__ import annotations

import heapq
import inspect
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterator, Mapping, Sequence, Union, overload

from typing_extensions import Literal, Self

from holosoma.config_types.frequency import DecimationLike, is_frequency_string, resolve_decimation

HookDependency = Union["HookHandle", "HookRef", "_OptionalTarget"]
"""What ``after``/``before`` accept: a :class:`HookHandle`, a :class:`HookRef`, or :func:`if_present`
of either. Hook names are labels for diagnostics and never identify a dependency."""


@dataclass(frozen=True)
class HookRef:
    """Reference to a hook published by a configured plugin instance, resolved once all are installed.

    Lets a consumer depend on a provider it cannot hold: the provider publishes a member of its own
    ``Enum`` contract with ``add(..., key=...)``, and the consumer imports that enum and names the
    plugin slot. Matching is by enum type and member identity, never by the member's value or
    spelling, so two plugins can each define a ``PUBLISH`` without colliding.
    """

    instance: str
    """Plugin slot the provider is configured in, i.e. the key in ``--plugin.<key>:<preset>``."""

    key: Enum
    """Member of the contract the provider publishes."""


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
    owner: str | None = None
    """Plugin slot that registered this hook, or ``None`` for the framework's own."""
    key: Enum | None = None
    """Contract member this hook is published under, for :class:`HookRef` to resolve."""
    after: tuple[_HookRecord | HookRef | _OptionalTarget, ...] = field(default=(), repr=False)
    """Same-phase hooks this one must run after, as records or unresolved references."""
    before: tuple[_HookRecord | HookRef | _OptionalTarget, ...] = field(default=(), repr=False)
    """Same-phase hooks this one must run before, as records or unresolved references."""
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
    def key(self) -> Enum | None:
        """Contract member this hook was published under, if any."""
        return self._record.key

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


@dataclass(frozen=True)
class _OptionalTarget:
    """An ``after``/``before`` entry whose absence is acceptable; see :func:`if_present`.

    Holds a handle as given and the record it resolves to once :meth:`HookRegistry.add` has seen it.
    """

    target: HookHandle | HookRef | _HookRecord


def if_present(target: HookHandle | HookRef) -> _OptionalTarget:
    """Mark an ``after``/``before`` target optional, so the run starts even if nothing provides it.

    Every other target is required: a reference nothing on the phase is registered under fails the
    run rather than silently ordering nothing. Presence is about registration, not execution — a
    required target that is disabled satisfies the reference and simply contributes no edge.
    """
    return _OptionalTarget(target)


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
        self._published: dict[tuple[str, Enum], _HookRecord] = {}
        self._owner: str | None = None
        self._emitting: set[Phase] = set()
        self._closed = False
        self._base_rates: dict[Phase, float] = dict(base_rates or {})

    @overload
    def add(
        self,
        phase: Literal[Phase.FRAME_BEGIN, Phase.PRE_STEP, Phase.POST_STEP, Phase.FRAME_END],
        callback: FrameCallback,
        *,
        name: str,
        every: DecimationLike = 1,
        key: Enum | None = None,
        after: Sequence[HookDependency] = (),
        before: Sequence[HookDependency] = (),
    ) -> HookHandle: ...

    @overload
    def add(
        self,
        phase: Literal[Phase.EPISODE_END, Phase.EPISODE_START],
        callback: EpisodeCallback,
        *,
        name: str,
        every: DecimationLike = 1,
        key: Enum | None = None,
        after: Sequence[HookDependency] = (),
        before: Sequence[HookDependency] = (),
    ) -> HookHandle: ...

    @overload
    def add(
        self,
        phase: Literal[Phase.CLOSE],
        callback: CloseCallback,
        *,
        name: str,
        every: DecimationLike = 1,
        key: Enum | None = None,
        after: Sequence[HookDependency] = (),
        before: Sequence[HookDependency] = (),
    ) -> HookHandle: ...

    def add(
        self,
        phase: Phase,
        callback: Callable[..., Any],
        *,
        name: str,
        every: DecimationLike = 1,
        key: Enum | None = None,
        after: Sequence[HookDependency] = (),
        before: Sequence[HookDependency] = (),
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

        ``name`` labels the hook for :meth:`names`, cycle reports and teardown failures. Nothing
        orders against it, so it only has to tell a reader which hook this is.

        ``after`` / ``before`` order this hook against other hooks *on the same phase*. An entry is
        either a :class:`HookHandle`, which pins the one hook it came from, or a :class:`HookRef`
        for a hook another plugin publishes under a key. Hook names never identify a target: they are
        labels, so renaming one cannot change what anything orders against. Ordering is phase-scoped,
        and a handle or reference for another phase is rejected.

        Targets are required: one that nothing on the phase is registered under fails the run rather
        than silently ordering nothing, so wrap it in :func:`if_present` when a run without it is
        valid. Presence is about registration, not execution — a disabled target satisfies the
        reference but contributes no edge, so ordering always reflects the hooks that fire and a
        chain holds only while its links run. Unconstrained hooks keep their order relative to each
        other, and ``CLOSE`` prefers the latest-registered among them so teardown reverses
        construction. In every phase, including ``CLOSE``, ``before`` and ``after`` describe the
        order callbacks actually run in: a node's cleanup declares ``before=[runtime.close_hook]``.
        """
        self._assert_mutable(phase)
        owner = self._owner
        if key is not None:
            self._claim_key(phase, name, owner, key)
        self._check_arity(phase, callback, name)
        record = _HookRecord(
            phase=phase,
            callback=callback,
            name=name,
            every=self._resolve_every(phase, every, name),
            owner=owner,
            key=key,
            after=self._normalize_deps(after, "after", name, phase),
            before=self._normalize_deps(before, "before", name, phase),
        )
        if key is not None and owner is not None:
            self._published[(owner, key)] = record
        self._records[phase].append(record)
        self._invalidate(phase)
        return HookHandle(self, record)

    @contextmanager
    def installing(self, instance: str) -> Iterator[None]:
        """Scope registrations to the plugin slot being constructed, so its keys are published under it.

        ``install_plugins`` wraps each plugin's construction in this. Publishing a key outside a scope
        has no instance to attach to and is rejected, so a plugin that wants to be referenced by
        :class:`HookRef` must register while it is being constructed.
        """
        if self._owner is not None:
            raise HookRegistryError(
                f"Cannot install plugin slot {instance!r} while slot {self._owner!r} is still being "
                f"constructed; each instance owns its own registrations."
            )
        self._owner = instance
        try:
            yield
        finally:
            self._owner = None

    def _claim_key(self, phase: Phase, name: str, owner: str | None, key: Enum) -> None:
        """Reserve a contract member for one instance, so a reference to it cannot be ambiguous."""
        if owner is None:
            raise HookRegistryError(
                f"hook {name!r} on phase {phase.value!r} publishes {key!r} with no owning plugin "
                f"instance. Keys identify one instance's hook, so publish while the plugin is "
                f"constructed, or depend on this hook by its handle instead."
            )
        claimed = self._published.get((owner, key))
        if claimed is not None and not claimed.removed:
            raise HookRegistryError(
                f"hook {name!r} on phase {phase.value!r} publishes {key!r}, which plugin instance "
                f"{owner!r} already published as {claimed.name!r} on phase "
                f"{claimed.phase.value!r}. A key names one hook per instance."
            )

    def _normalize_deps(
        self, deps: Sequence[HookDependency], field_name: str, name: str, phase: Phase
    ) -> tuple[_HookRecord | HookRef | _OptionalTarget, ...]:
        """Freeze a constraint list, resolving each handle to the single hook it refers to.

        Every rejection here is something that would otherwise order against nothing at all, which an
        unmatched *name* is allowed to do silently: a bare string would order against its own
        letters, a callback or plugin object matches no hook, and a handle for another phase or
        another registry cannot be reached from this one. Whoever holds a handle already knows where
        it came from, so those are mistakes rather than a run configuring things differently.
        """
        # Checked at runtime, not just by the overloads: plenty of callers are untyped.
        entries: Sequence[Any] = deps
        if isinstance(entries, str):
            raise HookRegistryError(
                f"hook {name!r} on phase {phase.value!r}: {field_name}={deps!r} is a string. Hook "
                f"names are labels, not dependency targets; pass the HookHandle add() returned, or "
                f"a HookRef for a hook another plugin publishes."
            )
        resolved: list[_HookRecord | HookRef | _OptionalTarget] = []
        for entry in entries:
            optional = isinstance(entry, _OptionalTarget)
            dep = entry.target if isinstance(entry, _OptionalTarget) else entry
            if isinstance(dep, HookRef):
                # Stays as written: its provider may not be constructed yet.
                resolved.append(_OptionalTarget(dep) if optional else dep)
                continue
            if isinstance(dep, str):
                raise HookRegistryError(
                    f"hook {name!r} on phase {phase.value!r}: {field_name} entry {dep!r} is a hook "
                    f"name. Hook names are labels, not dependency targets; pass the HookHandle "
                    f"add() returned, or a HookRef for a hook another plugin publishes."
                )
            if not isinstance(dep, HookHandle):
                raise HookRegistryError(
                    f"hook {name!r} on phase {phase.value!r}: {field_name} entry {dep!r} is a "
                    f"{type(dep).__name__}; pass the HookHandle add() returned, or a HookRef."
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
            resolved.append(_OptionalTarget(target) if optional else target)
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
        """Run due hooks in resolved order; CLOSE runs once and stops at the first failure."""
        self._validate_payload(phase, args)

        if phase is Phase.CLOSE:
            if self._closed:
                return

            if phase in self._emitting:
                raise HookRegistryError("Recursive close hook emission is not supported")

            self._closed = True
            self._emitting.add(phase)
            try:
                for record in self._order(phase):
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

    def validate(self) -> None:
        """Resolve every phase now, without invoking a callback, so a bad graph fails at startup.

        Raises :class:`HookOrderError` on a cycle or a required target nothing is registered under.
        Hooks registered afterwards are validated when their phase next resolves, so calling this
        early does not freeze the registry.
        """
        for phase in Phase:
            self._order(phase)

    def names(self, phase: Phase) -> tuple[str, ...]:
        """Names of the hooks that would fire on ``phase`` now, in the order they would run.

        Resolves the phase's constraints like :meth:`emit` does, so a cycle or an unsatisfied
        required target raises here too. Decimated hooks are included regardless of where their
        counter sits.
        """
        return tuple(record.name for record in self._order(phase))

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

    def _live(self, phase: Phase) -> list[_HookRecord]:
        """The phase's hooks that would fire now, in registration order."""
        return [record for record in self._records[phase] if record.enabled and not record.removed]

    def _resolve_order(self, phase: Phase) -> tuple[_HookRecord, ...]:
        """Topologically sort a phase's live hooks by their ``after``/``before`` constraints.

        Stable: among hooks whose constraints are already satisfied, the earliest-registered runs
        first, so an unconstrained phase emits exactly in registration order, and hooks a constraint
        displaces keep their order relative to each other.

        ``CLOSE`` breaks ties the other way, running the latest-registered ready hook first, so
        teardown reverses construction without anyone declaring it.

        Ordering only ever reflects hooks that fire: a target that is absent, disabled or removed
        contributes no edge, so in ``a -> b -> c`` disabling ``b`` leaves ``a`` and ``c`` unordered
        and a hook that needs ``a`` first has to say so. Requiredness is separate and about
        registration — a target nothing is registered under raises unless it was wrapped in
        :func:`if_present`, while a registered-but-disabled target satisfies the reference.
        """
        live = self._live(phase)
        by_record = {record: position for position, record in enumerate(live)}

        successors: list[set[int]] = [set() for _ in live]
        blockers = [0] * len(live)

        def link(earlier: int, later: int) -> None:
            if earlier != later and later not in successors[earlier]:
                successors[earlier].add(later)
                blockers[later] += 1

        for position, record in enumerate(live):
            for dep in record.after:
                for target in self._targets(phase, record, dep, by_record):
                    link(target, position)
            for dep in record.before:
                for target in self._targets(phase, record, dep, by_record):
                    link(position, target)

        # CLOSE tears down in reverse, so it prefers the latest-registered hook among the ready ones.
        sign = -1 if phase is Phase.CLOSE else 1
        ready = [sign * position for position in range(len(live)) if blockers[position] == 0]
        heapq.heapify(ready)
        order: list[_HookRecord] = []
        while ready:
            position = sign * heapq.heappop(ready)
            order.append(live[position])
            # Iteration order doesn't matter: the heap decides which of the unblocked runs next.
            for later in successors[position]:
                blockers[later] -= 1
                if blockers[later] == 0:
                    heapq.heappush(ready, sign * later)

        if len(order) != len(live):
            cycle = sorted({live[position].name for position in range(len(live)) if blockers[position] > 0})
            raise HookOrderError(
                f"Hook ordering cycle on phase {phase.value!r} among: {', '.join(cycle)}. "
                f"Their 'after'/'before' constraints cannot all hold."
            )
        return tuple(order)

    def _targets(
        self,
        phase: Phase,
        record: _HookRecord,
        dep: _HookRecord | HookRef | _OptionalTarget,
        by_record: Mapping[_HookRecord, int],
    ) -> Sequence[int]:
        """Live position a constraint points at, raising when a required target is not registered."""
        optional = isinstance(dep, _OptionalTarget)
        if isinstance(dep, _OptionalTarget):
            target = dep.target
            dep = target._record if isinstance(target, HookHandle) else target

        if isinstance(dep, HookRef):
            published = self._published.get((dep.instance, dep.key))
            if published is None or published.removed:
                if optional:
                    return ()
                raise HookOrderError(
                    f"Hook {record.name!r} on phase {phase.value!r} requires {dep.key!r} from plugin "
                    f"instance {dep.instance!r}, which published no such hook. Configured "
                    f"publications: {sorted((owner, str(k)) for owner, k in self._published)}. Wrap "
                    f"it in if_present(...) if a run without it is valid."
                )
            if published.phase is not phase:
                raise HookOrderError(
                    f"Hook {record.name!r} on phase {phase.value!r} orders against {dep.key!r} from "
                    f"{dep.instance!r}, which is published on phase {published.phase.value!r}. "
                    f"Constraints apply within one phase."
                )
            dep = published

        if dep.removed and not optional:
            raise HookOrderError(
                f"Hook {record.name!r} on phase {phase.value!r} requires {dep.name!r}, which has "
                f"been removed. Wrap it in if_present(...) if a run without it is valid."
            )
        position = by_record.get(dep)
        return () if position is None else (position,)

    def _assert_mutable(self, phase: Phase) -> None:
        if phase in self._emitting:
            raise HookRegistryError(f"Cannot mutate hooks while emitting phase {phase.value!r}")
