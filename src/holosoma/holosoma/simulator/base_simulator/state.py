"""Simulator state tensors, served as lazy reads into stable buffers.

A backend binds each state quantity once, in ``prepare_sim``, with :meth:`SimState.register`. Its read
returns the current value: a function returning a tensor (or a tuple of tensors, one per name), or a
:class:`Columns` spec, a column selection of a backend tensor. The registry copies each value into a
float32 buffer on ``device``, allocated on the first read, on the first access after the simulation
state changed, and ``write`` flushes in-place changes back to the simulation, row by row.

Every served tensor keeps its identity, so references to it stay valid. Reads see the current
simulation state. In-place writes reach the simulation before its next physics step, FK pass or any
state write that would replace them: the changed rows (environments) are flushed through the tensor's
``write``. A tensor bound without ``write`` is read-only.

Backends wrap every state mutation in :meth:`SimState.mutation`, which flushes pending writes first and
marks every tensor stale after.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, overload

import torch

Read = Callable[[], "torch.Tensor | tuple[torch.Tensor, ...]"]


class _Registration:
    __slots__ = ("buffers", "clean", "names", "read", "valid", "write")

    def __init__(self, names: tuple[str, ...], read: Read, write: Callable[..., None] | None) -> None:
        self.names = names
        self.read = read
        self.write = write
        self.buffers: tuple[torch.Tensor, ...] = ()
        self.valid = False
        self.clean: tuple[int, ...] = ()

    def values(self) -> tuple[torch.Tensor, ...]:
        """The read's current values, one per name."""
        values = self.read()
        values = (values,) if isinstance(values, torch.Tensor) else tuple(values)
        if len(values) != len(self.names):
            raise ValueError(f"Read for {', '.join(self.names)} returned {len(values)} tensors")
        return values

    def refresh(self, device: torch.device | str | None) -> None:
        """Copy the read's values into the buffers, allocating them on ``device`` (the read's, if None) first."""
        values = self.values()
        if not self.buffers:
            # Normal tensors even when first read inside inference mode: they are written in place later.
            with torch.inference_mode(False):
                self.buffers = tuple(
                    torch.empty(value.shape, dtype=torch.float32, device=device or value.device) for value in values
                )
        for buffer, value in zip(self.buffers, values, strict=True):
            buffer.copy_(value)
        self.mark_clean()

    def _versions(self) -> tuple[int, ...]:
        return tuple(buffer._version for buffer in self.buffers)

    def written(self) -> bool:
        """Whether a buffer was written in place since the last read or flush."""
        return self._versions() != self.clean

    def changed_rows(self) -> torch.Tensor:
        """Rows (environments) whose buffer differs from the simulation's current values."""
        changed = torch.zeros(self.buffers[0].shape[0], dtype=torch.bool, device=self.buffers[0].device)
        for buffer, value in zip(self.buffers, self.values(), strict=True):
            current = value.to(buffer)
            differs = (buffer != current) & ~(buffer.isnan() & current.isnan())
            changed |= differs.reshape(buffer.shape[0], -1).any(dim=1)
        return changed.nonzero().squeeze(1)

    def mark_clean(self) -> None:
        self.clean = self._versions()


@dataclass(frozen=True)
class Columns:
    """A read returning the columns ``ids`` of a backend tensor ``source`` (``[num_envs, n, ...]``).

    ``source`` may be a callable returning the current tensor, for backends that replace their storage.
    A contiguous ascending run of ids is returned as a slice, without a gather.
    """

    source: torch.Tensor | Callable[[], torch.Tensor]
    ids: Sequence[int]
    _run: tuple[int, int] | None = field(init=False, repr=False, compare=False)
    _index: torch.Tensor | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_run", contiguous_run(self.ids))

    def __call__(self) -> torch.Tensor:
        data = self.source if isinstance(self.source, torch.Tensor) else self.source()
        if self._run is not None:
            return data.narrow(1, *self._run)
        index = self._index
        if index is None or index.device != data.device:
            index = torch.as_tensor(list(self.ids), dtype=torch.long, device=data.device)
            object.__setattr__(self, "_index", index)
        return data.index_select(1, index)


class SimState:
    """Registry of a simulator's state tensors and their pending writes."""

    def __init__(self, device: torch.device | str | None = None) -> None:
        self.device = device
        self._tensors: dict[str, torch.Tensor] = {}
        self._by_name: dict[str, _Registration] = {}
        self._registrations: list[_Registration] = []

    @overload
    def register(
        self,
        names: str,
        read: Columns,
        *,
        write: Callable[[torch.Tensor, torch.Tensor], None] | None = None,
    ) -> None: ...

    @overload
    def register(
        self,
        names: str | Sequence[str],
        read: Read,
        *,
        write: Callable[..., None] | None = None,
    ) -> None: ...

    def register(
        self,
        names: str | Sequence[str],
        read: Columns | Read,
        *,
        write: Callable[..., None] | None = None,
    ) -> None:
        """Serve ``read``'s values as ``names``, re-read after a state change and flushed by ``write(rows, *buffers)``.

        ``read`` returns one tensor per name. Without ``write`` the tensors are read-only.
        """
        names = (names,) if isinstance(names, str) else tuple(names)
        registration = _Registration(names, read, write)
        for name in names:
            self._unbind(name)
            self._by_name[name] = registration
        self._registrations.append(registration)

    def _serve(self, name: str, tensor: Any) -> None:
        """Serve ``tensor`` as ``name`` as is, outside the registry's reads and writes."""
        self._unbind(name)
        self._tensors[name] = tensor

    @contextmanager
    def mutation(self) -> Iterator[None]:
        """Wrap a mutation of the simulation state: flush pending writes first, mark every tensor stale after."""
        self.flush_writes()
        try:
            yield
        finally:
            self.mark_stale()

    def flush_writes(self) -> None:
        """Write the rows written in place to the simulation, then mark every other tensor stale if any were.

        A flushed tensor stays valid: its buffer already holds what the simulation now has.
        """
        written = [
            registration for registration in self._registrations if registration.valid and registration.written()
        ]
        read_only = [registration for registration in written if registration.write is None]
        if read_only:
            for registration in read_only:  # discard the write, so the next access re-reads
                registration.mark_clean()
                registration.valid = False
            names = ", ".join(name for registration in read_only for name in registration.names)
            raise RuntimeError(f"Simulator state {names} is read-only")
        flushed = False
        # Compare every written tensor with the simulation before writing any of them back.
        for registration, rows in [(registration, registration.changed_rows()) for registration in written]:
            if rows.numel():
                assert registration.write is not None
                registration.write(rows, *registration.buffers)
                flushed = True
            registration.mark_clean()
        if flushed:
            for registration in self._registrations:
                if registration not in written:
                    registration.valid = False

    def mark_stale(self) -> None:
        """Mark every tensor stale, so the next access re-reads it."""
        for registration in self._registrations:
            registration.valid = False

    def get(self, name: str) -> Any:
        registration = self._by_name.get(name)
        if registration is None:
            try:
                return self._tensors[name]
            except KeyError:
                raise AttributeError(f"Simulator state {name!r} is not bound") from None
        if not registration.valid:
            if registration.written():
                registration.mark_clean()  # discard the write, so the next access re-reads
                names = ", ".join(registration.names)
                raise RuntimeError(f"Simulator state {names} was written in place after the simulation changed it")
            registration.refresh(self.device)
            registration.valid = True
        return registration.buffers[registration.names.index(name)]

    def names(self) -> list[str]:
        return [*self._by_name, *self._tensors]

    def _read_fresh(self, name: str) -> torch.Tensor:
        """Return a fresh read of ``name`` without touching the served tensor, for consistency checks."""
        registration = self._by_name.get(name)
        if registration is None:
            return torch.as_tensor(self._tensors[name][:]).clone()
        return registration.values()[registration.names.index(name)].to(torch.float32).clone()

    def _unbind(self, name: str) -> None:
        self._tensors.pop(name, None)
        registration = self._by_name.pop(name, None)
        if registration is not None and registration not in self._by_name.values():
            self._registrations.remove(registration)


def contiguous_run(ids: Sequence[int]) -> tuple[int, int] | None:
    """``(start, length)`` when ``ids`` is a contiguous ascending run, else ``None``."""
    ids = list(ids)
    if not ids or ids != list(range(ids[0], ids[0] + len(ids))):
        return None
    return ids[0], len(ids)


class StateTensor:
    """A simulator state quantity, served by the simulator's :class:`SimState`.

    Assigning serves the value as is, for setup placeholders and test fakes.
    """

    def __init__(self, doc: str) -> None:
        self.__doc__ = doc

    def __set_name__(self, owner: type, name: str) -> None:
        self._name = name

    @overload
    def __get__(self, sim: None, owner: type | None = None) -> StateTensor: ...

    @overload
    def __get__(self, sim: object, owner: type | None = None) -> torch.Tensor: ...

    def __get__(self, sim: Any, owner: type | None = None) -> Any:
        if sim is None:
            return self
        return sim_state(sim).get(self._name)

    def __set__(self, sim: Any, value: Any) -> None:
        sim_state(sim)._serve(self._name, value)


def sim_state(sim: Any) -> SimState:
    """Return ``sim``'s state registry, creating it for simulators built without ``__init__``."""
    state: SimState | None = vars(sim).get("state")
    if state is None:
        state = vars(sim)["state"] = SimState()
    return state
