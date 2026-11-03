"""Simulator state tensors, served as lazy reads into stable buffers.

A backend binds each state quantity once, in ``prepare_sim``, with :meth:`SimState.register`. Its read
returns the current value: a function returning a tensor (or a tuple of tensors, one per name), or a
:class:`Columns` spec, a column selection of a backend tensor. The registry copies each value into a
float32 buffer allocated on the first read, on the first access after the simulation state changed, and
``write`` flushes in-place changes back to the simulation, row by row.

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
from dataclasses import dataclass
from typing import Any, overload

import torch

Read = Callable[[], "torch.Tensor | tuple[torch.Tensor, ...]"]


class _Registration:
    __slots__ = ("buffers", "clean", "names", "read", "snapshots", "valid", "write")

    def __init__(self, names: tuple[str, ...], read: Read, write: Callable[..., None] | None) -> None:
        self.names = names
        self.read = read
        self.write = write
        self.buffers: tuple[torch.Tensor, ...] = ()
        self.snapshots: tuple[torch.Tensor, ...] = ()
        self.valid = False
        self.clean: tuple[int, ...] = ()

    def values(self) -> tuple[torch.Tensor, ...]:
        """The read's current values, one per name."""
        values = self.read()
        values = (values,) if isinstance(values, torch.Tensor) else tuple(values)
        if len(values) != len(self.names):
            raise ValueError(f"Read for {', '.join(self.names)} returned {len(values)} tensors")
        return values

    def refresh(self) -> None:
        """Copy the read's values into the buffers, allocating them on the first read."""
        values = self.values()
        if not self.buffers:
            self.buffers = tuple(torch.empty(value.shape, dtype=torch.float32, device=value.device) for value in values)
            # Writable registrations keep a copy of their buffers, to find changed rows.
            if self.write is not None:
                self.snapshots = tuple(torch.empty_like(buffer) for buffer in self.buffers)
        for buffer, value in zip(self.buffers, values, strict=True):
            buffer.copy_(value)
        self.mark_clean()

    def _versions(self) -> tuple[int, ...]:
        return tuple(buffer._version for buffer in self.buffers)

    def written(self) -> bool:
        """Whether a buffer was written in place since the last read or flush."""
        return self._versions() != self.clean

    def changed_rows(self) -> torch.Tensor:
        """Rows (environments) that differ from the last read or flush."""
        changed = torch.zeros(self.buffers[0].shape[0], dtype=torch.bool, device=self.buffers[0].device)
        for buffer, snapshot in zip(self.buffers, self.snapshots, strict=True):
            changed |= (buffer != snapshot).reshape(buffer.shape[0], -1).any(dim=1)
        return changed.nonzero().squeeze(1)

    def mark_clean(self) -> None:
        if self.snapshots:
            for buffer, snapshot in zip(self.buffers, self.snapshots, strict=True):
                snapshot.copy_(buffer)
        self.clean = self._versions()


@dataclass(frozen=True)
class Columns:
    """A read returning the columns ``ids`` of a backend tensor ``source`` (``[num_envs, n, ...]``).

    ``source`` may be a callable returning the current tensor, for backends that replace their storage.
    The value is moved to ``device`` when given. A contiguous ascending run of ids is returned as a slice,
    without a gather.
    """

    source: torch.Tensor | Callable[[], torch.Tensor]
    ids: Sequence[int]
    device: torch.device | str | None = None

    def __call__(self) -> torch.Tensor:
        data = self.source if isinstance(self.source, torch.Tensor) else self.source()
        run = contiguous_run(self.ids)
        if run is not None:
            value = data.narrow(1, run[0], run[1])
        else:
            value = data.index_select(1, torch.as_tensor(list(self.ids), dtype=torch.long, device=data.device))
        return value if self.device is None else value.to(self.device)


class SimState:
    """Registry of a simulator's state tensors and their pending writes."""

    def __init__(self) -> None:
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
        """Write the rows written in place to the simulation, marking every tensor stale if any were."""
        flushed = False
        for registration in self._registrations:
            if registration.valid and registration.written():
                if registration.write is None:
                    raise RuntimeError(f"Simulator state {', '.join(registration.names)} is read-only")
                rows = registration.changed_rows()
                if rows.numel():
                    registration.write(rows, *registration.buffers)
                    flushed = True
                registration.mark_clean()
        if flushed:
            self.mark_stale()

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
                names = ", ".join(registration.names)
                raise RuntimeError(f"Simulator state {names} was written in place after the simulation changed it")
            registration.refresh()
            registration.valid = True
        return registration.buffers[registration.names.index(name)]

    def names(self) -> list[str]:
        return [*self._by_name, *self._tensors]

    def read_fresh(self, name: str) -> torch.Tensor:
        """Return a fresh read of ``name`` without touching the served tensor."""
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
