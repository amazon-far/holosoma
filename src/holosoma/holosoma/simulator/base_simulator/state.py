"""Simulator state tensors, served as lazy reads into stable buffers.

A backend binds each state quantity once, in ``prepare_sim``, with :meth:`SimState.register`. It
serves preallocated buffers that a read fills on the first access after the simulation state
changed, and that ``write`` flushes to the simulation, row by row. The read is either a function that
fills the given buffers (several names may share one read), or a :class:`Columns` spec, a column
selection of a backend tensor that builds its own read and buffer.

Every served tensor keeps its identity, so references to it stay valid. Reads see the current
simulation state. In-place writes reach the simulation before its next physics step, FK pass, refresh
or any state write that would replace them: the changed rows (environments) are flushed through the
tensor's ``write``. A tensor bound without ``write`` is read-only.

Backends wrap every state mutation in :meth:`SimState.mutation`, which flushes pending writes first and
marks every tensor stale after.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, overload

import torch


class _LazyGroup:
    __slots__ = ("buffers", "clean", "names", "read", "snapshots", "valid", "write")

    def __init__(
        self,
        names: tuple[str, ...],
        read: Callable[..., None],
        write: Callable[..., None] | None,
        buffers: tuple[torch.Tensor, ...],
    ) -> None:
        self.names = names
        self.read = read
        self.write = write
        self.buffers = buffers
        self.valid = False
        # Writable groups keep a copy of each buffer that is not a view of another, to find changed rows.
        self.snapshots = None if write is None else tuple(buffer.clone() for buffer in buffers if buffer._base is None)
        self.clean = self._versions()

    def _versions(self) -> tuple[int, ...]:
        return tuple(buffer._version for buffer in self.buffers)

    def written(self) -> bool:
        """Whether a buffer was written in place since the last read or flush."""
        return self._versions() != self.clean

    def changed_rows(self) -> torch.Tensor:
        """Rows (environments) that differ from the last read or flush."""
        assert self.snapshots is not None
        bases = [buffer for buffer in self.buffers if buffer._base is None]
        changed = torch.zeros(bases[0].shape[0], dtype=torch.bool, device=bases[0].device)
        for buffer, snapshot in zip(bases, self.snapshots, strict=True):
            changed |= (buffer != snapshot).reshape(buffer.shape[0], -1).any(dim=1)
        return changed.nonzero().squeeze(1)

    def mark_clean(self) -> None:
        if self.snapshots is not None:
            bases = [buffer for buffer in self.buffers if buffer._base is None]
            for buffer, snapshot in zip(bases, self.snapshots, strict=True):
                snapshot.copy_(buffer)
        self.clean = self._versions()


@dataclass(frozen=True)
class Columns:
    """The columns ``ids`` of a backend tensor ``source`` (``[num_envs, n, ...]``), read into a float32 buffer.

    ``source`` may be a callable returning the current tensor, for backends that replace their storage.
    The buffer is on ``device`` (default: the source's). A contiguous ascending run of ids is copied as a
    slice, without a gather.
    """

    source: torch.Tensor | Callable[[], torch.Tensor]
    ids: Sequence[int]
    device: torch.device | str | None = None

    def reader(self) -> tuple[Callable[[torch.Tensor], None], torch.Tensor]:
        """A read function and the buffer it fills."""
        source = self.source

        def current() -> torch.Tensor:
            return source if isinstance(source, torch.Tensor) else source()

        sample = current()
        ids = list(self.ids)
        shape = (sample.shape[0], len(ids), *sample.shape[2:])
        device = self.device if self.device is not None else sample.device
        buffer = torch.zeros(shape, dtype=torch.float32, device=device)
        run = contiguous_run(ids)
        ids_t = torch.as_tensor(ids, dtype=torch.long, device=sample.device)

        def read(out: torch.Tensor) -> None:
            data = current()
            out.copy_(data.narrow(1, run[0], run[1]) if run is not None else data.index_select(1, ids_t))

        return read, buffer


class SimState:
    """Registry of a simulator's state tensors and their pending writes."""

    def __init__(self) -> None:
        self._tensors: dict[str, torch.Tensor] = {}
        self._groups: dict[str, _LazyGroup] = {}
        self._group_list: list[_LazyGroup] = []

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
        read: Callable[..., None],
        buffers: torch.Tensor | Sequence[torch.Tensor],
        *,
        write: Callable[..., None] | None = None,
    ) -> None: ...

    def register(
        self,
        names: str | Sequence[str],
        read: Columns | Callable[..., None],
        buffers: torch.Tensor | Sequence[torch.Tensor] | None = None,
        *,
        write: Callable[..., None] | None = None,
    ) -> None:
        """Serve buffers as ``names``, filled by ``read`` after a state change and flushed by ``write(rows, *buffers)``.

        ``read`` is a :class:`Columns` spec, which builds its own buffer, or a function ``read(*buffers)``
        that fills ``buffers``. Without ``write`` the tensors are read-only.
        """
        if isinstance(read, Columns):
            if buffers is not None:
                raise TypeError("A Columns spec builds its own buffer; do not pass buffers.")
            read, buffers = read.reader()
        elif buffers is None:
            raise TypeError("A read function needs the buffers it fills.")
        names = (names,) if isinstance(names, str) else tuple(names)
        buffers = (buffers,) if isinstance(buffers, torch.Tensor) else tuple(buffers)
        group = _LazyGroup(names, read, write, buffers)
        for name, buffer in zip(names, buffers, strict=True):
            self._unbind(name)
            self._tensors[name] = buffer
            self._groups[name] = group
        self._group_list.append(group)

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
        for group in self._group_list:
            if group.valid and group.written():
                if group.write is None:
                    raise RuntimeError(f"Simulator state {', '.join(group.names)} is read-only")
                rows = group.changed_rows()
                if rows.numel():
                    group.write(rows, *group.buffers)
                    flushed = True
                group.mark_clean()
        if flushed:
            self.mark_stale()

    def mark_stale(self) -> None:
        """Mark every tensor stale, so the next access re-reads it."""
        for group in self._group_list:
            group.valid = False

    def get(self, name: str) -> Any:
        try:
            tensor = self._tensors[name]
        except KeyError:
            raise AttributeError(f"Simulator state {name!r} is not bound") from None
        group = self._groups.get(name)
        if group is not None and not group.valid:
            if group.written():
                raise RuntimeError(
                    f"Simulator state {', '.join(group.names)} was written in place after the simulation changed it"
                )
            group.read(*group.buffers)
            group.mark_clean()
            group.valid = True
        return tensor

    def names(self) -> list[str]:
        return list(self._tensors)

    def read_fresh(self, name: str) -> torch.Tensor:
        """Return a fresh read of ``name`` without touching the served tensor."""
        group = self._groups.get(name)
        if group is None:
            return torch.as_tensor(self._tensors[name][:]).clone()
        scratch = _scratch_like(group.buffers)
        group.read(*scratch)
        return scratch[group.names.index(name)]

    def _unbind(self, name: str) -> None:
        group = self._groups.pop(name, None)
        if group is not None and group not in self._groups.values():
            self._group_list.remove(group)


def _scratch_like(buffers: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    """Fresh tensors laid out like ``buffers``, keeping views of an earlier buffer as views."""
    scratch: list[torch.Tensor] = []
    for i, buffer in enumerate(buffers):
        base = next(
            (j for j in range(i) if buffers[j].untyped_storage().data_ptr() == buffer.untyped_storage().data_ptr()),
            None,
        )
        if base is None:
            scratch.append(
                torch.empty_strided(buffer.size(), buffer.stride(), dtype=buffer.dtype, device=buffer.device)
            )
        else:
            offset = scratch[base].storage_offset() + buffer.storage_offset() - buffers[base].storage_offset()
            scratch.append(scratch[base].as_strided(buffer.size(), buffer.stride(), offset))
    return tuple(scratch)


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
