"""Simulator state tensors, served as live backend views or as lazy reads into stable buffers.

A backend binds each state quantity once, in ``prepare_sim``:

- :meth:`SimState.live` serves a tensor as is: a zero-copy view of backend storage, or a buffer the
  backend fills itself.
- :meth:`SimState.lazy` serves preallocated buffers that ``read`` fills on the first access after
  any of the group's sources changed, and that ``write`` flushes to the simulation, row by row. Several
  names may share one read.
- :meth:`SimState.columns` picks between the two for a column selection: a live ``narrow`` view when
  the selection is a contiguous ascending run of a stable float32 tensor, else a lazy gather.

Every served tensor keeps its identity, so references to it stay valid. Reads see the current
simulation state. In-place writes reach the simulation before its next physics step, FK pass, refresh
or any state write that would replace them: a live view writes backend storage directly, and a lazy
buffer's changed rows (environments) are flushed through its ``write``. A lazy tensor bound without
``write`` is read-only.

Backends bracket every state mutation with :meth:`SimState.prepare` (flush pending writes it depends
on or would replace) and :meth:`SimState.changed` (mark what it touched stale), naming the sources it
touches.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal, overload

import torch

Source = Literal["joints", "root", "bodies", "contacts"]
SOURCES: tuple[Source, ...] = ("joints", "root", "bodies", "contacts")


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
        self.snapshots = (
            None if write is None else tuple(buffer.clone() for buffer in buffers if buffer._base is None)
        )
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


class SimState:
    """Registry of a simulator's state tensors, the sources each one reads, and its pending writes."""

    def __init__(self) -> None:
        self._tensors: dict[str, torch.Tensor] = {}
        self._groups: dict[str, _LazyGroup] = {}
        self._by_source: dict[Source, list[_LazyGroup]] = {source: [] for source in SOURCES}

    def live(self, name: str, tensor: Any) -> None:
        """Serve ``tensor`` as ``name`` with no caching."""
        self._unbind(name)
        self._tensors[name] = tensor

    def lazy(
        self,
        names: str | Sequence[str],
        read: Callable[..., None],
        buffers: torch.Tensor | Sequence[torch.Tensor],
        *,
        sources: Sequence[Source],
        write: Callable[..., None] | None = None,
    ) -> None:
        """Serve ``buffers`` as ``names``; ``read(*buffers)`` fills them, ``write(rows, *buffers)`` flushes ``rows``."""
        names = (names,) if isinstance(names, str) else tuple(names)
        buffers = (buffers,) if isinstance(buffers, torch.Tensor) else tuple(buffers)
        group = _LazyGroup(names, read, write, buffers)
        for name, buffer in zip(names, buffers, strict=True):
            self._unbind(name)
            self._tensors[name] = buffer
            self._groups[name] = group
        for source in sources:
            self._by_source[source].append(group)

    def columns(
        self,
        name: str,
        source: torch.Tensor | Callable[[], torch.Tensor],
        ids: Sequence[int],
        *,
        sources: Sequence[Source],
        device: torch.device | str | None = None,
        write: Callable[[torch.Tensor, torch.Tensor], None] | None = None,
    ) -> None:
        """Serve ``source[:, ids]`` as ``name``, live when possible.

        ``source`` is a ``[num_envs, n, ...]`` tensor, or a callable returning the current one when the
        backend replaces its storage (always read lazily). The served tensor is float32 on ``device``
        (default: the source's). ``write(rows, buffer)`` flushes the lazy buffer; a live view writes ``source``.
        """
        if isinstance(source, torch.Tensor):
            view = contiguous_columns(source, ids)
            if view is not None and (device is None or _same_device(view.device, device)):
                self.live(name, view)
                return
            tensor = source

            def current() -> torch.Tensor:
                return tensor

        else:
            current = source
        sample = current()
        ids_t = torch.as_tensor(list(ids), dtype=torch.long, device=sample.device)
        shape = (sample.shape[0], len(ids), *sample.shape[2:])
        buffer = torch.zeros(shape, dtype=torch.float32, device=device if device is not None else sample.device)

        def read(out: torch.Tensor) -> None:
            out.copy_(current().index_select(1, ids_t))

        self.lazy(name, read, buffer, sources=sources, write=write)

    def prepare(self, *sources: Source, everything: bool = False) -> None:
        """Flush the rows written in place before a mutation of ``sources`` (no arguments, or ``everything``: all)."""
        groups = self._all_groups() if everything or not sources else self._groups_of(sources)
        flushed = False
        for group in groups:
            if group.valid and group.written():
                if group.write is None:
                    raise RuntimeError(f"Simulator state {', '.join(group.names)} is read-only")
                rows = group.changed_rows()
                if rows.numel():
                    group.write(rows, *group.buffers)
                    flushed = True
                group.mark_clean()
        if flushed:
            self.changed("bodies")

    def changed(self, *sources: Source) -> None:
        """Mark tensors reading ``sources`` stale (no arguments: every lazy tensor)."""
        for group in self._groups_of(sources) if sources else self._all_groups():
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

    def is_live(self, name: str) -> bool:
        return name in self._tensors and name not in self._groups

    def read_fresh(self, name: str) -> torch.Tensor:
        """Return a fresh read of ``name`` without touching the served tensor."""
        group = self._groups.get(name)
        if group is None:
            return torch.as_tensor(self._tensors[name][:]).clone()
        scratch = _scratch_like(group.buffers)
        group.read(*scratch)
        return scratch[group.names.index(name)]

    def _all_groups(self) -> list[_LazyGroup]:
        return list({id(group): group for group in self._groups.values()}.values())

    def _groups_of(self, sources: Sequence[Source]) -> list[_LazyGroup]:
        return list({id(group): group for source in sources for group in self._by_source[source]}.values())

    def _unbind(self, name: str) -> None:
        group = self._groups.pop(name, None)
        if group is not None and group not in self._groups.values():
            for groups in self._by_source.values():
                if group in groups:
                    groups.remove(group)


def _same_device(a: torch.device, b: torch.device | str) -> bool:
    return bool(a == torch.empty(0, device=b).device)


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


def contiguous_columns(source: torch.Tensor, ids: Sequence[int]) -> torch.Tensor | None:
    """Return ``source[:, ids]`` as a view when ``ids`` is a contiguous ascending run, else ``None``."""
    ids = list(ids)
    if source.dtype != torch.float32 or not ids or ids != list(range(ids[0], ids[0] + len(ids))):
        return None
    return source.narrow(1, ids[0], len(ids))


class StateTensor:
    """A simulator state quantity, served by the simulator's :class:`SimState`.

    Assigning binds the value as a live tensor.
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
        sim_state(sim).live(self._name, value)


def sim_state(sim: Any) -> SimState:
    """Return ``sim``'s state registry, creating it for simulators built without ``__init__``."""
    state: SimState | None = vars(sim).get("state")
    if state is None:
        state = vars(sim)["state"] = SimState()
    return state
