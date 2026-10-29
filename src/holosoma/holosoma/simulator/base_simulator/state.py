"""Simulator state tensors, served as live backend views or as lazy reads into stable buffers.

A backend binds each state quantity once, in ``prepare_sim``:

- :meth:`SimState.live` serves a tensor as is: a zero-copy view of backend storage, or a buffer the
  backend fills itself.
- :meth:`SimState.lazy` serves preallocated buffers that ``read`` fills on the first access after
  any of the group's sources changed. Several names may share one read.
- :meth:`SimState.columns` picks between the two for a column selection: a live ``narrow`` view when
  the selection is a contiguous ascending run of a stable float32 tensor, else a lazy gather.

The backend reports state changes with :meth:`SimState.changed`, naming the sources a mutation
touched. Every served tensor keeps its identity, so references to it stay valid. In-place writes into
a lazy buffer last until one of its sources changes; flush them to the simulation first.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal, overload

import torch

Source = Literal["joints", "root", "bodies", "contacts"]
SOURCES: tuple[Source, ...] = ("joints", "root", "bodies", "contacts")


class _LazyGroup:
    __slots__ = ("buffers", "names", "read", "valid")

    def __init__(self, names: tuple[str, ...], read: Callable[..., None], buffers: tuple[torch.Tensor, ...]) -> None:
        self.names = names
        self.read = read
        self.buffers = buffers
        self.valid = False


class SimState:
    """Registry of a simulator's state tensors and the sources each one reads."""

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
    ) -> None:
        """Serve ``buffers`` as ``names``; ``read(*buffers)`` fills them in place when stale."""
        names = (names,) if isinstance(names, str) else tuple(names)
        buffers = (buffers,) if isinstance(buffers, torch.Tensor) else tuple(buffers)
        group = _LazyGroup(names, read, buffers)
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
    ) -> None:
        """Serve ``source[:, ids]`` as ``name``, live when possible.

        ``source`` is a ``[num_envs, n, ...]`` tensor, or a callable returning the current one when the
        backend replaces its storage (always read lazily). A lazy buffer is float32 on ``device``
        (default: the source's).
        """
        if isinstance(source, torch.Tensor):
            view = contiguous_columns(source, ids)
            if view is not None:
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

        self.lazy(name, read, buffer, sources=sources)

    def changed(self, *sources: Source) -> None:
        """Mark tensors reading ``sources`` stale; no arguments marks every lazy tensor stale."""
        for source in sources or SOURCES:
            for group in self._by_source[source]:
                group.valid = False

    def get(self, name: str) -> Any:
        try:
            tensor = self._tensors[name]
        except KeyError:
            raise AttributeError(f"Simulator state {name!r} is not bound") from None
        group = self._groups.get(name)
        if group is not None and not group.valid:
            group.read(*group.buffers)
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
        scratch = tuple(torch.empty_like(buffer) for buffer in group.buffers)
        group.read(*scratch)
        return scratch[group.names.index(name)]

    def _unbind(self, name: str) -> None:
        group = self._groups.pop(name, None)
        if group is not None and group not in self._groups.values():
            for groups in self._by_source.values():
                if group in groups:
                    groups.remove(group)


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
