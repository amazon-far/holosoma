"""Simulator state tensors, served as lazy reads into stable buffers.

A backend binds each state quantity once, in ``prepare_sim``, with :meth:`SimState.register`. Its read
returns the current value: a function returning a tensor (or a tuple of tensors, one per name), or a
:class:`Columns` spec, a column selection of a backend tensor. The registry copies each value into a
float32 buffer on ``device``, allocated on the first read, on the first access after the simulation
state changed, and ``write`` flushes in-place changes back to the simulation, row by row.

Every served tensor keeps its identity, so references to it stay valid. Reads see the current
simulation state. In-place writes reach the simulation before its next physics step, FK pass or any
state write that would replace them: the changed rows (environments) are flushed through the tensor's
``write``. A tensor bound without ``write`` is read-only. :meth:`SimState.link` serves writable tensors whose memory is
written from outside: they are never re-read, and their changed rows flush like in-place writes.

Fields go stale by tag. Each is invalidated by its own name and by its ``invalidated_by`` tags; by default
writable fields by :data:`DYNAMICS` and read-only (derived) fields by :data:`DYNAMICS` and :data:`KINEMATICS`.
Backends wrap every state mutation in :meth:`SimState.mutation`, which flushes pending writes first and fires
its tag after: :data:`DYNAMICS` when the simulation state changes, :data:`KINEMATICS` when only quantities
derived from it are recomputed. A flush that writes rows of a field fires that field's ``invalidates`` tags
(by default :data:`KINEMATICS` for writable fields).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, overload

import torch

Read = Callable[[], "torch.Tensor | tuple[torch.Tensor, ...]"]
BufferAllocator = Callable[[str, torch.Tensor, torch.device | str | None], torch.Tensor]

DYNAMICS = "dynamics"
"""Fired when the simulation state changes: a physics step or a state write."""
KINEMATICS = "kinematics"
"""Fired when quantities derived from the simulation's inputs change: an FK pass, or a flushed input write."""


class _Registration:
    __slots__ = ("buffers", "clean", "invalidated_by", "invalidates", "linked", "names", "read", "valid", "write")

    def __init__(
        self,
        names: tuple[str, ...],
        read: Read,
        write: Callable[..., None] | None,
        invalidated_by: frozenset[str],
        invalidates: frozenset[str],
    ) -> None:
        self.names = names
        self.read = read
        self.write = write
        self.invalidated_by = invalidated_by | frozenset(names)
        self.invalidates = invalidates
        self.buffers: tuple[torch.Tensor, ...] = ()
        self.valid = False
        self.linked = False
        self.clean: tuple[int, ...] = ()

    def values(self) -> tuple[torch.Tensor, ...]:
        """The read's current values, one per name."""
        values = self.read()
        values = (values,) if isinstance(values, torch.Tensor) else tuple(values)
        if len(values) != len(self.names):
            raise ValueError(f"Read for {', '.join(self.names)} returned {len(values)} tensors")
        return values

    def allocate(
        self, values: Sequence[torch.Tensor], device: torch.device | str | None, allocator: BufferAllocator | None
    ) -> None:
        """Allocate float32 buffers shaped like ``values`` on ``device`` (each value's, if None)."""
        # Normal tensors even when first read inside inference mode: they are written in place later.
        with torch.inference_mode(False):
            buffers = tuple(
                allocator(name, value, device)
                if allocator is not None
                else torch.empty(value.shape, dtype=torch.float32, device=device or value.device)
                for name, value in zip(self.names, values, strict=True)
            )
        for buffer, value in zip(buffers, values, strict=True):
            if buffer.shape != value.shape or buffer.dtype != torch.float32 or buffer.device != _device(device, value):
                raise ValueError("State allocator must preserve shape, float32 dtype and device")
        self.buffers = buffers

    def refresh(self, device: torch.device | str | None, allocator: BufferAllocator | None = None) -> None:
        """Copy the read's values into the buffers, allocating them on ``device`` (the read's, if None) first."""
        values = self.values()
        if not self.buffers:
            self.allocate(values, device, allocator)
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


def _device(device: torch.device | str | None, value: torch.Tensor) -> torch.device:
    """The concrete device a buffer for ``value`` lives on when allocated on ``device``."""
    selected = torch.device(device or value.device)
    if selected.type == "cuda" and selected.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    if selected.type == "cpu":
        return torch.device("cpu")
    return selected


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
        self._buffer_allocator: BufferAllocator | None = None

    def set_buffer_allocator(self, allocator: BufferAllocator) -> None:
        """Allocate cache buffers with ``allocator(name, value, device)``, which returns float32 storage shaped
        like ``value``; call before the first cached read. Callers keep the storage's owner alive while it is used.
        """
        if any(registration.buffers for registration in self._registrations):
            raise RuntimeError("State allocator must be selected before the first cached read")
        self._buffer_allocator = allocator

    @overload
    def register(
        self,
        names: str,
        read: Columns,
        *,
        write: Callable[[torch.Tensor, torch.Tensor], None] | None = None,
        invalidated_by: Iterable[str] | None = None,
        invalidates: Iterable[str] | None = None,
    ) -> None: ...

    @overload
    def register(
        self,
        names: str | Sequence[str],
        read: Read,
        *,
        write: Callable[..., None] | None = None,
        invalidated_by: Iterable[str] | None = None,
        invalidates: Iterable[str] | None = None,
    ) -> None: ...

    def register(
        self,
        names: str | Sequence[str],
        read: Columns | Read,
        *,
        write: Callable[..., None] | None = None,
        invalidated_by: Iterable[str] | None = None,
        invalidates: Iterable[str] | None = None,
    ) -> None:
        """Serve ``read``'s values as ``names``, re-read after a state change and flushed by ``write(rows, *buffers)``.

        ``read`` returns one tensor per name. Without ``write`` the tensors are read-only. ``invalidated_by``
        and ``invalidates`` override the default tags (see the module docstring).
        """
        names = (names,) if isinstance(names, str) else tuple(names)
        if invalidated_by is None:
            invalidated_by = (DYNAMICS,) if write is not None else (DYNAMICS, KINEMATICS)
        if invalidates is None:
            invalidates = (KINEMATICS,) if write is not None else ()
        registration = _Registration(names, read, write, frozenset(invalidated_by), frozenset(invalidates))
        for name in names:
            self._unbind(name)
            self._by_name[name] = registration
        self._registrations.append(registration)

    def _serve(self, name: str, tensor: Any) -> None:
        """Serve ``tensor`` as ``name`` as is, outside the registry's reads and writes."""
        if name in self._by_name:
            raise AttributeError(f"Simulator state {name!r} is served by the simulator; write into it in place instead")
        self._unbind(name)
        self._tensors[name] = tensor

    @contextmanager
    def mutation(self, tag: str) -> Iterator[None]:
        """Wrap a mutation of the simulation state: flush pending writes first, fire ``tag`` after."""
        self.flush_writes()
        try:
            yield
        finally:
            self.invalidate({tag})

    def flush_writes(self) -> None:
        """Write the rows written in place to the simulation, then fire the ``invalidates`` tags of the fields written.

        A flushed buffer already holds what the simulation now has, so it stays valid.
        """
        stale_written = [
            registration
            for registration in self._registrations
            if not registration.valid and registration.buffers and registration.written()
        ]
        if stale_written:
            for registration in stale_written:  # discard the write, so the next access re-reads
                registration.mark_clean()
            names = ", ".join(name for registration in stale_written for name in registration.names)
            raise RuntimeError(f"Simulator state {names} was written in place after the simulation changed it")
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
        fired: set[str] = set()
        # Compare every written tensor with the simulation before writing any of them back.
        for registration, rows in [(registration, registration.changed_rows()) for registration in written]:
            if rows.numel():
                assert registration.write is not None
                registration.write(rows, *registration.buffers)
                fired |= registration.invalidates
            registration.mark_clean()
        self.invalidate(fired)

    def invalidate(self, tags: Iterable[str]) -> None:
        """Mark stale every tensor invalidated by one of ``tags``; linked tensors are never stale."""
        tags = set(tags)
        for registration in self._registrations:
            if not registration.linked and not tags.isdisjoint(registration.invalidated_by):
                registration.valid = False

    def mark_stale(self, names: str | Iterable[str] | None = None) -> None:
        """Mark every tensor stale, or only those registered with ``names``, so the next access re-reads it.

        Linked tensors stay valid.
        """
        if names is None:
            for registration in self._registrations:
                if not registration.linked:
                    registration.valid = False
        else:
            self.invalidate({names} if isinstance(names, str) else names)

    def link(self, sources: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Serve writable tensors from ``sources``' memory (``set_``), written outside the simulation; return them.

        Linked tensors are never re-read, and their versions do not follow writes made through ``sources``: the
        writer must bump the returned tensor after each write (``torch.autograd.graph.increment_version``) for
        the next flush to write the changed rows. The co-simulation bridge bumps every linked tensor each
        exchange; any other link must do this itself. Each registration's names are linked together.
        """
        registrations = {id(self._by_name.get(name)): self._by_name.get(name) for name in sources}
        for registration in registrations.values():
            if registration is None or registration.write is None:
                raise ValueError(f"Simulator state {sorted(sources)} must all be writable to link")
            if missing := [name for name in registration.names if name not in sources]:
                raise ValueError(f"Simulator state {', '.join(registration.names)} must be linked together: {missing}")
        tensors = {name: self.get(name) for name in sources}
        for name, source in sources.items():
            tensor = tensors[name]
            if (source.shape, source.dtype, source.device) != (tensor.shape, tensor.dtype, tensor.device):
                raise ValueError(
                    f"Simulator state {name!r} is {tensor.dtype} {tuple(tensor.shape)} on {tensor.device}, "
                    f"cannot link {source.dtype} {tuple(source.shape)} on {source.device}"
                )
        with torch.inference_mode(False):
            for name, source in sources.items():
                tensors[name].set_(source)
        for registration in registrations.values():
            assert registration is not None
            registration.linked = True
        return tensors

    def unlink(self, names: Iterable[str]) -> None:
        """Move ``names`` onto their own memory, holding the last values, and re-read them from the simulation.

        Writes not yet flushed are dropped.
        """
        with torch.inference_mode(False):
            for name in names:
                registration = self._by_name[name]
                buffer = registration.buffers[registration.names.index(name)]
                buffer.set_(buffer.clone())
                registration.linked = False
                registration.valid = False
                registration.mark_clean()

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
            registration.refresh(self.device, self._buffer_allocator)
            registration.valid = True
        return registration.buffers[registration.names.index(name)]

    def names(self) -> list[str]:
        return [*self._by_name, *self._tensors]

    def writable_names(self) -> list[str]:
        """Names of registered tensors whose in-place writes flush to the simulation."""
        return [name for name, registration in self._by_name.items() if registration.write is not None]

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
        return sim.state.get(self._name)

    def __set__(self, sim: Any, value: Any) -> None:
        sim.state._serve(self._name, value)
