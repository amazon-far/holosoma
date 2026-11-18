"""Simulator state tensors, served as lazy reads into stable buffers.

A backend binds each state quantity once, in ``prepare_sim``, with :meth:`SimState.register`. Its read
returns the current value: a function returning a tensor (or a tuple of tensors, one per name), or a
:class:`Columns` spec, a column selection of a backend tensor. The registry copies each value into a
buffer of the field's dtype (float32 by default) on ``device``, allocated on the first read, on the first
access after the simulation state changed, and ``write`` flushes in-place changes back to the simulation, row by row.

Every served tensor keeps its identity, so references to it stay valid. Reads see the current
simulation state. In-place writes reach the simulation before its next physics step, FK pass or any
state write that would replace them: the changed rows (environments) are flushed through the tensor's
``write``. A tensor bound without ``write`` is read-only. :meth:`SimState.link` serves writable tensors whose memory is
written from outside: they are never re-read, and their changed rows flush like in-place writes.

Fields go stale by validation group. Each field belongs to the ``validation_groups`` it registers with, and to
a group named after itself; invalidating a group marks its fields stale. Invalidation cascades: a field going
stale, or being flushed, invalidates its own ``invalidates`` groups. Simulation inputs belong to
:data:`STATE` and invalidate :data:`DERIVED`, to which quantities derived from them belong. Backends wrap every
state mutation in :meth:`SimState.mutation`, which flushes pending writes first and invalidates a group after:
:data:`STATE` when the simulation state changes, :data:`DERIVED` when only quantities derived from it are
recomputed.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, overload

import torch

Read = Callable[[], "torch.Tensor | tuple[torch.Tensor, ...]"]
BufferAllocator = Callable[[str, torch.Tensor, torch.device | str | None, torch.dtype], torch.Tensor]

STATE = "state"
"""Validation group of the simulation's inputs, invalidated by a physics step or a backend state write."""
DERIVED = "derived"
"""Validation group of quantities derived from the inputs, invalidated by an FK pass or a changed input."""


class _Registration:
    __slots__ = (
        "buffers",
        "clean",
        "dtype",
        "invalidates",
        "linked",
        "memory",
        "names",
        "read",
        "valid",
        "validation_groups",
        "write",
    )

    def __init__(
        self,
        names: tuple[str, ...],
        read: Read | None,
        write: Callable[..., None] | None,
        validation_groups: frozenset[str],
        invalidates: frozenset[str],
        dtype: torch.dtype,
    ) -> None:
        self.names = names
        self.dtype = dtype
        self.read = read
        self.write = write
        self.validation_groups = validation_groups | frozenset(names)
        self.invalidates = invalidates
        self.buffers: tuple[torch.Tensor, ...] = ()
        self.valid = False
        self.linked = False
        self.clean: tuple[int, ...] = ()
        self.memory: tuple[int, ...] = ()

    def values(self) -> tuple[torch.Tensor, ...]:
        """The read's current values, one per name."""
        if self.read is None:
            raise RuntimeError(f"Simulator state {', '.join(self.names)} has no read; it is served by a link")
        values = self.read()
        values = (values,) if isinstance(values, torch.Tensor) else tuple(values)
        if len(values) != len(self.names):
            raise ValueError(f"Read for {', '.join(self.names)} returned {len(values)} tensors")
        return values

    def allocate(
        self, values: Sequence[torch.Tensor], device: torch.device | str | None, allocator: BufferAllocator | None
    ) -> None:
        """Allocate buffers of the field's dtype shaped like ``values`` on ``device`` (each value's, if None)."""
        # Normal tensors even when first read inside inference mode: they are written in place later.
        with torch.inference_mode(False):
            buffers = tuple(
                allocator(name, value, device, self.dtype)
                if allocator is not None
                else torch.empty(value.shape, dtype=self.dtype, device=device or value.device)
                for name, value in zip(self.names, values, strict=True)
            )
        for buffer, value in zip(buffers, values, strict=True):
            if buffer.shape != value.shape or buffer.dtype != self.dtype or buffer.device != _device(device, value):
                raise ValueError("State allocator must preserve shape, the field's dtype and device")
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
        """Whether a buffer was written in place since the last read or flush; moving one (``set_``) is not."""
        if tuple(buffer.data_ptr() for buffer in self.buffers) != self.memory:
            self.mark_clean()
            return False
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
        self.memory = tuple(buffer.data_ptr() for buffer in self.buffers)


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
        """Allocate cache buffers with ``allocator(name, value, device, dtype)``, which returns ``dtype`` storage
        shaped like ``value``; call before the first cached read. Callers keep the storage's owner alive while it
        is used.
        """
        if any(registration.buffers and registration.read is not None for registration in self._registrations):
            raise RuntimeError("State allocator must be selected before the first cached read")
        self._buffer_allocator = allocator

    @overload
    def register(
        self,
        names: str,
        read: Columns,
        *,
        write: Callable[[torch.Tensor, torch.Tensor], None] | None = None,
        validation_groups: Iterable[str],
        invalidates: Iterable[str],
        dtype: torch.dtype = torch.float32,
    ) -> None: ...

    @overload
    def register(
        self,
        names: str | Sequence[str],
        read: None,
        *,
        validation_groups: Iterable[str],
        invalidates: Iterable[str],
        dtype: torch.dtype = torch.float32,
    ) -> None: ...

    @overload
    def register(
        self,
        names: str | Sequence[str],
        read: Read,
        *,
        write: Callable[..., None] | None = None,
        validation_groups: Iterable[str],
        invalidates: Iterable[str],
        dtype: torch.dtype = torch.float32,
    ) -> None: ...

    def register(
        self,
        names: str | Sequence[str],
        read: Columns | Read | None,
        *,
        write: Callable[..., None] | None = None,
        validation_groups: Iterable[str],
        invalidates: Iterable[str],
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Serve ``read``'s values as ``names``, re-read after a state change and flushed by ``write(rows, *buffers)``.

        ``read`` returns one tensor per name. Without ``write`` the tensors are read-only. Without ``read`` they
        are served only by :meth:`link`, never re-read and never stale, and are written by their producer inside
        :meth:`producing`. They go stale when one
        of their ``validation_groups`` is invalidated, and going stale or being flushed invalidates the groups in
        ``invalidates``; each name is also a validation group of its own. Buffers have ``dtype``.
        """
        names = (names,) if isinstance(names, str) else tuple(names)
        if read is None and write is not None:
            raise ValueError(f"Simulator state {names} needs a read to flush its writes")
        registration = _Registration(names, read, write, frozenset(validation_groups), frozenset(invalidates), dtype)
        for name in names:
            self._unbind(name)
            self._by_name[name] = registration
        self._registrations.append(registration)

    @contextmanager
    def producing(self, names: str | Iterable[str]) -> Iterator[None]:
        """Wrap the producer's writes to the read-only fields ``names``: raise first if another caller wrote one
        in place, and record the write after, including writes made outside Torch (e.g. Warp kernels)."""
        names = [names] if isinstance(names, str) else list(names)
        registrations = {id(self._by_name[name]): self._by_name[name] for name in names}.values()
        if writable := [name for name in names if self._by_name[name].write is not None]:
            raise ValueError(f"Simulator state {writable} is writable; producing() records read-only fields")
        if written := [name for name in names if self._by_name[name].written()]:
            for registration in registrations:  # report a write once
                registration.mark_clean()
            raise RuntimeError(f"Simulator state {written} is read-only; only its producer writes it")
        try:
            yield
        finally:
            for registration in registrations:
                for buffer in registration.buffers:
                    torch.autograd.graph.increment_version(buffer)
            self.produced(names)

    def produced(self, names: str | Iterable[str]) -> None:
        """Record that the producer of the read-only fields ``names`` wrote them, with their versions already
        bumped (e.g. by the co-simulation bridge, which checks for other writers itself)."""
        for name in [names] if isinstance(names, str) else names:
            self._by_name[name].mark_clean()

    def _serve(self, name: str, tensor: Any) -> None:
        """Serve ``tensor`` as ``name`` as is, outside the registry's reads and writes."""
        if name in self._by_name:
            raise AttributeError(f"Simulator state {name!r} is served by the simulator; write into it in place instead")
        self._unbind(name)
        self._tensors[name] = tensor

    @contextmanager
    def mutation(self, group: str) -> Iterator[None]:
        """Wrap a mutation of the simulation state: flush pending writes first, invalidate ``group`` after."""
        self.flush_writes()
        try:
            yield
        finally:
            self.invalidate({group})

    def flush_writes(self) -> None:
        """Write the rows written in place to the simulation, then invalidate the ``invalidates`` groups of the fields
        written.

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

    def invalidate(self, groups: Iterable[str]) -> None:
        """Mark stale every tensor in one of ``groups``, and in turn every tensor in the groups those invalidate;
        linked tensors are never stale."""
        fired: set[str] = set()
        pending = set(groups)
        while pending:
            fired |= pending
            matched = [
                registration
                for registration in self._registrations
                if registration.read is not None
                and not registration.linked
                and not pending.isdisjoint(registration.validation_groups)
            ]
            for registration in matched:
                registration.valid = False
            pending = {group for registration in matched for group in registration.invalidates} - fired

    def mark_stale(self, names: str | Iterable[str] | None = None) -> None:
        """Mark every tensor stale, or only those registered with ``names``, so the next access re-reads it.

        Linked tensors stay valid.
        """
        if names is None:
            for registration in self._registrations:
                if registration.read is not None and not registration.linked:
                    registration.valid = False
        else:
            self.invalidate({names} if isinstance(names, str) else names)

    def link(self, sources: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Serve tensors from ``sources``' memory (``set_``) in place of their read, and return them.

        Linked tensors are never re-read, and their versions do not follow writes made through ``sources``: the
        writer reports each write as a version bump on the returned tensor. A writable tensor then flushes its
        changed rows into the simulation; a read-only one is recorded with :meth:`producing`. The co-simulation
        bridge reports every linked tensor each exchange; any other link must do this itself. Each
        registration's names are linked together.
        """
        registrations = {id(self._by_name[name]): self._by_name[name] for name in sources}.values()
        for registration in registrations:
            if missing := [name for name in registration.names if name not in sources]:
                raise ValueError(f"Simulator state {', '.join(registration.names)} must be linked together: {missing}")
        with torch.inference_mode(False):
            for registration in registrations:
                if registration.read is None and not registration.buffers:
                    registration.buffers = tuple(
                        torch.empty(0, dtype=registration.dtype, device=sources[name].device)
                        for name in registration.names
                    )
        tensors = {name: self.get(name) for name in sources}
        for name, source in sources.items():
            tensor = tensors[name]
            same_layout = tensor.numel() == 0 or (tensor.shape, tensor.device) == (source.shape, source.device)
            if tensor.dtype != source.dtype or not same_layout:
                raise ValueError(
                    f"Simulator state {name!r} is {tensor.dtype} {tuple(tensor.shape)} on {tensor.device}, "
                    f"cannot link {source.dtype} {tuple(source.shape)} on {source.device}"
                )
        with torch.inference_mode(False):
            for name, source in sources.items():
                tensors[name].set_(source)
        for registration in registrations:
            registration.linked = True
            registration.valid = True
            registration.mark_clean()
        return tensors

    def unlink(self, names: Iterable[str]) -> None:
        """Move ``names`` onto their own memory, holding the last values; fields with a read are re-read from the
        simulation. Writes not yet flushed are dropped.
        """
        with torch.inference_mode(False):
            for name in names:
                registration = self._by_name[name]
                buffer = registration.buffers[registration.names.index(name)]
                buffer.set_(buffer.clone())
                registration.linked = False
                registration.valid = registration.read is None
                registration.mark_clean()

    def get(self, name: str) -> Any:
        registration = self._by_name.get(name)
        if registration is None:
            try:
                return self._tensors[name]
            except KeyError:
                raise AttributeError(f"Simulator state {name!r} is not bound") from None
        if not registration.buffers and registration.read is None:
            raise RuntimeError(f"Simulator state {name!r} has no read; link it before reading")
        if not registration.valid and registration.read is not None:
            if registration.written():
                registration.mark_clean()  # discard the write, so the next access re-reads
                names = ", ".join(registration.names)
                raise RuntimeError(f"Simulator state {names} was written in place after the simulation changed it")
            registration.refresh(self.device, self._buffer_allocator)
            registration.valid = True
        return registration.buffers[registration.names.index(name)]

    def names(self) -> list[str]:
        return [*self._by_name, *self._tensors]

    def dtype(self, name: str) -> torch.dtype:
        """The dtype of registered tensor ``name``'s buffer."""
        return self._by_name[name].dtype

    def writable_names(self) -> list[str]:
        """Names of registered tensors whose in-place writes flush to the simulation."""
        return [name for name, registration in self._by_name.items() if registration.write is not None]

    def _read_fresh(self, name: str) -> torch.Tensor:
        """Return a fresh read of ``name`` without touching the served tensor, for consistency checks."""
        registration = self._by_name.get(name)
        if registration is None:
            return torch.as_tensor(self._tensors[name][:]).clone()
        if registration.read is None:
            return registration.buffers[registration.names.index(name)].clone()
        return registration.values()[registration.names.index(name)].to(registration.dtype).clone()

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
