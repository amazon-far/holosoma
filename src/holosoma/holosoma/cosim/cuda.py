"""Shared CUDA allocations and the lifetime of imported ones.

Exports are Warp arrays allocated outside Warp's memory pool and shared through CUDA IPC (Warp 1.15 to 1.19).
An allocation must outlive every peer mapping of it. Importing this module loads no GPU SDK.
"""

from __future__ import annotations

import gc
import weakref
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator


@dataclass(frozen=True)
class BufferSpec:
    name: str
    shape: tuple[int, ...]
    handle: str
    dtype: str = "float32"

    @classmethod
    def from_message(cls, raw: Mapping[str, Any]) -> BufferSpec:
        return cls(**{**raw, "shape": tuple(raw["shape"])})


SHARED_DTYPES = ("float32", "uint8")


def allocate_shared(shape: tuple[int, ...], dtype: str, device: Any) -> Any:
    """Allocate an exportable Warp array outside Warp's memory pool; retain it while peers map it."""
    import torch
    import warp as wp

    if dtype not in SHARED_DTYPES:
        raise ValueError(f"IPC buffers must be one of {SHARED_DTYPES}, got {dtype}")
    selected = torch.device(device)
    if selected.type != "cuda":
        raise ValueError("CUDA IPC requires GPU buffers")
    with wp.ScopedMempool(wp.device_from_torch(selected), False):
        return wp.empty(tuple(shape), dtype=getattr(wp, dtype), device=wp.device_from_torch(selected))


def share(name: str, allocation: Any) -> BufferSpec:
    """Describe an :func:`allocate_shared` array for import by another process."""
    return BufferSpec(name, tuple(allocation.shape), allocation.ipc_handle().hex(), allocation.dtype.__name__)


class SharedStateAllocator:
    """Allocate selected SimState caches outside Warp's memory pool.

    Retain this allocator until the peer has unmapped its imports. Other caches,
    and empty selected ones, keep ordinary Torch storage. Warp is imported only
    when a selected cache is allocated.
    """

    def __init__(self, fields: Iterable[str]) -> None:
        self.fields = frozenset(fields)
        self.allocations: dict[str, Any] = {}
        self.tensors: dict[str, torch.Tensor] = {}

    def __call__(self, name: str, value: torch.Tensor, device: Any) -> torch.Tensor:
        import torch
        import warp as wp

        selected = torch.device(device or value.device)
        if name not in self.fields or not value.numel():
            return torch.empty(value.shape, dtype=torch.float32, device=selected)
        if name in self.allocations:
            raise RuntimeError(f"Published cache allocation was replaced: {name}")
        allocation = allocate_shared(tuple(value.shape), "float32", selected)
        tensor = wp.to_torch(allocation)
        self.allocations[name], self.tensors[name] = allocation, tensor
        return tensor

    def specs(self, values: Mapping[str, torch.Tensor]) -> list[BufferSpec]:
        result = []
        for name, tensor in values.items():
            allocation = self.allocations[name]
            if tensor.data_ptr() != allocation.ptr or tensor.numel() != self.tensors[name].numel():
                raise RuntimeError(f"State tensor no longer aliases its published cache: {name}")
            if not tensor.is_contiguous():
                raise RuntimeError(f"Published state tensors must be contiguous: {name}")
            result.append(share(name, allocation))
        return result


_ALLOCATORS: weakref.WeakKeyDictionary[BaseSimulator, SharedStateAllocator] = weakref.WeakKeyDictionary()


def shared_state_allocator(simulator: BaseSimulator) -> SharedStateAllocator:
    """The allocator sharing every non-empty state field of ``simulator``, installed by the first caller.

    Co-simulation plugins on one simulator share it; it must be installed before state is first read.
    """
    allocator = _ALLOCATORS.get(simulator)
    if allocator is None:
        allocator = SharedStateAllocator(simulator.state.names())
        simulator.state.set_buffer_allocator(allocator)
        _ALLOCATORS[simulator] = allocator
    return allocator


class MappedBuffers:
    """Own imports until SDK and caller views have been released.

    Each import is served as one tensor: Warp's view, or one :meth:`serve` substitutes for it. Its version moves
    for local in-place writes and for :meth:`record_writes`, which records the owner's writes.
    """

    def __init__(self, specs: Iterable[BufferSpec], *, device: str) -> None:
        import warp as wp

        self.arrays: dict[str, Any] = {}
        self.tensors: dict[str, torch.Tensor] = {}
        self.versions: dict[str, int] = {}
        self.references: dict[str, tuple[Any, Any]] = {}
        for spec in specs:
            if spec.name in self.arrays:
                raise ValueError(f"Duplicate IPC buffer: {spec.name}")
            if spec.dtype not in SHARED_DTYPES:
                raise ValueError(f"IPC buffer {spec.name} must be one of {SHARED_DTYPES}, got {spec.dtype}")
            allocation = wp.from_ipc_handle(
                bytes.fromhex(spec.handle), shape=spec.shape, dtype=getattr(wp, spec.dtype), device=device
            )
            tensor = wp.to_torch(allocation)
            self.arrays[spec.name], self.tensors[spec.name] = allocation, tensor
            self.versions[spec.name] = tensor._version
            self.references[spec.name] = (weakref.ref(allocation), weakref.ref(tensor))

    def serve(self, tensors: Mapping[str, torch.Tensor]) -> None:
        """Serve ``tensors``, already linked onto these imports' memory, in place of Warp's views.

        Their holders must move them off the imports before :meth:`release`.
        """
        for name, tensor in tensors.items():
            mapped = self.tensors[name]
            if tensor.data_ptr() != mapped.data_ptr() or tensor.shape != mapped.shape or tensor.dtype != mapped.dtype:
                raise ValueError(f"Tensor served as {name!r} does not use the import's memory")
            self.tensors[name] = tensor
            self.versions[name] = tensor._version

    def _written(self) -> list[str]:
        return [name for name, tensor in self.tensors.items() if tensor._version != self.versions[name]]

    def verify_read_only(self) -> None:
        """Raise if a caller wrote borrowed state; detects in-place writes through these Torch tensors only."""
        if written := self._written():
            raise RuntimeError(
                f"Borrowed IPC buffers {written} were written in place; only their allocating process writes them"
            )

    def record_writes(self, names: Iterable[str]) -> None:
        """Record that the owner wrote ``names``: bump each version as an in-place write would."""
        import torch

        self.verify_read_only()
        names = list(names)
        if unknown := sorted(set(names) - set(self.tensors)):
            raise RuntimeError(f"The peer wrote buffers {unknown} that are not mapped here")
        for name in names:
            tensor = self.tensors[name]
            torch.autograd.graph.increment_version(tensor)
            self.versions[name] = tensor._version

    def release(self) -> None:
        """Drop every mapping; raise if a tensor still uses one. Call :meth:`verify_read_only` first."""
        self.tensors.clear()
        self.arrays.clear()
        gc.collect()
        retained = [name for name, refs in self.references.items() if any(ref() is not None for ref in refs)]
        if retained:
            raise RuntimeError(f"IPC mappings are still referenced: {retained}")


class WriteTracker:
    """Report which of ``tensors`` were written in place since the last call; the first call reports all."""

    def __init__(self, tensors: Mapping[str, torch.Tensor]) -> None:
        self.tensors = dict(tensors)
        self.versions = dict.fromkeys(self.tensors, -1)

    def __call__(self) -> list[str]:
        written = [name for name, tensor in self.tensors.items() if tensor._version != self.versions[name]]
        for name in written:
            self.versions[name] = self.tensors[name]._version
        return written
