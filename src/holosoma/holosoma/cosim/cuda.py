"""Owning shared allocations and lifetime of imported CUDA buffers."""

from __future__ import annotations

import gc
import weakref
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class BufferSpec:
    name: str
    shape: tuple[int, ...]
    handle: str
    dtype: str = "float32"


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


def share(name: str, owner: Any) -> BufferSpec:
    """Describe an :func:`allocate_shared` array for import by another process."""
    return BufferSpec(name, tuple(owner.shape), owner.ipc_handle().hex(), owner.dtype.__name__)


class CacheAllocator:
    """Allocate selected SimState caches outside Warp's memory pool.

    Retain this owner until the receiver has unmapped its imports. Other caches,
    and empty selected ones, keep ordinary Torch storage. IPC APIs are checked only
    when sharing is used.
    """

    def __init__(self, fields: Iterable[str]) -> None:
        self.fields = frozenset(fields)
        self.owners: dict[str, Any] = {}
        self.tensors: dict[str, torch.Tensor] = {}

    def __call__(self, name: str, value: torch.Tensor, device: Any) -> torch.Tensor:
        import torch
        import warp as wp

        selected = torch.device(device or value.device)
        if name not in self.fields or not value.numel():
            return torch.empty(value.shape, dtype=torch.float32, device=selected)
        if name in self.owners:
            raise RuntimeError(f"Published cache allocation was replaced: {name}")
        owner = allocate_shared(tuple(value.shape), "float32", selected)
        tensor = wp.to_torch(owner)
        self.owners[name], self.tensors[name] = owner, tensor
        return tensor

    def exports(self, values: Mapping[str, torch.Tensor]) -> list[BufferSpec]:
        result = []
        for name, tensor in values.items():
            owner = self.owners[name]
            if tensor.data_ptr() != owner.ptr or tensor.numel() != self.tensors[name].numel():
                raise RuntimeError(f"State tensor no longer aliases its published cache: {name}")
            if not tensor.is_contiguous():
                raise RuntimeError(f"Published state tensors must be contiguous: {name}")
            result.append(share(name, owner))
        return result


class MappedBuffers:
    """Own imports until SDK and caller views have been released."""

    def __init__(self, descriptions: Iterable[Mapping[str, Any]], *, device: str) -> None:
        import warp as wp

        self.arrays: dict[str, Any] = {}
        self.tensors: dict[str, torch.Tensor] = {}
        self.versions: dict[str, int] = {}
        self.references: dict[str, tuple[Any, Any]] = {}
        for raw in descriptions:
            spec = BufferSpec(**{**raw, "shape": tuple(raw["shape"])})
            if spec.name in self.arrays:
                raise ValueError(f"Duplicate IPC buffer: {spec.name}")
            if spec.dtype not in SHARED_DTYPES:
                raise ValueError(f"IPC buffer {spec.name} must be one of {SHARED_DTYPES}, got {spec.dtype}")
            owner = wp.from_ipc_handle(
                bytes.fromhex(spec.handle), shape=spec.shape, dtype=getattr(wp, spec.dtype), device=device
            )
            tensor = wp.to_torch(owner)
            self.arrays[spec.name], self.tensors[spec.name] = owner, tensor
            self.versions[spec.name] = tensor._version
            self.references[spec.name] = (weakref.ref(owner), weakref.ref(tensor))

    def verify_read_only(self) -> None:
        """Raise if a caller wrote borrowed state; detects in-place writes through these Torch views only."""
        if any(t._version != self.versions[name] for name, t in self.tensors.items()):
            raise RuntimeError("A receiver modified borrowed IPC state")

    def release(self) -> None:
        self.verify_read_only()
        self.tensors.clear()
        self.arrays.clear()
        gc.collect()
        retained = [name for name, refs in self.references.items() if any(ref() is not None for ref in refs)]
        if retained:
            raise RuntimeError(f"IPC mappings are still referenced: {retained}")
