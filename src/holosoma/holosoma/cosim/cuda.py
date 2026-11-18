"""Shared CUDA allocations and the lifetime of imported ones.

Exports are Warp arrays allocated outside Warp's memory pool and shared through CUDA IPC (Warp 1.15 to 1.19).
An owner must outlive every peer mapping of it. Importing this module loads no GPU SDK.
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


def share(name: str, owner: Any) -> BufferSpec:
    """Describe an :func:`allocate_shared` array for import by another process."""
    return BufferSpec(name, tuple(owner.shape), owner.ipc_handle().hex(), owner.dtype.__name__)


class CacheAllocator:
    """Allocate selected SimState caches outside Warp's memory pool.

    Retain this owner until the receiver has unmapped its imports. Other caches,
    and empty selected ones, keep ordinary Torch storage. Warp is imported only
    when a selected cache is allocated.
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


_SHARED_STATE: weakref.WeakKeyDictionary[BaseSimulator, CacheAllocator] = weakref.WeakKeyDictionary()


def shared_state(simulator: BaseSimulator) -> CacheAllocator:
    """The allocator sharing every non-empty state field of ``simulator``, installed by the first caller.

    Co-simulation plugins on one simulator share it; it must be installed before state is first read.
    """
    allocator = _SHARED_STATE.get(simulator)
    if allocator is None:
        allocator = CacheAllocator(simulator.state.names())
        simulator.state.set_buffer_allocator(allocator)
        _SHARED_STATE[simulator] = allocator
    return allocator


class MappedBuffers:
    """Own imports until SDK and caller views have been released.

    Each import is served as one tensor: the given target, moved onto it with ``set_``, or Warp's view. Its
    version moves for local in-place writes and for :meth:`advance`, which records the owner's writes.
    """

    def __init__(
        self, specs: Iterable[BufferSpec], *, device: str, targets: Mapping[str, torch.Tensor] | None = None
    ) -> None:
        import torch
        import warp as wp

        targets = targets or {}
        self.arrays: dict[str, Any] = {}
        self.tensors: dict[str, torch.Tensor] = {}
        self.versions: dict[str, int] = {}
        self.references: dict[str, tuple[Any, ...]] = {}
        self.adopted: list[str] = []
        for spec in specs:
            if spec.name in self.arrays:
                raise ValueError(f"Duplicate IPC buffer: {spec.name}")
            if spec.dtype not in SHARED_DTYPES:
                raise ValueError(f"IPC buffer {spec.name} must be one of {SHARED_DTYPES}, got {spec.dtype}")
            owner = wp.from_ipc_handle(
                bytes.fromhex(spec.handle), shape=spec.shape, dtype=getattr(wp, spec.dtype), device=device
            )
            tensor = wp.to_torch(owner)
            self.arrays[spec.name] = owner
            self.references[spec.name] = (weakref.ref(owner), weakref.ref(tensor))
            target = targets.get(spec.name)
            if target is not None:
                if (target.shape, target.dtype, target.device) != (tensor.shape, tensor.dtype, tensor.device):
                    raise ValueError(
                        f"Target for {spec.name!r} is {target.dtype} {tuple(target.shape)} on {target.device}, "
                        f"the import is {tensor.dtype} {tuple(tensor.shape)} on {tensor.device}"
                    )
                with torch.inference_mode(False):
                    target.set_(tensor)
                tensor = target
                self.adopted.append(spec.name)
            self.tensors[spec.name] = tensor
            self.versions[spec.name] = tensor._version

    def _written(self) -> list[str]:
        return [name for name, tensor in self.tensors.items() if tensor._version != self.versions[name]]

    def verify_read_only(self) -> None:
        """Raise if a caller wrote borrowed state; detects in-place writes through these Torch tensors only."""
        if written := self._written():
            raise RuntimeError(f"Borrowed IPC buffers {written} were written in place; their owner writes them")

    def advance(self, names: Iterable[str]) -> None:
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
        """Move adopted targets onto their own memory, holding the last values, and drop every mapping."""
        import torch

        written = self._written()
        with torch.inference_mode(False):
            for name in self.adopted:
                tensor = self.tensors[name]
                tensor.set_(tensor.clone())
        self.tensors.clear()
        self.arrays.clear()
        gc.collect()
        if written:
            raise RuntimeError(f"Borrowed IPC buffers {written} were written in place; their owner writes them")
        retained = [name for name, refs in self.references.items() if any(ref() is not None for ref in refs)]
        if retained:
            raise RuntimeError(f"IPC mappings are still referenced: {retained}")


class WriteLog:
    """Report which of ``tensors`` were written in place since the last call; the first call reports all."""

    def __init__(self, tensors: Mapping[str, torch.Tensor]) -> None:
        self.tensors = dict(tensors)
        self.versions = dict.fromkeys(self.tensors, -1)

    def __call__(self) -> list[str]:
        written = [name for name, tensor in self.tensors.items() if tensor._version != self.versions[name]]
        for name in written:
            self.versions[name] = self.tensors[name]._version
        return written
