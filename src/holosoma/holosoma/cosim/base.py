"""Co-simulation bases: the head's plugin, which owns its bridge, and the peer's handler.

Neither assumes what the peer is; a peer need not be a Holosoma simulator.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import torch

from holosoma.cosim.bridge import HeadBridge, Snapshot
from holosoma.cosim.cuda import BufferSpec, FieldSpec, allocate_shared, shared_state_allocator
from holosoma.simulator.base_simulator.hooks import Phase

if TYPE_CHECKING:
    from holosoma.config_types.plugin import CosimPluginConfig
    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator
    from holosoma.simulator.base_simulator.hooks import HookHandle, HookRegistry


class HeadPlugin:
    """Share ``simulator``'s state with a peer through a :class:`HeadBridge` and close it with the simulator.

    Subclasses choose what to request from the peer, the metadata the peer receives, and the hooks that
    publish. Installs the shared cache allocator, so it must run before anything reads simulator state;
    construction blocks until the peer accepts.
    """

    name = "cosim"

    def __init__(self, cfg: CosimPluginConfig, simulator: BaseSimulator) -> None:
        if not cfg.endpoint:
            raise ValueError("Set the co-simulation endpoint with --plugin.<key>.endpoint")
        if torch.device(simulator.sim_device).type != "cuda":
            raise ValueError("Co-simulation shares state through CUDA IPC; run the head on a CUDA device")
        slot = simulator.hooks.installing_slot
        if slot is None:
            raise RuntimeError("Co-simulation plugins are constructed by install_plugins, which names their slot")
        self.slot = slot
        self.simulator = simulator
        self.bridge = HeadBridge(
            simulator,
            shared_state_allocator(simulator),
            endpoint=cfg.endpoint,
            requests=self.requests(),
            meta=self.meta(),
        )
        try:
            self.accept(self.bridge.peer_meta)
        except BaseException:
            self.bridge.close()
            raise
        self._hooks = list(self.install(simulator.hooks))
        simulator.hooks.add(Phase.CLOSE, self.close, name=f"{self.name}.close")

    def requests(self) -> list[FieldSpec]:
        """Fields to request from the peer, read in place through ``bridge.imports`` after each exchange."""
        return []

    def meta(self) -> dict[str, Any]:
        """JSON metadata sent to the peer at the handshake."""
        return {}

    def accept(self, meta: Mapping[str, Any]) -> None:
        """Check the peer's handshake metadata; raise to refuse it."""

    def install(self, hooks: HookRegistry) -> Iterable[HookHandle]:
        """Register the hooks that publish; they are removed on close."""
        return ()

    def close(self) -> None:
        if self.bridge.closed:
            return
        for hook in self._hooks:
            hook.remove()
        self.bridge.close()


class PeerHandler:
    """A peer's side of the exchange: choose the state to read, answer requests and fill exports.

    Each export is filled from its :meth:`export_source`: a tensor is moved onto the export, so writes to it
    land there; a callable is copied into the export after every :meth:`step`. Subclasses override :meth:`accept`,
    :meth:`reads`, :meth:`export_source` and :meth:`step`, and :meth:`link` and :meth:`unlink` to read the
    head's state through tensors of their own.
    """

    def __init__(self, device: str) -> None:
        self.device = device
        self.allocations: dict[str, Any] = {}
        self._exports: list[tuple[torch.Tensor, Callable[[], torch.Tensor]]] = []

    def open(
        self, state: Sequence[BufferSpec], requests: Sequence[FieldSpec], meta: Mapping[str, Any]
    ) -> tuple[str, ...]:
        """Accept the head and allocate one export per request; return the state fields to read."""
        import warp as wp

        self.accept(meta)
        reads = self.reads({spec.name: spec for spec in state})
        sources = [(request, self.export_source(request)) for request in requests]
        for request, source in sources:
            allocation = allocate_shared(request.shape, request.dtype, self.device)
            self.allocations[request.name] = allocation
            export = wp.to_torch(allocation)
            if not isinstance(source, torch.Tensor):
                self._exports.append((export, source))
                continue
            if (source.shape, source.dtype, source.device) != (export.shape, export.dtype, export.device):
                raise ValueError(
                    f"Export source for {request.name!r} is {source.dtype} {tuple(source.shape)} on "
                    f"{source.device}, the request is {export.dtype} {tuple(export.shape)} on {export.device}"
                )
            with torch.inference_mode(False):
                source.set_(export)
        return reads

    def process(self, snapshot: Snapshot, state: Mapping[str, torch.Tensor]) -> None:
        """Run :meth:`step`, then copy each export from its source."""
        self.step(snapshot, state)
        for export, source in self._exports:
            export.copy_(source())

    def accept(self, meta: Mapping[str, Any]) -> None:
        """Check the head's handshake metadata; raise to refuse it."""

    def meta(self) -> Mapping[str, Any]:
        """JSON metadata sent back to the head once the handshake is accepted."""
        return {}

    def reads(self, state: Mapping[str, BufferSpec]) -> tuple[str, ...]:
        """The head's shared state fields to map and receive each exchange."""
        return ()

    def link(self, imports: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        """Link tensors of the handler's own onto the mapped head fields ``imports``; they then serve as ``state``."""
        return {}

    def unlink(self) -> None:
        """Move what :meth:`link` linked off the mapping; called before the mapping is released."""

    def export_source(self, request: FieldSpec) -> torch.Tensor | Callable[[], torch.Tensor]:
        """Return what fills the export for ``request``; raise if it cannot be answered.

        A tensor shaped like the request is moved onto the export (``set_``), so whoever writes it writes the
        export and it keeps that memory; a callable returns the tensor to copy into the export after each step.
        """
        raise ValueError(f"Peer has no export for the head's request {request.name!r}")

    def step(self, snapshot: Snapshot, state: Mapping[str, torch.Tensor]) -> None:
        """Process the head's read-only ``state`` for ``snapshot``; it is valid only during the call."""

    @staticmethod
    def matching_fields(
        state: Mapping[str, BufferSpec], wanted: Mapping[str, tuple[tuple[int, ...], str]]
    ) -> tuple[str, ...]:
        """The ``wanted`` fields, by name to ``(shape, dtype)``, that ``state`` has; raise if one differs."""
        for name, (shape, dtype) in wanted.items():
            if name in state and (state[name].shape, state[name].dtype) != (tuple(shape), dtype):
                spec = state[name]
                raise ValueError(
                    f"State field {name!r} differs between simulators: "
                    f"head {spec.dtype} {spec.shape}, peer {dtype} {tuple(shape)}"
                )
        return tuple(name for name in state if name in wanted)
