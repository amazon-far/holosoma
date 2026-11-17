"""Co-simulation plugin and peer bases: the head's publisher lifecycle and a peer's buffer handling.

Neither assumes what the peer is; a peer need not be a Holosoma simulator.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import torch

from holosoma.cosim.bridge import BufferRequest, Frame, Publisher
from holosoma.cosim.cuda import BufferSpec, allocate_shared, shared_state
from holosoma.simulator.base_simulator.hooks import Phase

if TYPE_CHECKING:
    from holosoma.config_types.plugin import CosimPluginConfig
    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator
    from holosoma.simulator.base_simulator.hooks import HookHandle, HookRegistry


class CosimPlugin:
    """Share ``simulator``'s state with a peer through a :class:`Publisher` and close it with the simulator.

    Subclasses choose what to request from the peer, the metadata the peer receives, and the hooks that
    publish. Installs the shared cache allocator, so it must run before anything reads simulator state;
    construction blocks until the peer accepts.
    """

    name = "cosim"

    def __init__(self, cfg: CosimPluginConfig, simulator: BaseSimulator) -> None:
        if not cfg.endpoint:
            raise ValueError("Set the co-simulation endpoint with --plugin.<key>.endpoint")
        if torch.device(simulator.sim_device).type != "cuda":
            raise ValueError("Co-simulation shares state through CUDA IPC; run the source on a CUDA device")
        slot = simulator.hooks.installing_slot
        if slot is None:
            raise RuntimeError("Co-simulation plugins are constructed by install_plugins, which names their slot")
        self.slot = slot
        self.simulator = simulator
        self.publisher = Publisher(
            simulator, shared_state(simulator), endpoint=cfg.endpoint, requests=self.requests(), meta=self.meta()
        )
        self._hooks = list(self.install(simulator.hooks))
        simulator.hooks.add(Phase.CLOSE, self.close, name=f"{self.name}.close")

    def requests(self) -> list[BufferRequest]:
        """Buffers to request from the peer, available as ``publisher.imports`` after each exchange."""
        return []

    def meta(self) -> dict[str, Any]:
        """JSON metadata sent to the peer at the handshake."""
        return {}

    def install(self, hooks: HookRegistry) -> Iterable[HookHandle]:
        """Register the hooks that publish; they are removed on close."""
        return ()

    def close(self) -> None:
        if self.publisher.closed:
            return
        for hook in self._hooks:
            hook.remove()
        self.publisher.close()


class BufferHandler:
    """A peer's side of the exchange: choose the state to read, answer requests and fill exports.

    Each export is copied from its source after every :meth:`step`. Subclasses override :meth:`accept`,
    :meth:`reads`, :meth:`export_source` and :meth:`step`.
    """

    def __init__(self, device: str) -> None:
        self.device = device
        self.owners: dict[str, Any] = {}
        self._exports: list[tuple[torch.Tensor, Callable[[], torch.Tensor]]] = []

    def open(
        self, exports: Sequence[BufferSpec], requests: Sequence[BufferRequest], meta: Mapping[str, Any]
    ) -> tuple[str, ...]:
        """Accept the source and allocate one export per request; return the exported fields to read."""
        import warp as wp

        self.accept(meta)
        reads = self.reads({spec.name: spec for spec in exports})
        sources = [(request, self.export_source(request)) for request in requests]
        for request, source in sources:
            owner = allocate_shared(request.shape, request.dtype, self.device)
            self.owners[request.name] = owner
            self._exports.append((wp.to_torch(owner), source))
        return reads

    def process(self, frame: Frame, state: Mapping[str, torch.Tensor]) -> None:
        """Run :meth:`step`, then copy each export from its source."""
        self.step(frame, state)
        for export, source in self._exports:
            export.copy_(source())

    def accept(self, meta: Mapping[str, Any]) -> None:
        """Check the source's handshake metadata; raise to refuse it."""

    def reads(self, exports: Mapping[str, BufferSpec]) -> tuple[str, ...]:
        """The exported fields to map and receive each frame."""
        return ()

    def export_source(self, request: BufferRequest) -> Callable[[], torch.Tensor]:
        """Return what fills the export for ``request`` after each step; raise if it cannot be answered."""
        raise ValueError(f"Receiver has no export for the source's request {request.name!r}")

    def step(self, frame: Frame, state: Mapping[str, torch.Tensor]) -> None:
        """Process the source's read-only ``state`` for ``frame``; it is valid only during the call."""

    @staticmethod
    def matching(
        exports: Mapping[str, BufferSpec], wanted: Mapping[str, tuple[tuple[int, ...], str]]
    ) -> tuple[str, ...]:
        """The ``wanted`` fields, by name to ``(shape, dtype)``, that ``exports`` has; raise if one differs."""
        for name, (shape, dtype) in wanted.items():
            if name in exports and (exports[name].shape, exports[name].dtype) != (tuple(shape), dtype):
                spec = exports[name]
                raise ValueError(
                    f"State field {name!r} differs between simulators: "
                    f"source {spec.dtype} {spec.shape}, receiver {dtype} {tuple(shape)}"
                )
        return tuple(name for name in exports if name in wanted)
