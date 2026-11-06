"""Version adapters for one shared Isaac simulator.

The launcher selects an adapter before importing scene helpers. Isaac Lab owns
physics-engine dispatch within each version; Holosoma owns only its interface.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    import torch

    from .lab2 import Lab2Backend

_active: ContextVar[Lab2Backend | None] = ContextVar("holosoma_isaac_backend", default=None)


def current() -> Lab2Backend:
    backend = _active.get()
    if backend is None:
        from .lab2 import Lab2Backend

        backend = Lab2Backend()
    return backend


def as_tensor(value: Any) -> torch.Tensor:
    """Borrow a native tensor; Lab 3 ProxyArray owns its cached Torch view."""
    import torch

    if isinstance(value, torch.Tensor):
        return value
    if hasattr(value, "torch"):
        return cast(torch.Tensor, value.torch)
    import warp

    return cast(torch.Tensor, warp.to_torch(value))
