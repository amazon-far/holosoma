"""MuJoCo backend module with optional Warp support.

This module provides two backends for MuJoCo simulation:
- ClassicBackend: CPU-based single-environment simulation (always available)
- WarpBackend: GPU-accelerated multi-environment simulation (optional)

WarpBackend requires additional dependencies:
- warp-lang: GPU kernel compilation framework
- mujoco-warp: MuJoCo integration with Warp

To install with GPU support:
    bash scripts/setup_mujoco.sh --with-warp

Or install dependencies manually:
    pip install warp-lang mujoco-warp
"""

from __future__ import annotations

from .base import IMujocoBackend
from .classic_backend import ClassicBackend

# Try to import WarpBackend - gracefully handle if warp not installed
try:
    from .warp_backend import WarpBackend

    WARP_AVAILABLE = True
except ImportError:
    # Warp dependencies not available (expected for CPU-only installs). Bind WarpBackend to a
    # never-instantiated sentinel subclass so `isinstance(backend, WarpBackend)` is a safe False
    # everywhere without per-callsite None guards.
    class WarpBackend(IMujocoBackend):  # type: ignore[no-redef]  # optional-import sentinel
        """Sentinel stand-in when Warp is unavailable; never instantiated."""

    WARP_AVAILABLE = False


__all__ = ["WARP_AVAILABLE", "ClassicBackend", "IMujocoBackend", "WarpBackend"]
