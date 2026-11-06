"""Re-export of ``importlib.metadata.entry_points`` behind a stable import path.

Kept as a seam so the two consumers (``utils/path.py``, ``utils/config_registry.py``) import one
name; there is no version shim left now that the floor is Python 3.10.
"""

from __future__ import annotations

from importlib.metadata import entry_points

__all__ = ["entry_points"]
