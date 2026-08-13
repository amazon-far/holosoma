"""Python version compatibility shims."""

import sys

if sys.version_info >= (3, 12):
    from importlib.metadata import entry_points
else:
    from importlib_metadata import entry_points

__all__ = ["entry_points"]
