"""Python version compatibility shims.

``entry_points(group=...)`` keyword selection is only available on the stdlib
``importlib.metadata`` from Python 3.10+. On 3.8/3.9 the call takes no arguments and
returns a dict keyed by group, so we wrap it to present the same ``group=`` API.
"""

from __future__ import annotations

import sys
from importlib.metadata import entry_points as _entry_points

__all__ = ["entry_points"]


if sys.version_info >= (3, 10):
    entry_points = _entry_points
else:

    def entry_points(*, group: str):  # type: ignore[misc]
        """3.8/3.9 fallback: filter the group-keyed dict returned by argless entry_points()."""
        return _entry_points().get(group, [])
