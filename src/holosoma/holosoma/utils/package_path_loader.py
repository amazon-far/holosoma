"""Reusable loader for symbolic paths rooted in an installed Python package."""

from __future__ import annotations

import glob
import sys
from collections.abc import Iterable

if sys.version_info >= (3, 9):
    from importlib.resources import files
else:
    from importlib_resources import files


def _prefix_suffix(value: str, prefix: str) -> str | None:
    """Return the path after one exact symbolic prefix."""
    if prefix.endswith("/"):
        if value == prefix:
            return ""
        if value.startswith(prefix):
            return value[len(prefix) :].lstrip("/\\")
        return None

    if value == prefix:
        return ""
    marker = f"{prefix}/"
    if value.startswith(marker):
        return value[len(marker) :].lstrip("/\\")
    return None


class PackagePathLoader:
    """Resolve symbolic prefixes under one installed Python package."""

    def __init__(self, package: str, prefixes: str | Iterable[str], *, multiple: bool) -> None:
        self.package = package
        self.prefixes = (prefixes,) if isinstance(prefixes, str) else tuple(prefixes)
        self.multiple = multiple
        if not self.prefixes or any(not prefix for prefix in self.prefixes):
            raise ValueError("Package path loader prefixes must be non-empty.")

    def _suffix(self, value: str) -> str | None:
        for prefix in self.prefixes:
            suffix = _prefix_suffix(value, prefix)
            if suffix is not None:
                return suffix
        return None

    def matches(self, value: str) -> bool:
        return self._suffix(value) is not None

    def load(self, value: str) -> str | Iterable[str]:
        suffix = self._suffix(value)
        if suffix is None:
            raise ValueError(f"Unsupported package path {value!r}.")

        root = files(self.package)
        path = str(root / suffix) if suffix else str(root)
        if self.multiple and glob.has_magic(path):
            return iter(sorted(glob.glob(path, recursive=True)))  # noqa: PTH207
        if self.multiple:
            return iter((path,))
        return path
