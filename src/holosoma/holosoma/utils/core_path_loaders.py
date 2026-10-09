"""Built-in loaders for local and cached remote paths."""

from __future__ import annotations

import glob
from collections.abc import Iterable
from pathlib import Path

from holosoma.utils.package_path_loader import PackagePathLoader
from holosoma.utils.path import _URI_PREFIX, _resolve_local_path

_REMOTE_PREFIXES = ("wandb://", "http://", "https://")


def _download(uri: str) -> str:
    # Keep cache policy out of this module's import path and avoid the cache's
    # local-path compatibility call forming an eager import cycle.
    from holosoma.utils.file_cache import get_cached_file_path

    return get_cached_file_path(uri)


def _expand_local(pattern: str) -> Iterable[str]:
    if glob.has_magic(pattern):
        return iter(sorted(glob.glob(pattern, recursive=True)))  # noqa: PTH207
    return iter((pattern,))


def _is_package(value: str) -> bool:
    return holosoma_package_path_loader.matches(value)


def _is_local(value: str) -> bool:
    return not _is_package(value) and not value.startswith("@") and _URI_PREFIX.match(value) is None


def _is_cached_remote(value: str) -> bool:
    return value.startswith(_REMOTE_PREFIXES)


class LocalPathLoader:
    """Resolve one local path or expand one local glob."""

    def __init__(self, *, multiple: bool) -> None:
        self.multiple = multiple

    def matches(self, value: str) -> bool:
        return _is_local(value)

    def accepts_asset_root(self, value: str) -> bool:
        """Return whether ``value`` is relative and may be joined to an asset root."""
        return not Path(value).expanduser().is_absolute()

    def load(self, value: str) -> str | Iterable[str]:
        path = _resolve_local_path(value)
        return _expand_local(path) if self.multiple else path


class CachedRemotePathLoader:
    """Localize one cached URI, optionally as a singleton iterator."""

    def __init__(self, *, multiple: bool) -> None:
        self.multiple = multiple

    def matches(self, value: str) -> bool:
        return _is_cached_remote(value)

    def load(self, value: str) -> str | Iterable[str]:
        if self.multiple:
            return (_download(uri) for uri in (value,))
        return _download(value)


holosoma_package_path_loader = PackagePathLoader(
    "holosoma",
    ("@holosoma", "holosoma/"),
    multiple=False,
)
holosoma_package_paths_loader = PackagePathLoader(
    "holosoma",
    ("@holosoma", "holosoma/"),
    multiple=True,
)
local_path_loader = LocalPathLoader(multiple=False)
local_paths_loader = LocalPathLoader(multiple=True)
cached_remote_path_loader = CachedRemotePathLoader(multiple=False)
cached_remote_paths_loader = CachedRemotePathLoader(multiple=True)
