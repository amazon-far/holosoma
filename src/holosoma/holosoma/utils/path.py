"""Extensible conversion from path specifications to local machine paths."""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Generic, Iterable, Protocol, TypeVar, Union

from loguru import logger

from holosoma.utils.pycompat import entry_points

PATH_LOADER_ENTRYPOINT_GROUP = "holosoma.path_loader"
PATHS_LOADER_ENTRYPOINT_GROUP = "holosoma.paths_loader"

if TYPE_CHECKING:
    LocalPath = Union[str, os.PathLike[str]]
else:
    LocalPath = Union[str, os.PathLike]


class PathLoader(Protocol):
    """A generic one-to-one path loader."""

    def matches(self, path: str) -> bool:
        """Return whether this loader handles ``path`` without doing I/O."""
        ...

    def load(self, path: str) -> LocalPath:
        """Resolve a matched string to one local machine path."""
        ...


class PathsLoader(Protocol):
    """A generic one-to-many path loader."""

    def matches(self, path: str) -> bool:
        """Return whether this loader handles ``path`` without doing I/O."""
        ...

    def load(self, path: str) -> Iterable[LocalPath]:
        """Resolve a matched string to zero or more local machine paths."""
        ...


LoaderT = TypeVar("LoaderT", PathLoader, PathsLoader)


@dataclass(frozen=True)
class _RegisteredLoader(Generic[LoaderT]):
    loader: LoaderT
    source: str


_URI_PREFIX = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")


def _resolve_local_path(path: str) -> str:
    """Expand a plain local path to an absolute path."""
    path_obj = Path(path).expanduser()
    if path_obj.is_absolute():
        return str(path_obj)
    return str(path_obj.resolve())


def _validate_loader(loader: object, *, kind: str) -> None:
    """Validate the common matcher/loader object contract."""
    if not callable(getattr(loader, "matches", None)):
        raise TypeError(f"{kind} loader must define callable matches(path).")
    if not callable(getattr(loader, "load", None)):
        raise TypeError(f"{kind} loader must define callable load(path).")


def _load_entrypoint_group(
    group: str,
    loaders: list[_RegisteredLoader[LoaderT]],
    *,
    kind: str,
) -> None:
    """Load one generic loader entry-point group with per-extension isolation."""
    for ep in entry_points(group=group):
        try:
            loader = ep.load()
            _validate_loader(loader, kind=kind)
            loaders.append(_RegisteredLoader(loader, ep.value))
        except Exception as exc:  # noqa: PERF203 - each entry point must fail independently
            logger.warning(f"Skipping path loader from {ep.value!r}: {exc}")


@lru_cache(maxsize=1)
def _entrypoint_loaders() -> tuple[
    tuple[_RegisteredLoader[PathLoader], ...],
    tuple[_RegisteredLoader[PathsLoader], ...],
]:
    """Load and cache generic path loaders from package entry points."""
    path_loaders: list[_RegisteredLoader[PathLoader]] = []
    paths_loaders: list[_RegisteredLoader[PathsLoader]] = []
    _load_entrypoint_group(PATH_LOADER_ENTRYPOINT_GROUP, path_loaders, kind="Path")
    _load_entrypoint_group(PATHS_LOADER_ENTRYPOINT_GROUP, paths_loaders, kind="Paths")
    return tuple(path_loaders), tuple(paths_loaders)


def _select_loader(
    path: str,
    loaders: Iterable[_RegisteredLoader[LoaderT]],
    *,
    kind: str,
) -> _RegisteredLoader[LoaderT] | None:
    """Select the sole matching loader, rejecting ambiguous claims."""
    matching = []
    for registered in loaders:
        try:
            result = registered.loader.matches(path)
        except Exception as exc:
            raise RuntimeError(f"{kind} loader {registered.source!r} failed while matching {path!r}.") from exc
        if not isinstance(result, bool):
            raise TypeError(
                f"{kind} loader {registered.source!r} returned {type(result).__name__} from matches(); expected bool."
            )
        if result:
            matching.append(registered)
    if len(matching) > 1:
        sources = ", ".join(repr(registered.source) for registered in matching)
        raise ValueError(f"Multiple {kind.lower()} loaders matched {path!r}: {sources}.")
    return matching[0] if matching else None


def _as_local_path(path: LocalPath, *, loader: str) -> str:
    """Validate and normalize a loader result to an absolute local path string."""
    try:
        raw_path = os.fspath(path)
    except TypeError as exc:
        raise TypeError(f"Path loader {loader!r} did not return a local path.") from exc
    if not isinstance(raw_path, str):
        raise TypeError(f"Path loader {loader!r} returned bytes; expected a local string path.")
    if raw_path.startswith("@") or _URI_PREFIX.match(raw_path) is not None:
        raise ValueError(f"Path loader {loader!r} returned non-local path {raw_path!r}.")
    return _resolve_local_path(raw_path)


def _select_path_loader(path: str) -> _RegisteredLoader[PathLoader]:
    path_loaders, _ = _entrypoint_loaders()
    registered = _select_loader(path, path_loaders, kind="Path")
    if registered is None:
        raise ValueError(f"No path loader could resolve {path!r} to a local path.")
    return registered


def _load_path(path: str, registered: _RegisteredLoader[PathLoader]) -> str:
    return _as_local_path(registered.loader.load(path), loader=registered.source)


def resolve_path(path: str) -> str:
    """Resolve one string to one local machine path."""
    return _load_path(path, _select_path_loader(path))


def _iter_local_paths(loaded: Iterable[LocalPath], *, source: str) -> Iterator[str]:
    """Normalize a loader iterable one item at a time."""
    if isinstance(loaded, (str, bytes, os.PathLike)):
        raise TypeError(f"Paths loader {source!r} returned one path instead of an iterable.")
    try:
        iterator = iter(loaded)
    except TypeError as exc:
        raise TypeError(f"Paths loader {source!r} did not return an iterable.") from exc
    return (_as_local_path(item, loader=source) for item in iterator)


def resolve_paths(path: str) -> Iterator[str]:
    """Resolve one string to an iterator of local machine paths.

    Loader selection and ``load()`` happen immediately. The returned iterable is
    consumed and each result is normalized only when the iterator advances. Plural
    loaders are independent: this function never invokes a singular loader.
    """
    _, paths_loaders = _entrypoint_loaders()
    registered = _select_loader(path, paths_loaders, kind="Paths")
    if registered is None:
        raise ValueError(f"No paths loader could resolve {path!r} to local paths.")
    return _iter_local_paths(registered.loader.load(path), source=registered.source)


def resolve_data_file_path(file_path: str) -> str:
    """Compatibility wrapper for callers that resolve one data file."""
    return resolve_path(file_path)


def resolve_asset_path(asset_file: str, asset_root: str | None) -> str:
    """Resolve an asset path, optionally rooted under ``asset_root``.

    The selected loader may opt into asset-root joining by defining
    ``accepts_asset_root(path)``. Holosoma's local loader opts in for relative
    filesystem paths; package, remote, and extension paths remain self-locating.
    """
    registered = _select_path_loader(asset_file)
    accepts_asset_root = getattr(registered.loader, "accepts_asset_root", None)
    if asset_root and callable(accepts_asset_root) and accepts_asset_root(asset_file):
        return resolve_path(f"{asset_root.rstrip('/')}/{asset_file}")
    return _load_path(asset_file, registered)
