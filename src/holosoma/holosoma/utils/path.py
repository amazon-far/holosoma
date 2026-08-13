"""Path resolution utilities for package data files."""

from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 9):
    from importlib.resources import files
else:
    from importlib_resources import files  # type: ignore[import-not-found]


def resolve_data_file_path(file_path: str) -> str:
    """
    Resolve a data file path.

    Handles multiple path formats:
    1. S3 paths: "s3://bucket/path/to/file.npz" -> returned as-is
    2. Package paths: "holosoma/..." (or the "@holosoma/..." alias) -> resolved
       relative to the installed holosoma package via importlib.resources
    3. Absolute paths: "/path/to/file.npz" -> returned as-is
    4. Relative paths: "./data/file.npz" or "../data/file.npz" -> resolved relative to CWD

    Args:
        file_path: The path to resolve

    Returns:
        The resolved absolute path as a string

    Examples:
        >>> # Package data (both forms resolve to the same location)
        >>> resolve_data_file_path("holosoma/data/robots/g1/g1_29dof.xml")
        '/path/to/installed/holosoma/data/robots/g1/g1_29dof.xml'
        >>> resolve_data_file_path("@holosoma/data/robots/g1/g1_29dof.xml")
        '/path/to/installed/holosoma/data/robots/g1/g1_29dof.xml'

        >>> # User's custom file (absolute)
        >>> resolve_data_file_path("/home/user/my_motions/custom.npz")
        '/home/user/my_motions/custom.npz'

        >>> # User's custom file (relative to CWD)
        >>> resolve_data_file_path("./my_data/custom.npz")
        '/current/working/dir/my_data/custom.npz'
    """
    # 1. If it's an S3 path, return as-is
    if file_path.startswith("s3://"):
        return file_path

    # 2. Package path. "@holosoma/..." is a spelling alias for "holosoma/...";
    #    normalize it so both forms resolve via the same importlib.resources path
    #    (robust for namespace/zipped packages, unlike __file__ arithmetic).
    if file_path.startswith("@holosoma/"):
        file_path = file_path[len("@") :]  # "@holosoma/..." -> "holosoma/..."
    if file_path.startswith("holosoma/"):
        suffix = file_path[len("holosoma") :].lstrip("/")  # path within the holosoma package
        base = files("holosoma")
        return str(base / suffix) if suffix else str(base)

    # 3. If it's an absolute path, return as-is
    path_obj = Path(file_path)
    if path_obj.is_absolute():
        return file_path

    # 4. Otherwise, resolve relative path to absolute (relative to CWD)
    resolved = path_obj.resolve()
    return str(resolved)


def resolve_asset_path(asset_file: str, asset_root: str | None) -> str:
    """Resolve an asset path, optionally rooted under ``asset_root``.

    A package path ("holosoma/..." or "@holosoma/...") or an absolute/S3 path is
    self-locating and resolved directly, ignoring ``asset_root`` — joining a root
    onto it would turn it into a bogus filesystem path. Only a plain relative path is
    joined onto ``asset_root`` (when one is given) before resolution. The self-locating
    cases are exactly the non-CWD-relative branches of :func:`resolve_data_file_path`.
    """
    is_self_locating = asset_file.startswith(("holosoma/", "@holosoma/", "s3://")) or Path(asset_file).is_absolute()
    if asset_root and not is_self_locating:
        asset_file = f"{asset_root.rstrip('/')}/{asset_file}"
    return resolve_data_file_path(asset_file)
