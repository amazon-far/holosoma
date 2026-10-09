"""File caching utilities for remote resources (S3, W&B, HTTP)."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any, cast

from loguru import logger

from holosoma.utils.wandb import download_wandb_file

# Lazy imports for optional dependencies
_smart_open = None


def _get_smart_open() -> Any:
    """Lazy import smart_open."""
    global _smart_open  # noqa: PLW0603
    if _smart_open is None:
        import smart_open

        _smart_open = smart_open
    return _smart_open


# Configuration
def _get_cache_dir() -> Path:
    """Get the cache directory path."""
    cache_dir_str = os.environ.get("HOLOSOMA_CACHE_DIR", "~/.cache/holosoma/file_cache")
    cache_dir = Path(cache_dir_str).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _is_cache_enabled() -> bool:
    """Check if caching is enabled via environment variable."""
    return os.environ.get("HOLOSOMA_CACHE_ENABLED", "true").lower() == "true"


# URI parsing and classification
def _is_remote_uri(uri: str) -> bool:
    """Check if URI is remote (s3://, wandb://, http://, https://)."""
    return uri.startswith(("s3://", "wandb://", "http://", "https://"))


def _get_protocol(uri: str) -> str:
    """Extract protocol from URI (s3, wandb, http, https)."""
    if uri.startswith("s3://"):
        return "s3"
    if uri.startswith("wandb://"):
        return "wandb"
    if uri.startswith("http://"):
        return "http"
    if uri.startswith("https://"):
        return "https"
    return "local"


def _uri_to_hash(uri: str) -> str:
    """Generate a hash from URI for cache file naming."""
    return hashlib.sha256(uri.encode()).hexdigest()


def _get_cache_path(uri: str) -> Path:
    """Get the cache file path for a given URI."""
    protocol = _get_protocol(uri)
    cache_dir = _get_cache_dir()
    protocol_dir = cache_dir / protocol
    protocol_dir.mkdir(parents=True, exist_ok=True)

    # Use hash for filename, preserve extension
    uri_hash = _uri_to_hash(uri)
    extension = Path(uri).suffix or ""
    return protocol_dir / f"{uri_hash}{extension}"


def _get_metadata_path(cache_path: Path) -> Path:
    """Get the metadata file path for a cache file."""
    return cache_path.parent / f"{cache_path.stem}.meta.json"


def _save_metadata(cache_path: Path, uri: str, additional_info: dict[str, Any] | None = None) -> None:
    """Save metadata about the cached file."""
    metadata = {
        "uri": uri,
        "cached_at": time.time(),
        "cached_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "size_bytes": cache_path.stat().st_size if cache_path.exists() else 0,
        "protocol": _get_protocol(uri),
    }
    if additional_info:
        metadata.update(additional_info)

    metadata_path = _get_metadata_path(cache_path)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def _load_metadata(cache_path: Path) -> dict[str, Any] | None:
    """Load metadata about a cached file."""
    metadata_path = _get_metadata_path(cache_path)
    if not metadata_path.exists():
        return None
    try:
        with open(metadata_path) as f:
            return cast("dict[str, Any]", json.load(f))
    except Exception as e:
        logger.warning(f"Failed to load metadata from {metadata_path}: {e}")
        return None


def _is_cache_valid(cache_path: Path) -> bool:
    """Check if a cached file is valid and not expired."""
    if not cache_path.exists():
        return False

    # Check if file has content
    if cache_path.stat().st_size == 0:
        return False

    # Check TTL if metadata exists
    metadata = _load_metadata(cache_path)
    if metadata is not None:
        # Get TTL from env var (default 60 minutes = 1 hour)
        ttl_minutes = float(os.environ.get("HOLOSOMA_CACHE_TTL_MINUTES", "60"))

        # Special value 0 = no expiration (cache forever)
        if ttl_minutes > 0:
            ttl_seconds = ttl_minutes * 60
            cached_at = metadata.get("cached_at", 0)
            age_seconds = time.time() - cached_at

            if age_seconds > ttl_seconds:
                logger.debug(
                    f"Cache expired for {cache_path.name} (age: {age_seconds / 60:.1f}min, TTL: {ttl_minutes}min)"
                )
                return False

    return True


def _download_remote_file(uri: str, cache_path: Path) -> None:
    """Download a remote file using appropriate method based on protocol.

    Implements race condition protection for concurrent downloads:
    1. Double-check pattern: Check cache again before expensive download
    2. Process-specific temp files: Use PID to avoid conflicts
    3. Race-aware rename: Handle FileExistsError gracefully
    """
    protocol = _get_protocol(uri)

    if protocol == "wandb":
        download_wandb_file(uri, cache_path)
    elif protocol in ("s3", "http", "https"):
        # Double-check: Another process may have cached it while we were checking
        if _is_cache_valid(cache_path):
            logger.debug(f"Cache appeared during check, using it: {cache_path}")
            return

        # Use smart_open for S3, HTTP, HTTPS
        smart_open = _get_smart_open()
        logger.info(f"Downloading {uri}...")

        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Use process-specific temp file to avoid conflicts
        import os

        temp_file = cache_path.with_suffix(cache_path.suffix + f".tmp.{os.getpid()}")

        try:
            # Let smart_open handle streaming internally
            with smart_open.open(uri, "rb") as src, open(temp_file, "wb") as dst:
                dst.write(src.read())

            # Race-aware rename: Handle case where another process finished first
            try:
                temp_file.rename(cache_path)
                logger.info(f"Cached {uri} to {cache_path}")
            except FileExistsError:
                # Another process won the race - that's fine!
                logger.debug(f"Another process cached {uri}, using their version")
                # Fall through to cleanup
        finally:
            # Clean up our temp file if it still exists
            if temp_file.exists():
                temp_file.unlink()
    else:
        raise ValueError(f"Unsupported protocol for downloading: {protocol}")


# Public API
def get_cached_file_path(uri: str, *, use_cache: bool = True) -> str:
    """Get a local file path for the given URI, downloading and caching if necessary.

    Supports:
    - s3://bucket/path/to/file.ext
    - wandb://entity/project/run_id/file.ext
    - http://... and https://...
    - /absolute/local/path
    - ./relative/local/path

    Parameters
    ----------
    uri : str
        URI to access
    use_cache : bool, optional
        If False, bypass cache. For remote URIs, downloads to a temporary file.
        For local files, returns the original path. (default: True)

    Returns
    -------
    str
        Local file path

    Examples
    --------
    >>> # S3 file
    >>> path = get_cached_file_path("s3://bucket/data.npz")
    >>> with open(path, "rb") as f:
    ...     data = np.load(f)

    >>> # W&B checkpoint
    >>> path = get_cached_file_path("wandb://entity/project/run_id/model.pt")
    >>> checkpoint = torch.load(path)

    >>> # Local file (unchanged)
    >>> path = get_cached_file_path("/local/path/file.txt")
    >>> # Returns: "/local/path/file.txt"
    """
    # Check if caching is globally enabled
    cache_enabled = _is_cache_enabled() and use_cache

    # Handle local and package-prefixed files through the shared resolver.
    if not _is_remote_uri(uri):
        from holosoma.utils.path import resolve_path

        return resolve_path(uri)

    # Handle remote files with caching disabled
    if not cache_enabled:
        # Download to temporary location
        import os

        fd, temp_path = tempfile.mkstemp(suffix=Path(uri).suffix)
        os.close(fd)  # Close the file descriptor
        temp_file = Path(temp_path)
        _download_remote_file(uri, temp_file)
        return str(temp_file)

    # Handle remote files with caching enabled
    cache_path = _get_cache_path(uri)

    # Check if valid cached version exists
    if _is_cache_valid(cache_path):
        logger.debug(f"Cache hit: {uri} -> {cache_path}")
        return str(cache_path)

    # Download and cache
    logger.info(f"Cache miss: {uri}")
    try:
        _download_remote_file(uri, cache_path)
        _save_metadata(cache_path, uri)
    except Exception as e:
        # Clean up partial download
        if cache_path.exists():
            cache_path.unlink()
        metadata_path = _get_metadata_path(cache_path)
        if metadata_path.exists():
            metadata_path.unlink()
        raise RuntimeError(f"Failed to download and cache {uri}: {e}") from e

    return str(cache_path)


@contextmanager
def cached_open(uri: str, mode: str = "rb", *, use_cache: bool = True, **kwargs: Any) -> Iterator[IO[Any]]:
    """Context manager for opening files with caching.

    Drop-in replacement for smart_open.open() with caching support.

    Parameters
    ----------
    uri : str
        URI to access
    mode : str
        File mode (default: "rb")
    use_cache : bool
        Whether to use cache (default: True)
    **kwargs
        Additional arguments passed to open()

    Yields
    ------
    file object
        Opened file handle

    Examples
    --------
    >>> with cached_open("s3://bucket/data.npz", "rb") as f:
    ...     data = np.load(f)

    >>> with cached_open("wandb://entity/project/run/config.yaml") as f:
    ...     config = yaml.safe_load(f)
    """
    cache_path = get_cached_file_path(uri, use_cache=use_cache)
    with open(cache_path, mode, **kwargs) as f:
        yield f


def clear_cache(uri: str | None = None, protocol: str | None = None) -> None:
    """Clear cache for specific URI, protocol, or entire cache.

    Parameters
    ----------
    uri : str | None
        Specific URI to clear from cache. If None, clears based on protocol or all.
    protocol : str | None
        Protocol to clear (s3, wandb, http, https). If None and uri is None, clears all.

    Examples
    --------
    >>> # Clear specific file
    >>> clear_cache("s3://bucket/large_file.npz")

    >>> # Clear all S3 cached files
    >>> clear_cache(protocol="s3")

    >>> # Clear entire cache
    >>> clear_cache()
    """
    cache_dir = _get_cache_dir()

    if uri is not None:
        # Clear specific URI
        cache_path = _get_cache_path(uri)
        if cache_path.exists():
            cache_path.unlink()
            logger.info(f"Cleared cache for: {uri}")

        metadata_path = _get_metadata_path(cache_path)
        if metadata_path.exists():
            metadata_path.unlink()

    elif protocol is not None:
        # Clear all files for a protocol
        protocol_dir = cache_dir / protocol
        if protocol_dir.exists():
            for file in protocol_dir.iterdir():
                file.unlink()
            logger.info(f"Cleared all {protocol} cached files")

    # Clear entire cache
    elif cache_dir.exists():
        for protocol_dir in cache_dir.iterdir():
            if protocol_dir.is_dir():
                for file in protocol_dir.iterdir():
                    file.unlink()
                protocol_dir.rmdir()
        logger.info("Cleared entire cache")


def get_cache_stats() -> dict[str, Any]:
    """Get statistics about the cache.

    Returns
    -------
    dict
        Dictionary containing cache statistics:
        - total_files: Total number of cached files
        - total_size_bytes: Total size of cached files
        - protocols: Per-protocol statistics
    """
    cache_dir = _get_cache_dir()

    if not cache_dir.exists():
        return {"total_files": 0, "total_size_bytes": 0, "protocols": {}}

    total_files = 0
    total_size = 0
    protocols: dict[str, dict[str, int]] = {}

    for protocol_dir in cache_dir.iterdir():
        if not protocol_dir.is_dir():
            continue

        protocol = protocol_dir.name
        protocol_files = 0
        protocol_size = 0

        for file in protocol_dir.iterdir():
            # Skip metadata files
            if file.suffix == ".json":
                continue

            protocol_files += 1
            protocol_size += file.stat().st_size

        protocols[protocol] = {
            "files": protocol_files,
            "size_bytes": protocol_size,
        }

        total_files += protocol_files
        total_size += protocol_size

    return {
        "total_files": total_files,
        "total_size_bytes": total_size,
        "protocols": protocols,
    }
