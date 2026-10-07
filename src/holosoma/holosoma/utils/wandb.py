"""Utilities for interacting with Weights & Biases."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import ModuleType

from loguru import logger

# Global wandb module reference for lazy loading
_wandb: ModuleType | None = None


def get_wandb() -> ModuleType:
    """Lazy import wandb to avoid import overhead and conflicts."""
    global _wandb  # noqa: PLW0603
    if _wandb is None:
        import wandb

        _wandb = wandb
    return _wandb


def parse_wandb_uri(uri: str) -> tuple[str, str]:
    """Parse wandb:// URI into run_path and file_name.

    Format: wandb://entity/project/run_id/file_name
    or: wandb://entity/project/runs/run_id/file_name
    """
    if not uri.startswith("wandb://"):
        raise ValueError(f"Not a wandb URI: {uri}")

    remainder = uri[len("wandb://") :]
    parts = remainder.split("/")

    if len(parts) < 3:
        raise ValueError(f"Invalid wandb URI: {uri}. Expected format: wandb://entity/project/run_id/file_name")

    entity, project = parts[0], parts[1]
    run_id_index = 2
    if len(parts) > 3 and parts[2] == "runs":
        run_id_index = 3
    if run_id_index >= len(parts):
        raise ValueError(f"Invalid wandb URI: {uri}. Missing run_id")

    run_id = parts[run_id_index]
    file_name_parts = parts[run_id_index + 1 :]

    if not file_name_parts:
        raise ValueError(f"Invalid wandb URI: {uri}. Missing file name")

    wandb_run_path = f"{entity}/{project}/{run_id}"
    file_name = "/".join(file_name_parts)

    return wandb_run_path, file_name


def download_wandb_file(uri: str, cache_path: Path) -> None:
    """Download a file from W&B with race condition protection.

    Parameters
    ----------
    uri : str
        W&B URI in format: wandb://entity/project/run_id/file_name
    cache_path : Path
        Local path where file should be cached
    """
    from holosoma.utils.file_cache import _is_cache_valid

    # Double-check: Another process may have cached it while we were checking
    if _is_cache_valid(cache_path):
        logger.debug(f"Cache appeared during check, using it: {cache_path}")
        return

    wandb = get_wandb()
    run_path, file_name = parse_wandb_uri(uri)

    logger.info(f"Downloading {file_name} from W&B run {run_path}...")

    api = wandb.Api()
    run = api.run(run_path)

    # Ensure cache directory exists
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Download to temp directory first, then move to cache
    with tempfile.TemporaryDirectory() as temp_dir:
        file_obj = run.file(file_name)
        file_obj.download(root=temp_dir, replace=True)

        temp_file = Path(temp_dir) / file_name
        if not temp_file.exists():
            raise FileNotFoundError(f"Downloaded file not found: {temp_file}")

        # Atomic move to cache location
        import os

        # Use process-specific temp name to avoid conflicts
        temp_cache = cache_path.with_suffix(f".tmp.{os.getpid()}")
        try:
            temp_file.rename(temp_cache)
            # Try atomic rename
            try:
                temp_cache.rename(cache_path)
            except FileExistsError:
                # Another process beat us to it, clean up our temp file
                temp_cache.unlink()
                logger.debug("Another process cached the file first")
        except Exception:
            # Clean up on error
            if temp_cache.exists():
                temp_cache.unlink()
            raise
