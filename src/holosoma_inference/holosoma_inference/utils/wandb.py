from __future__ import annotations

import os
import re
import shutil
import warnings
from pathlib import Path

import wandb
from loguru import logger

_WANDB_PREFIX = "wandb://"
_WANDB_HTTPS_PATTERN = re.compile(r"https://[^/]+/([^/]+)/([^/]+)/runs/([^/]+)/files/(.+)")
_CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "holosoma_inference" / "weights"


def _resolve_wandb_source(
    wandb_run_path: str | None,
    checkpoint: str,
) -> tuple[str | None, str | None, str]:
    """Normalize checkpoint input into ``(run_path, run_id, filename)``.

    Accepts three input shapes:
      * ``wandb://<entity>/<project>/<run_id>/<filename>`` in ``checkpoint``
      * a W&B HTTPS URL in ``checkpoint``
      * an explicit ``wandb_run_path`` with ``checkpoint`` as the filename

    Returns ``(None, None, checkpoint)`` for local paths.
    """
    if checkpoint.startswith(_WANDB_PREFIX):
        try:
            entity, project, run_id, filename = checkpoint[len(_WANDB_PREFIX) :].split("/", 3)
        except ValueError:
            raise ValueError(
                f"Invalid wandb checkpoint path: {checkpoint}. "
                f"Expected format: {_WANDB_PREFIX}<entity>/<project>/<run_id>/<checkpoint_name>"
            )
        return f"{entity}/{project}/{run_id}", run_id, filename

    if match := _WANDB_HTTPS_PATTERN.match(checkpoint):
        entity, project, run_id, filename = match.groups()
        return f"{entity}/{project}/{run_id}", run_id, filename

    if wandb_run_path is not None:
        run_id = wandb_run_path.rsplit("/", 1)[-1]
        return wandb_run_path, run_id, checkpoint

    return None, None, checkpoint


def _download_atomic(checkpoint_file, filename: str, cache_dir: Path, final_path: Path) -> None:
    """Download ``checkpoint_file`` into a staging dir, then atomically rename.

    Staging under the same cache dir keeps the rename on one filesystem; a
    per-PID suffix avoids collisions between concurrent processes sharing the
    cache. An interrupted download leaves only the staging dir behind, never a
    truncated file at ``final_path``.
    """
    staging_dir = cache_dir / f".tmp.{os.getpid()}"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)
    try:
        checkpoint_file.download(root=str(staging_dir), replace=True)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        staged_file = staging_dir / filename
        staged_file.replace(final_path)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def load_checkpoint(
    wandb_run_path: str | None,
    checkpoint: str,
    log_dir: str | None = None,
) -> Path:
    """Download checkpoint from W&B or use local checkpoint.

    W&B downloads are cached under ``$XDG_CACHE_HOME/holosoma_inference/weights/<run_id>/<filename>``
    (defaulting to ``~/.cache/holosoma_inference/weights``); if the file is
    already present the download is skipped.

    Parameters
    ----------
    wandb_run_path : str | None
        Path to the W&B run (e.g., 'username/project/run_id'). If None, checkpoint must be provided.
    checkpoint : str
        Name of checkpoint file in W&B run or path to local checkpoint file.
    log_dir : str, optional
        Deprecated. Ignored; W&B downloads always go to the cache directory.

    Returns
    -------
    Path
        Path to the downloaded or local checkpoint file.
    """
    if log_dir is not None:
        warnings.warn(
            "load_checkpoint(log_dir=...) is deprecated and ignored; "
            "W&B checkpoints are cached under $XDG_CACHE_HOME/holosoma_inference/weights.",
            DeprecationWarning,
            stacklevel=2,
        )

    run_path, run_id, filename = _resolve_wandb_source(wandb_run_path, checkpoint)
    if run_path is None:
        return Path(filename)

    assert run_id is not None
    cache_dir = _CACHE_DIR / run_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = cache_dir / filename
    if checkpoint_path.exists():
        logger.info(f"Using cached checkpoint {checkpoint_path} (run {run_path})")
        return checkpoint_path

    run = wandb.Api().run(run_path)
    _download_atomic(run.file(filename), filename, cache_dir, checkpoint_path)
    logger.info(f"Finished downloading checkpoint {filename} to {cache_dir} from W&B run {run_path}")
    return checkpoint_path
