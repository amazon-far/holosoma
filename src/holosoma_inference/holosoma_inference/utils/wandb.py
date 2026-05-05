from __future__ import annotations

import re
from pathlib import Path

import wandb

_WANDB_PREFIX = "wandb://"
_WANDB_HTTPS_PATTERN = re.compile(r"https://[^/]+/([^/]+)/([^/]+)/runs/([^/]+)/files/(.+)")
_CACHE_DIR = Path.home() / ".hs_weights"


def load_checkpoint(
    wandb_run_path: str | None,
    checkpoint: str,
    log_dir: str,
) -> Path:
    """Download checkpoint from W&B or use local checkpoint.

    W&B downloads are cached under ``~/.hs_weights/<run_id>/<filename>``; if the
    file is already present the download is skipped. ``log_dir`` is ignored for
    W&B paths but retained for backwards compatibility.

    Parameters
    ----------
    wandb_run_path : str | None
        Path to the W&B run (e.g., 'username/project/run_id'). If None, checkpoint must be provided.
    checkpoint : str
        Name of checkpoint file in W&B run or path to local checkpoint file.
    log_dir : str
        Unused for W&B downloads (kept for backwards compatibility).

    Returns
    -------
    Path
        Path to the downloaded or local checkpoint file.
    """
    del log_dir  # superseded by the on-disk cache under ~/.hs_weights

    wandb_run_id: str | None = None
    if checkpoint.startswith(_WANDB_PREFIX):
        try:
            wandb_entity, wandb_project, wandb_run_id, checkpoint = checkpoint[len(_WANDB_PREFIX) :].split("/", 3)
        except ValueError:
            raise ValueError(
                f"Invalid wandb checkpoint path: {checkpoint}. "
                f"Expected format: {_WANDB_PREFIX}<entity>/<project>/<run_id>/<checkpoint_name>"
            )
        wandb_run_path = f"{wandb_entity}/{wandb_project}/{wandb_run_id}"
    elif match := _WANDB_HTTPS_PATTERN.match(checkpoint):
        wandb_entity, wandb_project, wandb_run_id, checkpoint = match.groups()
        wandb_run_path = f"{wandb_entity}/{wandb_project}/{wandb_run_id}"
    elif wandb_run_path is not None:
        wandb_run_id = wandb_run_path.rsplit("/", 1)[-1]

    if wandb_run_path is not None:
        assert wandb_run_id is not None
        cache_dir = _CACHE_DIR / wandb_run_id
        cache_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = cache_dir / checkpoint
        if checkpoint_path.exists():
            print(f"Using cached checkpoint {checkpoint_path} (run {wandb_run_path})")
            return checkpoint_path

        api = wandb.Api()
        run = api.run(wandb_run_path)
        checkpoint_file = run.file(checkpoint)
        # wandb preserves the file's relative path under `root`, so the final
        # location is `cache_dir / checkpoint`.
        checkpoint_file.download(root=str(cache_dir), replace=True)
        print(f"Finished downloading checkpoint {checkpoint} to {cache_dir} from W&B run {wandb_run_path}")
        return checkpoint_path

    return Path(checkpoint)
