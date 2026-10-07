from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path
from typing import TypedDict, cast

import yaml
from loguru import logger
from omegaconf import DictConfig
from pydantic.dataclasses import dataclass
from tqdm import tqdm

# CONFIG_NAME is "holosoma_config.yaml" - the primary configuration file for Holosoma
# This file contains all settings for training and evaluation of models
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.utils.config_utils import CONFIG_NAME
from holosoma.utils.file_cache import get_cached_file_path
from holosoma.utils.logging import LoguruLoggingBridge
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.simulator_config import SimulatorType, get_simulator_type

_WANDB_PREFIX = "wandb://"
_WANDB_REFERENCE_FORMAT = f"{_WANDB_PREFIX}<entity>/<project>/<run_id>/[<artifact_name>]"


def _parse_wandb_reference(reference: str) -> tuple[str, str | None]:
    """Split a wandb:// URI into run path and optional artifact/checkpoint name."""

    remainder = reference[len(_WANDB_PREFIX) :]
    parts = remainder.split("/")
    if len(parts) < 3:
        raise ValueError(f"Invalid wandb URI: {reference}. Expected format {_WANDB_REFERENCE_FORMAT}")
    entity, project = parts[0], parts[1]
    run_id_index = 2
    if len(parts) > 3 and parts[2] == "runs":
        run_id_index = 3
    if run_id_index >= len(parts):
        raise ValueError(f"Invalid wandb URI: {reference}. Expected format {_WANDB_REFERENCE_FORMAT}")
    run_id = parts[run_id_index]
    artifact_start = run_id_index + 1
    wandb_run_path = f"{entity}/{project}/{run_id}"
    artifact_path = "/".join(parts[artifact_start:]) or None
    return wandb_run_path, artifact_path


def init_eval_logging() -> None:
    logger.remove()

    # Get log level from LOGURU_LEVEL environment variable or use INFO as default
    console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
    logger.add(sys.stdout, level=console_log_level, colorize=True)

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().addHandler(LoguruLoggingBridge())


@dataclass(frozen=True)
class CheckpointConfig:
    checkpoint: str | None = None
    """Path to a local checkpoint file, or W&B URI in the format
    ``wandb://<entity>/<project>/<run_id>[/<checkpoint_name>]``. If the
    trailing ``<checkpoint_name>`` is omitted, the run's latest ``model_N.pt``
    is picked automatically."""


def load_saved_experiment_config(checkpoint_cfg: CheckpointConfig) -> tuple[ExperimentConfig, str | None]:
    """Load checkpoint configuration from either W&B run or local checkpoint.

    Returns
    -------
    (ExperimentConfig, str | None)
        Loaded experiment config and the originating wandb run path, if available.
    """

    checkpoint = checkpoint_cfg.checkpoint

    if checkpoint is None:
        raise ValueError("No checkpoint provided")

    checkpoint_str = str(checkpoint)
    if not checkpoint_str.startswith(_WANDB_PREFIX):
        checkpoint_path = Path(checkpoint_str).expanduser()
        config, stored_wandb_path = _load_config_from_checkpoint(checkpoint_path)
        if stored_wandb_path:
            logger.info(f"Checkpoint originated from W&B run: {stored_wandb_path}")
        logger.info(f"Loaded experiment config from checkpoint: {checkpoint_path}")
        return config, stored_wandb_path

    wandb_run_path, _ = _parse_wandb_reference(checkpoint_str)

    # Construct wandb:// URI for the config file and use caching
    config_uri = f"{_WANDB_PREFIX}{wandb_run_path}/{CONFIG_NAME}"
    cached_config_path = get_cached_file_path(config_uri)

    with open(cached_config_path) as f:
        return ExperimentConfig(**yaml.safe_load(f)), wandb_run_path


def _load_config_from_checkpoint(checkpoint_path: Path) -> tuple[ExperimentConfig, str | None]:
    """Attempt to load the serialized ExperimentConfig from a checkpoint file."""

    checkpoint_contents = torch.load(checkpoint_path, map_location="cpu")
    config_data = checkpoint_contents["experiment_config"]
    return ExperimentConfig(**config_data), checkpoint_contents.get("wandb_run_path")


class CheckpointMetadata(TypedDict):
    file_name: str
    """Name of the checkpoint file."""

    global_step: int
    """Global step of the checkpoint."""

    train_runtime: float | None
    """Number of seconds that have elapsed since the start of training."""

    num_samples: int | None
    """Number of training samples that have been collected up to the checkpoint."""


def get_all_checkpoint_metadata(override_config: DictConfig) -> list[CheckpointMetadata]:
    """Get all checkpoint names and their global steps from either W&B run or local directory.

    Parameters
    ----------
    override_config : DictConfig
        Configuration object containing:
        - wandb_run_path: str | None
            Path to the W&B run (e.g., 'username/project/run_id'). If None, checkpoint_dir must be provided.
        - checkpoint_dir: str | None
            Path to local directory containing checkpoints. If None, wandb_run_path must be provided.
        - checkpoint_names: list[str] | None
            List of checkpoint names to evaluate. If None, all checkpoints will be evaluated.

    Returns
    -------
    list[CheckpointMetadata]
        List of checkpoint metadata.

    Raises
    ------
    ValueError
        If neither wandb_run_path nor checkpoint_dir is provided.
    """
    import wandb

    def extract_global_step(filename: str) -> int | None:
        """Extract global step from checkpoint filename."""
        match = re.match(r"model_(\d+)\.pt", filename)
        if match:
            return int(match.group(1))
        return None

    checkpoint_metadata: list[CheckpointMetadata]
    if override_config.get("wandb_run_path", None) is not None:
        api = wandb.Api()
        run = api.run(override_config.wandb_run_path)
        # Get all files in the run
        files = run.files()
        # Filter for checkpoint files (assuming they end with .pt)
        checkpoint_names = [f.name for f in files if f.name.endswith(".pt") and extract_global_step(f.name) is not None]
        runtimes: dict[int, float] = {}
        num_samples: dict[int, int] = {}
        logger.info("Scanning W&B history to extract runtime data...")
        for hist in tqdm(run.scan_history(keys=["_runtime", "global_step", "Train/num_samples"])):
            hist_global_step = hist["global_step"]
            hist_runtime = hist["_runtime"]
            hist_num_samples = hist["Train/num_samples"]
            if hist_global_step is not None and hist_runtime is not None:
                runtimes[hist_global_step] = min(runtimes.get(hist_global_step, float("inf")), hist_runtime)
            if hist_global_step is not None and hist_num_samples is not None:
                num_samples[hist_global_step] = hist_num_samples
        checkpoint_metadata = []
        for checkpoint_name in checkpoint_names:
            checkpoint_global_step = extract_global_step(checkpoint_name)
            assert checkpoint_global_step is not None
            if checkpoint_global_step not in runtimes:
                logger.warning(
                    f"Checkpoint {checkpoint_name} and the corresponding global step {checkpoint_global_step} "
                    "has no _runtime data in W&B. Setting train_runtime to 0."
                )
            if checkpoint_global_step not in num_samples:
                logger.warning(
                    f"Checkpoint {checkpoint_name} and the corresponding global step {checkpoint_global_step} "
                    "has no Train/num_samples data in W&B. Setting num_samples to 0."
                )
            checkpoint_metadata.append(
                {
                    "file_name": checkpoint_name,
                    "global_step": checkpoint_global_step,
                    "train_runtime": runtimes.get(checkpoint_global_step, 0.0),
                    "num_samples": num_samples.get(checkpoint_global_step, 0),
                }
            )
    elif override_config.get("checkpoint_dir", None) is not None:
        checkpoint_dir = Path(override_config.checkpoint_dir)
        # Get all checkpoint files in the directory
        checkpoint_names = [f.name for f in checkpoint_dir.glob("*.pt") if extract_global_step(f.name) is not None]
        checkpoint_metadata = [
            {
                "file_name": checkpoint_name,
                "global_step": cast("int", extract_global_step(checkpoint_name)),
                "train_runtime": None,
                "num_samples": None,
            }
            for checkpoint_name in checkpoint_names
        ]
    else:
        raise ValueError("No checkpoint directory or wandb run path provided")

    if override_config.get("checkpoint_names", None) is not None:
        checkpoint_metadata = [
            metadata for metadata in checkpoint_metadata if metadata["file_name"] in override_config.checkpoint_names
        ]

    return sorted(checkpoint_metadata, key=lambda x: x["global_step"])


def load_checkpoint(checkpoint: str, log_dir: str) -> Path:
    """Download checkpoint from W&B or use local checkpoint.

    For W&B checkpoints, files are cached globally in ~/.cache/holosoma/file_cache/
    for performance, but also copied to log_dir for backward compatibility with
    downstream tools and users who expect checkpoints in log_dir.

    Parameters
    ----------
    checkpoint : str
        W&B checkpoint URI or path to local checkpoint file.
    log_dir : str
        Directory to save downloaded checkpoint. W&B checkpoints are copied here
        from the global cache.

    Returns
    -------
    Path
        Path to the checkpoint file in log_dir (for W&B) or original path (for local).
    """
    import shutil

    if checkpoint.startswith(_WANDB_PREFIX):
        # Resolve "wandb://entity/project/run_id" (no trailing filename) to the
        # latest model_N.pt on the run. Keeps get_cached_file_path's contract
        # (needs a filename) intact by rewriting the URI here.
        run_path, artifact_path = _parse_wandb_reference(checkpoint)
        if artifact_path is None:
            from holosoma.utils.wandb_registry import get_wandb_run_and_file

            _, resolved_run, fname = get_wandb_run_and_file(run_path)
            checkpoint = f"{_WANDB_PREFIX}{resolved_run}/{fname}"
            logger.info(f"Auto-resolved latest checkpoint: {checkpoint}")

        # 1. Cache globally (fast on repeated access)
        cached_path = get_cached_file_path(checkpoint)
        logger.info(f"Checkpoint cached at: {cached_path}")

        # 2. Extract original filename from W&B URI to preserve it in log_dir
        _, artifact_path = _parse_wandb_reference(checkpoint)
        if artifact_path:
            checkpoint_filename = Path(artifact_path).name
        else:
            # Fallback to cache name if no artifact path in URI
            checkpoint_filename = Path(cached_path).name

        # 3. Copy to log_dir for backward compatibility. Prefix with the run_id
        # so multi-teacher checkpoints from different runs don't collide on the
        # same filename (every run's latest is "model_N.pt", so without a
        # run-id prefix, loading N teachers would overwrite each other and all
        # teachers end up being the *last* one loaded — a silent catastrophic
        # bug for multi-teacher distillation).
        run_id = run_path.rsplit("/", 1)[-1]
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)
        log_dir_checkpoint = log_dir_path / f"{run_id}_{checkpoint_filename}"

        # Copy if not already there or outdated
        if not log_dir_checkpoint.exists() or log_dir_checkpoint.stat().st_mtime < Path(cached_path).stat().st_mtime:
            shutil.copy2(cached_path, log_dir_checkpoint)
            logger.info(f"Copied checkpoint to log_dir: {log_dir_checkpoint}")
        else:
            logger.info(f"Checkpoint already in log_dir: {log_dir_checkpoint}")

        return log_dir_checkpoint

    # Local file path
    return Path(checkpoint)


def init_sim_imports(tyro_config: ExperimentConfig):
    """Initialize simulator imports - DEPRECATED.

    This function is deprecated in favor of the more focused functions in sim_utils.py.
    Use setup_simulation_environment() for new code.

    Parameters
    ----------
    tyro_config : ExperimentConfig
        Configuration containing simulator settings.

    Returns
    -------
    Any | None
        Simulation app instance for IsaacSim, None for other simulators.
    """
    from holosoma.utils.sim_utils import setup_isaaclab_launcher, setup_simulator_imports

    # Use the new focused functions
    setup_simulator_imports(tyro_config)

    simulator_type = get_simulator_type()
    if simulator_type == SimulatorType.ISAACSIM:
        return setup_isaaclab_launcher(tyro_config)

    # For other simulators, no app is needed
    return None
