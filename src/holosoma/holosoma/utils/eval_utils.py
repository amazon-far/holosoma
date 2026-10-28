from __future__ import annotations

import copy
import dataclasses
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, TypedDict, cast

import yaml
from loguru import logger
from omegaconf import DictConfig
from pydantic.dataclasses import dataclass
from tqdm import tqdm

# CONFIG_NAME is "holosoma_config.yaml" - the primary configuration file for Holosoma
# This file contains all settings for training and evaluation of models
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_values.simulator import SIMULATOR_REGISTRY
from holosoma.utils.config_utils import CONFIG_NAME
from holosoma.utils.file_cache import get_cached_file_path
from holosoma.utils.logging import LoguruLoggingBridge
from holosoma.utils.path import resolve_path
from holosoma.utils.safe_torch_import import torch

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
    """Path to a local checkpoint file, or W&B URI in the format `wandb://<entity>/<project>/<run_id>[/<checkpoint_name>]`."""


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
        checkpoint_path = Path(resolve_path(checkpoint_str))
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
        return _experiment_config_from_saved(yaml.safe_load(f)), wandb_run_path


def _load_config_from_checkpoint(checkpoint_path: Path) -> tuple[ExperimentConfig, str | None]:
    """Attempt to load the serialized ExperimentConfig from a checkpoint file."""

    checkpoint_contents = torch.load(checkpoint_path, map_location="cpu")
    config_data = checkpoint_contents["experiment_config"]
    return _experiment_config_from_saved(config_data), checkpoint_contents.get("wandb_run_path")


def _experiment_config_from_saved(config_data: dict[str, Any]) -> ExperimentConfig:
    """Transition historical saved fields; CLI/YAML inputs keep the current schema."""
    data = _drop_retired_isaacgym_fields(config_data)
    simulator = data.get("simulator", {}).get("config", {})
    if "mujoco_warp" in simulator:
        native = simulator["sim"].setdefault("mujoco", {})
        warp = native.setdefault("warp", {})
        for name, value in simulator.pop("mujoco_warp").items():
            if name in warp and warp[name] is not None and value is not None and warp[name] != value:
                raise ValueError(f"Saved config has conflicting mujoco_warp.{name} and sim.mujoco.warp.{name}")
            if warp.get(name) is None:
                warp[name] = value
    native = simulator.get("sim", {}).get("mujoco", {})
    gravity = native.pop("gravity", None)
    if gravity is not None:
        extra = native.setdefault("extra", {})
        if "gravity" in extra and extra["gravity"] != gravity:
            raise ValueError("Saved config has conflicting sim.mujoco.gravity and sim.mujoco.extra.gravity")
        extra["gravity"] = gravity
    return ExperimentConfig(**data)


def replace_retired_simulator(eval_cfg: ExperimentConfig, args: list[str]) -> ExperimentConfig:
    """Swap a saved Isaac Gym simulator for the one a ``simulator:<name>`` CLI token selects.

    tyro attaches a default that matches no registered preset to the most similar one, so a saved
    Isaac Gym simulator would take over the ``isaacsim`` slot and ``simulator:isaacsim`` would
    silently keep Isaac Gym. Starting from the selected preset instead makes every override take
    effect. Without a token the saved simulator is kept, and startup reports it as unsupported.
    """
    if eval_cfg.simulator.config.name != "isaacgym":
        return eval_cfg
    selected = [arg.split(":", 1)[1] for arg in args if arg.startswith("simulator:")]
    if not selected or selected[-1] not in SIMULATOR_REGISTRY:
        return eval_cfg
    return dataclasses.replace(eval_cfg, simulator=SIMULATOR_REGISTRY[selected[-1]])


def _drop_retired_isaacgym_fields(config_data: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a saved config without the Isaac Gym fields the schema no longer has.

    ``to_serializable_dict()`` writes every backend's sub-config, active or not, so configs saved
    before Isaac Gym was retired carry ``isaacgym`` blocks even when they ran on Isaac Sim or
    MuJoCo. ``PhysicsConfig`` and ``CameraSensorConfig`` forbid extra fields, so those blocks would
    otherwise reject the whole config. Only the known schema paths are touched; the selected
    simulator and every other backend's settings are kept as saved.
    """
    data = copy.deepcopy(config_data)

    def _drop(block: Any, key: str = "isaacgym") -> None:
        if isinstance(block, dict):
            block.pop(key, None)

    robot = data.get("robot") or {}
    _drop(robot, "apply_dof_armature_in_isaacgym")
    _drop((robot.get("asset") or {}).get("link_physics"))

    scene = data.get("scene") or {}
    for rigid_object in (scene.get("rigid_objects") or {}).values():
        _drop(rigid_object.get("physics"))
    for scene_file in (scene.get("scene_files") or {}).values():
        for pattern in (scene_file.get("object_configs") or {}).values():
            _drop(pattern.get("physics"))

    for sensor in (data.get("sensor") or {}).values():
        _drop(sensor)

    for terms in (data.get("randomization") or {}).values():
        if isinstance(terms, dict):
            for term in terms.values():
                if isinstance(term, dict):
                    _drop((term.get("params") or {}).get("material"))

    return data


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
        checkpoint_dir = Path(resolve_path(override_config.checkpoint_dir))
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
        # 1. Cache globally (fast on repeated access)
        cached_path = get_cached_file_path(checkpoint)
        logger.info(f"Checkpoint cached at: {cached_path}")

        # 2. Extract original filename from W&B URI to preserve it in log_dir
        run_path, artifact_path = _parse_wandb_reference(checkpoint)
        if artifact_path:
            checkpoint_filename = Path(artifact_path).name
        else:
            # Fallback to cache name if no artifact path in URI
            checkpoint_filename = Path(cached_path).name

        # 3. Copy to log_dir for backward compatibility. Prefix with the run_id so
        # checkpoints with the same filename from different runs (e.g. several
        # distillation teachers) don't overwrite each other.
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

    # Any non-W&B path supported by the generic loader system.
    return Path(resolve_path(checkpoint))
