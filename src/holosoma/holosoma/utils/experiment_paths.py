"""Utility functions for computing experiment directory paths."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from holosoma.config_types.experiment import TrainingConfig
    from holosoma.config_types.logger import LoggerConfig


def get_timestamp() -> str:
    """Get current timestamp in experiment format."""
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")


def get_experiment_dir(
    logger_config: LoggerConfig,
    training_config: TrainingConfig,
    timestamp: str | None = None,
    task_name: str = "locomotion",
) -> Path:
    """Compute experiment directory from logger and training config.

    Parameters
    ----------
    logger_config : LoggerConfig
        Logger configuration (WandbLoggerConfig or DisabledLoggerConfig)
    training_config : TrainingConfig
        Training configuration with project/name
    timestamp : str | None
        Timestamp string. If None, generates a new one.
    task_name : str
        Task name for the experiment (e.g., "locomotion", "manipulation")

    Returns
    -------
    Path
        Experiment directory path

    Examples
    --------
    >>> exp_dir = get_experiment_dir(logger_cfg, training_cfg, "20250115_143022", "locomotion")
    >>> # Result: logs/my_project/20250115_143022-my_run-locomotion
    """
    if timestamp is None:
        timestamp = get_timestamp()

    base_dir = Path(logger_config.base_dir)

    # Fallback chain: logger config → training config → default
    project = getattr(logger_config, "project", None) or training_config.project or "default_project"
    name = getattr(logger_config, "name", None) or training_config.name or "run"

    # Build structured path if we have any project/name info
    if project or name:
        date_str = timestamp.split("_")[0] if "_" in timestamp else timestamp
        exp_name = f"{date_str}_{name}"
        return base_dir / project / exp_name

    # Fallback to simple structure
    return base_dir / "runs" / timestamp


def get_output_dir(experiment_dir: Path) -> Path:
    """Get output directory from experiment directory.

    Parameters
    ----------
    experiment_dir : Path
        Experiment directory path

    Returns
    -------
    Path
        Output directory path (experiment_dir/output)
    """
    return experiment_dir / "output"


def get_video_dir(experiment_dir: Path) -> Path:
    """Get video directory from experiment directory.

    Parameters
    ----------
    experiment_dir : Path
        Experiment directory path

    Returns
    -------
    Path
        Video directory path (experiment_dir/renderings_training)
    """
    return experiment_dir / "renderings_training"


def get_eval_log_dir(
    logger_config: LoggerConfig, training_config: TrainingConfig, eval_timestamp: str | None = None
) -> Path:
    """Compute evaluation log directory from logger and training config.

    Parameters
    ----------
    logger_config : LoggerConfig
        Logger configuration
    training_config : TrainingConfig
        Training configuration with project name
    eval_timestamp : str | None
        Evaluation timestamp. If None, generates a new one.

    Returns
    -------
    Path
        Evaluation log directory path
    """
    if eval_timestamp is None:
        eval_timestamp = get_timestamp()

    base_dir = Path(logger_config.base_dir).parent / "logs_eval"

    # Use training config for project, with fallback to logger config
    project: str | None = training_config.project
    if not project and hasattr(logger_config, "project"):
        project = logger_config.project

    if project:
        return base_dir / project / eval_timestamp
    return base_dir / eval_timestamp
