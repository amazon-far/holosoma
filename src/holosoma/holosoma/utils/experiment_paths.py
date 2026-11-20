"""Utility functions for computing experiment directory paths."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from holosoma.config_types.logger import LoggerConfig


def get_timestamp() -> str:
    """Get current timestamp in experiment format."""
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")


def get_experiment_dir(
    logger_config: LoggerConfig,
    project: str,
    name: str,
    timestamp: str | None = None,
    task_name: str = "locomotion",
) -> Path:
    """Compute the experiment directory a run is filed under.

    Parameters
    ----------
    logger_config : LoggerConfig
        Logger configuration (for ``base_dir`` and, for W&B, ``group``).
    project : str
        Project the run is filed under (``ExperimentConfig.project``).
    name : str
        Name the run is filed under (``ExperimentConfig.name``).
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
    >>> exp_dir = get_experiment_dir(logger_cfg, "my_project", "my_run", "20250115_143022", "locomotion")
    >>> # Result: logs/my_project/20250115_143022-my_run-locomotion
    """
    if timestamp is None:
        timestamp = get_timestamp()
    group = getattr(logger_config, "group", None)
    return Path(logger_config.base_dir) / project / f"{timestamp}-{name}-{group or task_name}"


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
