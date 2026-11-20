"""Early stopping checker for training phases.

Monitors a metric from the environment's log_dict and signals graduation
when the metric sustains a threshold for a specified patience window.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class EarlyStopCfg:
    """Configuration for early phase termination."""

    metric: str = "average_episode_length"
    """Key in env.log_dict to monitor."""

    threshold: float = 990.0
    """Graduate when metric crosses this value."""

    direction: str = "above"
    """'above' = graduate when metric >= threshold, 'below' = when metric <= threshold."""

    patience: int = 200
    """Consecutive checks meeting condition before graduation."""

    min_iterations: int = 3000
    """Floor: early stop is not allowed before this many iterations."""


class EarlyStopChecker:
    """Checks a metric at each logging interval and tracks patience.

    Parameters
    ----------
    cfg : EarlyStopCfg
        Early stop configuration.
    log_dir : str or Path
        Directory where the graduation marker and checkpoint will be written.
    """

    def __init__(self, cfg: EarlyStopCfg, log_dir: str | Path):
        self.cfg = cfg
        self.log_dir = Path(log_dir)
        self._consecutive_count: int = 0
        self._graduated: bool = False

    @property
    def graduated(self) -> bool:
        return self._graduated

    def check(self, global_step: int, log_dict: dict[str, Any]) -> bool:
        """Check the metric and return True if graduation condition is met.

        Parameters
        ----------
        global_step : int
            Current training iteration.
        log_dict : dict
            Environment's log dictionary containing the monitored metric.

        Returns
        -------
        bool
            True if the phase should terminate early.
        """
        if self._graduated:
            return True

        if global_step < self.cfg.min_iterations:
            self._consecutive_count = 0
            return False

        metric_key = self.cfg.metric
        if metric_key not in log_dict:
            return False

        value = log_dict[metric_key]
        # Handle torch tensors
        if hasattr(value, "item"):
            value = value.item()
        value = float(value)

        # Check condition
        if self.cfg.direction == "above":
            condition_met = value >= self.cfg.threshold
        else:
            condition_met = value <= self.cfg.threshold

        if condition_met:
            self._consecutive_count += 1
        else:
            self._consecutive_count = 0

        if self._consecutive_count >= self.cfg.patience:
            self._graduated = True
            logger.info(
                f"[EarlyStop] Graduation! {metric_key}={value:.2f} "
                f"(threshold={self.cfg.threshold}, direction={self.cfg.direction}) "
                f"sustained for {self.cfg.patience} checks at step {global_step}."
            )
            # Write marker file so the pipeline orchestrator can detect graduation
            marker_path = self.log_dir / "_graduated.marker"
            marker_path.write_text(f"graduated_at_step={global_step}\nmetric={metric_key}\nvalue={value}\n")
            return True

        return False
