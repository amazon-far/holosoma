"""In-process sigma scheduler for reward parameter annealing.

Linearly interpolates reward term parameters (e.g., tracking_sigma) over
a training phase, enabling smooth curriculum transitions without subprocess
restarts.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from loguru import logger


@dataclass
class SigmaScheduleEntry:
    """A single parameter to anneal over the phase."""

    term_name: str
    """Reward term name (e.g., 'tracking_ee_pos')."""

    param_name: str
    """Parameter key within the term's params dict (e.g., 'tracking_sigma').
    When target_field='weight', this is ignored."""

    start_value: float
    """Value at the beginning of the ramp."""

    end_value: float
    """Value at the end of the ramp."""

    ramp_start_pct: float = 0.2
    """Fraction of phase iterations where the ramp begins (before this, hold start_value)."""

    ramp_end_pct: float = 0.7
    """Fraction of phase iterations where the ramp ends (after this, hold end_value)."""

    target_field: str = "params"
    """Which field to anneal: 'params' (default) updates term_cfg.params[param_name],
    'weight' updates term_cfg.weight directly, and 'command' sets attribute
    ``param_name`` on the live command term ``term_name`` (e.g. a curriculum
    scale that the term reads when it samples)."""


@dataclass
class SigmaSchedulerCfg:
    """Configuration for the sigma scheduler."""

    entries: list[SigmaScheduleEntry] = field(default_factory=list)
    """List of parameters to anneal."""

    phase_start_step: int = 0
    """Global step at which this phase started (set at runtime from checkpoint)."""

    phase_total_iterations: int = 50000
    """Total iterations for this phase."""


class SigmaScheduler:
    """Applies linear interpolation to reward params based on training progress.

    Call :meth:`update` at each logging interval. It computes the progress
    fraction within the phase and updates the reward manager's term configs
    accordingly.

    Parameters
    ----------
    cfg : SigmaSchedulerCfg
        Scheduler configuration with entries and phase boundaries.
    env : Any
        Environment instance with a ``reward_manager`` attribute.
    """

    def __init__(self, cfg: SigmaSchedulerCfg, env: Any):
        self.cfg = cfg
        self.env = env
        self._last_logged_step: int = -1

    def update(self, global_step: int) -> None:
        """Compute interpolated values and apply to reward manager.

        Parameters
        ----------
        global_step : int
            Current training iteration (absolute, not relative to phase start).
        """
        if not self.cfg.entries:
            return

        # Compute progress within this phase [0.0, 1.0]
        elapsed = global_step - self.cfg.phase_start_step
        progress = max(0.0, min(1.0, elapsed / max(1, self.cfg.phase_total_iterations)))

        reward_manager = self.env.reward_manager

        for entry in self.cfg.entries:
            # Compute interpolated value
            value = self._interpolate(entry, progress)

            if entry.target_field == "command":
                try:
                    state = self.env.command_manager.get_state(entry.term_name)
                except (AttributeError, KeyError):
                    state = None
                if state is not None and hasattr(state, entry.param_name):
                    setattr(state, entry.param_name, value)
                continue

            # Update the reward term's params or weight
            try:
                term_cfg = reward_manager.get_term_cfg(entry.term_name)
            except KeyError:
                continue

            if entry.target_field == "weight":
                new_cfg = replace(term_cfg, weight=value)
            else:
                # Build new params dict with updated value
                new_params = dict(term_cfg.params)
                new_params[entry.param_name] = value
                new_cfg = replace(term_cfg, params=new_params)
            reward_manager.set_term_cfg(entry.term_name, new_cfg)

        # Log periodically (every 5000 steps)
        if global_step - self._last_logged_step >= 5000:
            self._last_logged_step = global_step
            sample_entry = self.cfg.entries[0]
            sample_value = self._interpolate(sample_entry, progress)
            logger.info(
                f"[SigmaScheduler] step={global_step}, progress={progress:.3f}, "
                f"{sample_entry.term_name}.{sample_entry.param_name}={sample_value:.6f}"
            )

            # Also log to wandb via env.log_dict if available
            if hasattr(self.env, "log_dict"):
                import torch
                for entry in self.cfg.entries:
                    val = self._interpolate(entry, progress)
                    key = f"sigma_schedule/{entry.term_name}_{entry.param_name}"
                    self.env.log_dict[key] = torch.tensor(val, dtype=torch.float)

    @staticmethod
    def _interpolate(entry: SigmaScheduleEntry, progress: float) -> float:
        """Linear interpolation with hold regions before and after the ramp."""
        if progress <= entry.ramp_start_pct:
            return entry.start_value
        if progress >= entry.ramp_end_pct:
            return entry.end_value

        # Linear ramp
        ramp_progress = (progress - entry.ramp_start_pct) / (entry.ramp_end_pct - entry.ramp_start_pct)
        return entry.start_value + (entry.end_value - entry.start_value) * ramp_progress
