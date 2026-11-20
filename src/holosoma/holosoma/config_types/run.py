"""Configuration type for how a run executes: its environments, pacing, naming and logging."""

from __future__ import annotations

import datetime
import warnings
from typing import Annotated

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

import holosoma.config_values.logger
from holosoma.config_types.frequency import DecimationLike, validate_decimation_like
from holosoma.config_types.logger import LoggerConfig
from holosoma.utils.config_registry import UseRegistry

_VIEWER_DT_REMOVAL_DATE = datetime.date(2027, 3, 1)


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class RunConfig:
    """Settings for one run that every workflow reads the same way."""

    num_envs: int = 4096
    """Number of parallel environments."""

    headless: bool = True
    """Run simulation without rendering."""

    seed: int = 42
    """Random seed for reproducibility."""

    torch_deterministic: bool = False
    """Enable PyTorch deterministic mode."""

    device: str | None = None
    """Device for the task. None (the default) resolves from the simulator backend: cuda:0 for
    MuJoCo Warp, cuda:0 for IsaacSim when CUDA is available (else cpu), cpu for classic MuJoCo,
    and the process rank's GPU when distributed. Pass e.g. "cuda:0" to override."""

    time_scale: float = 0.0
    """Sim-time to real-time ratio for the control loop: 1 paces sim time to match real time, 2 runs
    twice as fast, 0 disables pacing entirely (run as fast as possible)."""

    viewer_update_decimation: DecimationLike = "<60Hz"
    """Viewer render rate: a physics-step decimation, or a frequency string resolved against the
    physics rate."""

    viewer_dt: float | None = None
    """Deprecated, removed after 2027-03-01: use ``viewer_update_decimation``. Converted to
    ``">{1 / viewer_dt}Hz"``, the same decimation it gave before."""

    project: str = "default_project"
    """Project the run is filed under: its log and checkpoint directory and its W&B project."""

    name: str = "run"
    """Name the run is filed under: its log and checkpoint directory, and the base of its W&B run name."""

    logger: Annotated[LoggerConfig, UseRegistry(holosoma.config_values.logger.LOGGER_REGISTRY)] = (
        holosoma.config_values.logger.disabled
    )
    """Where the run's metrics and videos go."""

    def __post_init__(self) -> None:
        if self.time_scale < 0:
            raise ValueError(f"run.time_scale must be >= 0, got {self.time_scale}.")
        if self.viewer_dt is not None:
            if self.viewer_dt <= 0:
                raise ValueError(f"run.viewer_dt must be > 0, got {self.viewer_dt}.")
            warnings.warn(
                "run.viewer_dt is deprecated; use run.viewer_update_decimation. It will be removed "
                f"after {_VIEWER_DT_REMOVAL_DATE}.",
                FutureWarning,
                stacklevel=3,
            )
            object.__setattr__(self, "viewer_update_decimation", f">{1 / self.viewer_dt!r}Hz")
            object.__setattr__(self, "viewer_dt", None)
        validate_decimation_like(self.viewer_update_decimation, field="run.viewer_update_decimation")
