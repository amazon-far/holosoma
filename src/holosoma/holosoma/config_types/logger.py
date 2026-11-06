from __future__ import annotations

from dataclasses import field
from typing import Annotated, Literal

from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from holosoma.config_types.video import VideoConfig


@dataclass(frozen=True, config=ConfigDict(extra="ignore"))
class DisabledLoggerConfig:
    """Configuration for disabled Weights & Biases logging.

    Note: TensorBoard is always enabled. This only disables wandb integration.
    """

    type: Literal["disabled"] = "disabled"
    """Logger type identifier."""

    # Video recording configuration. Set to disabled by default when logger is disabled.
    video: VideoConfig = field(default_factory=lambda: VideoConfig(enabled=False))
    """Comprehensive video recording configuration; disabled by default when logger is disabled."""

    headless_recording: bool = False
    """Enable video recording in headless mode (saves to local directory)."""

    # Directory settings
    base_dir: str = "logs"
    """Base directory for all logs and outputs."""


@dataclass(frozen=True)
class WandbLoggerConfig:
    """Configuration for Weights & Biases logging.

    Note: TensorBoard is always enabled and configured separately.
    This config only controls Weights & Biases integration.
    """

    type: Literal["wandb"] = "wandb"
    """Logger type identifier."""

    mode: Literal["online", "offline"] = "online"
    """Logging mode for wandb (online or offline sync)."""

    entity: str | None = None
    """Weights & Biases entity (team or user)."""

    project: str | None = None
    """Project name to use when logging."""

    name: str | None = None
    """Optional override for the run name."""

    group: str | None = None
    """Optional run group."""

    id: str | None = None
    """Optional run ID (used for resume)."""

    tags: tuple[str, ...] = ()
    """Optional tags to attach to the run."""

    dir: str | None = None
    """Directory to store wandb metadata locally."""

    resume: bool | None | Literal["allow", "never", "must", "auto"] = None
    """Resume behaviour passed directly to wandb.init."""

    # Video recording configuration
    video: VideoConfig = field(default_factory=VideoConfig)
    """Video recording configuration."""

    headless_recording: bool = False
    """Enable video recording in headless mode (saves to local directory and logs to wandb).
       Kept for backwards compatibility, overrides video.enabled.
    """

    # Directory settings
    base_dir: str = "logs"
    """Base directory for all logs and outputs."""


# Union type for logger configs
# Note: TensorBoard is always enabled. This config only controls Weights & Biases.
LoggerConfig = Annotated[DisabledLoggerConfig | WandbLoggerConfig, Field(discriminator="type")]
