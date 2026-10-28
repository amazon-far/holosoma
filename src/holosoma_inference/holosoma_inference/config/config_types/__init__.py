"""Type definitions for holosoma_inference configuration system."""

from .inference import InferenceConfig
from .observation import ObservationConfig
from .robot import RobotConfig
from .task import TaskConfig

__all__ = [
    "InferenceConfig",
    "ObservationConfig",
    "RobotConfig",
    "TaskConfig",
]
