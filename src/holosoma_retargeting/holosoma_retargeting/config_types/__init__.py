"""Configuration types for holosoma_retargeting."""

from holosoma_retargeting.config_types.data_conversion import DataConversionConfig
from holosoma_retargeting.config_types.data_type import MotionDataConfig
from holosoma_retargeting.config_types.retargeter import (
    OrientationTrackingConfig,
    RetargeterConfig,
    TennisRacketTrackingConfig,
)
from holosoma_retargeting.config_types.retargeting import (
    ParallelRetargetingConfig,
    RetargetingConfig,
    XsensMorphologyConfig,
)
from holosoma_retargeting.config_types.robot import RobotConfig
from holosoma_retargeting.config_types.task import TaskConfig
from holosoma_retargeting.config_types.viser import ViserConfig, XsensViserConfig

__all__ = [
    "DataConversionConfig",
    "EvaluationConfig",
    "MotionDataConfig",
    "OrientationTrackingConfig",
    "ParallelRetargetingConfig",
    "RetargeterConfig",
    "RetargetingConfig",
    "RobotConfig",
    "TaskConfig",
    "TennisRacketTrackingConfig",
    "ViserConfig",
    "XsensMorphologyConfig",
    "XsensViserConfig",
]
