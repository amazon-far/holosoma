"""Configuration types for data conversion."""

from __future__ import annotations

from dataclasses import dataclass, field

from holosoma_retargeting.config_types.data_type import MotionDataConfig
from holosoma_retargeting.config_types.robot import RobotConfig

_ROBOT_JOINT_NAMES_DEFAULT = {
    "g1": [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ],
    "t1": [
        "AAHead_yaw",
        "Head_pitch",
        "Left_Shoulder_Pitch",
        "Left_Shoulder_Roll",
        "Left_Elbow_Pitch",
        "Left_Elbow_Yaw",
        "Right_Shoulder_Pitch",
        "Right_Shoulder_Roll",
        "Right_Elbow_Pitch",
        "Right_Elbow_Yaw",
        "Waist",
        "Left_Hip_Pitch",
        "Left_Hip_Roll",
        "Left_Hip_Yaw",
        "Left_Knee_Pitch",
        "Left_Ankle_Pitch",
        "Left_Ankle_Roll",
        "Right_Hip_Pitch",
        "Right_Hip_Roll",
        "Right_Hip_Yaw",
        "Right_Knee_Pitch",
        "Right_Ankle_Pitch",
        "Right_Ankle_Roll",
    ],
}


@dataclass(frozen=True)
class DataConversionConfig:
    """Configuration for data conversion.

    This follows the pattern from holosoma's config_types.
    Uses a flat structure with all conversion parameters.
    """

    input_file: str
    """Path to input motion file."""

    robot: str = "g1"
    """Robot model to use. Use str to allow dynamic robot types."""

    data_format: str = "smplh"
    """Motion data format. Use str to allow dynamic data formats."""

    object_name: str | None = None
    """Override object name (default depends on robot and data type)."""

    input_fps: int = 30
    """FPS of the input motion."""

    output_fps: int = 50
    """FPS of the output motion."""

    line_range: tuple[int, int] | None = None
    """Line range (start, end) for loading data (both inclusive)."""

    has_dynamic_object: bool = False
    """Whether the motion has a dynamic object."""

    output_name: str | None = None
    """Name of the output motion npz file."""

    once: bool = False
    """Run the motion once and exit."""

    use_omniretarget_data: bool = False
    """Use OmniRetarget data format."""

    # --- Nested configs ---
    robot_config: RobotConfig = field(default_factory=lambda: RobotConfig(robot_type="g1"))
    """Robot configuration (nested - can override robot_urdf_file, robot_dof, etc.
    via --robot-config.robot-urdf-file)."""

    motion_data_config: MotionDataConfig = field(
        default_factory=lambda: MotionDataConfig(data_format="smplh", robot_type="g1")
    )
    """Motion data configuration (nested - can override data_format, robot_type, etc.
    via --motion-data-config.data-format)."""

    # --- Joint names, follow the pattern in holosoma_retargeting/config_types/robot.py ---
    joint_names: list[str] | None = None
    """Joint names to use."""

    def _joint_names(self) -> list[str]:
        """Get joint names - use override if provided, else use default."""
        if self.joint_names is not None:
            return self.joint_names
        if self.robot not in _ROBOT_JOINT_NAMES_DEFAULT:
            raise ValueError(f"No joint names found for robot: {self.robot}")
        return _ROBOT_JOINT_NAMES_DEFAULT[self.robot]

    JOINT_NAMES = property(
        _joint_names,
        doc="Get joint names - use override if provided, else use default.",
    )
