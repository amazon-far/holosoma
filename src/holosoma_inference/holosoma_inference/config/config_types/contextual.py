"""Configuration for the native depth-conditioned policy, without Python inference."""

from __future__ import annotations

import math
from dataclasses import field

from pydantic.dataclasses import dataclass

G1_WAIST_GRAVITY_FRAME = (("waist_yaw_joint", "z"), ("waist_roll_joint", "x"), ("waist_pitch_joint", "y"))


@dataclass(frozen=True)
class ContextualConfig:
    command_dim: int = 15
    depth_height: int = 58
    depth_width: int = 87
    require_climb_head: bool = True
    force_manual_climbing: bool = False
    climb_enter: str = "dz_or_pred"
    climb_exit: str = "both_off"
    discretizer: str = "simple"
    yaw_request_gain: float = 2.0
    switch_margin: float = 0.0
    velocity_primitives: dict[str, list[float]] = field(default_factory=dict)
    forward_distance_scale: float = 0.5
    yaw_distance_scale: float = 1.2
    in_place_yaw_threshold: float | None = None
    gravity_frame_joints: tuple[tuple[str, str], ...] = G1_WAIST_GRAVITY_FRAME
    """Joints from the IMU body to the frame the model reads gravity in, as ``(name, axis)``.

    Gravity is first projected into the IMU body frame (from the orientation, or the interface's
    corrected gravity when it supplies one), then rotated through these joints in order. The
    default is the G1 waist chain, giving torso-frame gravity from the pelvis IMU. ``()`` keeps
    gravity in the IMU frame. Every listed joint must be one of the robot's joints.
    """
    max_depth_age_s: float = 0.25
    """Oldest depth frame, by local receive time, the runner will act on.

    An engaged policy whose newest frame is older than this is stopped, as if the operator had
    pressed stop, and a start is refused until frames arrive again.
    """

    def __post_init__(self):
        if not (math.isfinite(self.max_depth_age_s) and self.max_depth_age_s > 0):
            raise ValueError("max_depth_age_s must be finite and positive")
        for name, axis in self.gravity_frame_joints:
            if axis not in {"x", "y", "z"}:
                raise ValueError(f"gravity_frame_joints axis for {name} must be x, y or z")
