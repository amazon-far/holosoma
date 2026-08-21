"""Configuration types for retargeter settings."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class FootLockConfig:
    """Configuration for explicit frame-range based foot locking constraints."""

    enable: bool = False
    """Whether to enforce explicit frame-range based foot locking constraints."""

    windows: dict[str, list[tuple[int, int] | tuple[int, int, float]]] | None = None
    """Per-foot inclusive frame windows for locking.
    Each window is (start, end) or (start, end, z_floor).
    If z_floor is given per-window, it overrides the global z_floor for that window.
    Example: {"L_Toe": [(30, 60, 0.15)], "R_Toe": [(10, 20), (80, 95, 0.30)]}"""

    z_floor: float = 0.0
    """Default floor height used by Z pinning constraints (overridden by per-window z)."""

    tolerance: float = 5e-3
    """Tolerance for Z floor pinning constraints."""


@dataclass(frozen=True)
class SelfCollisionConfig:
    """Configuration for self-collision avoidance constraints."""

    enable: bool = False
    """Whether to enforce self-collision constraints."""

    pairs: list[tuple[str, str]] = field(default_factory=list)
    """Body name pairs to check for self-collision.
    Example: [("left_elbow_link", "left_knee_link"), ("left_wrist_yaw_link", "left_knee_link")]"""

    windows: list[tuple[int, int]] | None = None
    """Inclusive frame windows during which self-collision is enforced.
    If None, enforced on all frames.
    Example: [(50, 120)] means only enforce on frames 50..120."""

    tolerance: float = 0.02
    """Minimum distance (meters) to maintain between body pairs."""


@dataclass(frozen=True)
class TennisRacketTrackingConfig:
    """Configure how the G1 right-hand orientation follows an Xsens racket."""

    mode: Literal["hand", "racket", "filtered"] = "hand"
    """Track the Xsens hand, always track the racket, or use feasibility-filtered racket tracking."""

    attachment_source: Literal["global", "embedded_tpose", "observed_window"] = "embedded_tpose"
    """Source used to correct the model-specific global G1 hand-to-racket grasp."""

    attachment_path: Path | None = None
    """Optional JSON override for the model-specific global G1 grasp."""

    observed_window_s: tuple[float, float] | None = None
    """Recording-relative [start, end) seconds used by ``observed_window`` calibration."""

    detach_exit_threshold_m: float = 0.10
    """Hand-relative racket-origin deviation that immediately disables racket tracking."""

    detach_reentry_threshold_m: float = 0.05
    """Lower origin-deviation threshold required while considering re-entry."""

    feasible_entry_error_rad: float = math.radians(45.0)
    """Maximum achieved symmetry-aware error for entering racket tracking."""

    feasible_exit_error_rad: float = math.radians(60.0)
    """Maximum achieved symmetry-aware error while racket tracking is already active."""

    min_wrist_limit_margin_rad: float = math.radians(5.0)
    """Minimum distance required from every right-wrist joint limit."""

    reentry_frames: int = 5
    """Consecutive feasible frames required before filtered mode re-enters racket tracking."""

    def __post_init__(self) -> None:
        if self.detach_reentry_threshold_m < 0.0:
            raise ValueError("detach_reentry_threshold_m must be non-negative")
        if self.detach_exit_threshold_m <= self.detach_reentry_threshold_m:
            raise ValueError("detach_exit_threshold_m must exceed detach_reentry_threshold_m")
        if self.feasible_entry_error_rad < 0.0:
            raise ValueError("feasible_entry_error_rad must be non-negative")
        if self.feasible_exit_error_rad < self.feasible_entry_error_rad:
            raise ValueError("feasible_exit_error_rad must be at least feasible_entry_error_rad")
        if self.min_wrist_limit_margin_rad < 0.0:
            raise ValueError("min_wrist_limit_margin_rad must be non-negative")
        if self.reentry_frames < 1:
            raise ValueError("reentry_frames must be positive")


@dataclass(frozen=True)
class OrientationTrackingConfig:
    """Configuration for additive Xsens orientation and segment-axis tracking."""

    enable: bool = False
    """Whether to enable orientation-aware retargeting costs."""

    calibration_path: Path | None = None
    """Path to the Xsens T-pose calibration artifact used for orientation/axis correspondences."""

    orientation_weight: float = 2.0
    """Weight for full segment orientation tracking residuals."""

    axis_weight: float = 5.0
    """Global weight for segment-axis direction tracking residuals."""

    orientation_error_clip_rad: float = 0.7
    """Maximum rotation-vector magnitude used by orientation residuals."""

    tennis_racket: TennisRacketTrackingConfig = field(default_factory=TennisRacketTrackingConfig)
    """Optional right-hand target selection driven by the tracked Xsens tennis racket."""


@dataclass(frozen=True)
class RetargeterConfig:
    """Configuration for retargeter parameters.

    These parameters control the retargeting optimization process.
    """

    q_a_init_idx: int = -7
    """Index in robot's configuration where optimization variables start.
    -7: starts from floating base, -3: starts from translation of floating base,
    0: starts from actuated DOF, 12: starts from waist, 15: starts from left shoulder"""

    activate_joint_limits: bool = True
    """Whether to enforce joint limits during retargeting."""

    activate_obj_non_penetration: bool = True
    """Whether to enforce ground/object non-penetration constraints."""

    activate_foot_sticking: bool = True
    """Whether to enforce foot sticking constraints."""

    penetration_tolerance: float = 0.001
    """Tolerance for penetration when enforcing non-penetration constraints."""

    foot_sticking_tolerance: float = 1e-3
    """Tolerance for foot sticking constraints in x, y."""

    foot_lock: FootLockConfig = field(default_factory=FootLockConfig)
    """Configuration for explicit frame-range based foot locking."""

    step_size: float = 0.2
    """Trust region for each SQP iteration."""

    initial_iterations: int = 50
    """SQP iterations used to solve the first motion frame."""

    iterations_per_frame: int = 10
    """SQP iterations used for each subsequent frame; raise for sparse keyframes."""

    visualize: bool = False
    """Whether to visualize the retargeting process."""

    debug: bool = False
    """Whether to enable debug mode."""

    self_collision: SelfCollisionConfig = field(default_factory=SelfCollisionConfig)
    """Configuration for self-collision avoidance."""

    orientation: OrientationTrackingConfig = field(default_factory=OrientationTrackingConfig)
    """Configuration for optional Xsens orientation and segment-axis tracking."""

    w_nominal_tracking_init: float = 5.0
    """Initial weight for nominal tracking cost."""

    nominal_tracking_tau: float = 1e6
    """Time constant for the nominal tracking cost."""
