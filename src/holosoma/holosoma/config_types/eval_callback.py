"""Config types for eval callbacks."""

from __future__ import annotations

import dataclasses

from pydantic import model_validator
from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class RecordingConfig:
    """Settings for trajectory recording during evaluation."""

    enabled: bool = False
    """Whether to enable trajectory recording."""

    output_path: str = "eval_recording.npz"
    """Path to save NPZ recording."""

    record_motion_ref: bool = False
    """Record motion reference data from MotionCommand (time_steps, root, joints)."""

    note: str = ""
    """Free-form note stored in NPZ metadata (e.g. eval conditions)."""


@dataclass(frozen=True)
class RecordingCallbackConfig:
    """Instantiation config for EvalRecordingCallback."""

    _target_: str = "holosoma.agents.callbacks.recording.EvalRecordingCallback"
    """Class to instantiate."""

    config: RecordingConfig = RecordingConfig()
    """Recording settings."""


@dataclass(frozen=True)
class GridEvalVelocityConfig:
    """Settings for velocity sweep in grid evaluation.

    Each non-trivial axis is swept independently (other axes default to 0).
    E.g. lin_vel_x=[0.25,0.5], lin_vel_y=[0.25,0.5] -> 4 conditions.
    """

    enabled: bool = False
    """Whether to enable grid velocity sweep."""

    lin_vel_x: tuple[float, ...] = (0.0,)
    """Linear velocity X values to sweep (m/s)."""

    lin_vel_y: tuple[float, ...] = (0.0,)
    """Linear velocity Y values to sweep (m/s)."""

    ang_vel_yaw: tuple[float, ...] = (0.0,)
    """Angular velocity yaw values to sweep (rad/s)."""

    warmup_s: float = 1.0
    """Duration of warmup period in seconds. During warmup all velocity commands are
    zero. After warmup, grid velocities are applied. The warmup period is still
    recorded (analysis scripts skip it using the warmup_steps metadata field)."""


@dataclass(frozen=True)
class GridEvalVelocityCallbackConfig:
    """Instantiation config for GridEvalVelocityCallback."""

    _target_: str = "holosoma.agents.callbacks.grid_eval_velocity.GridEvalVelocityCallback"
    """Class to instantiate."""

    config: GridEvalVelocityConfig = GridEvalVelocityConfig()
    """Grid velocity sweep settings."""


@dataclass(frozen=True)
class GridEvalPayloadConfig:
    """Settings for payload sweep in grid evaluation.

    Sweeps payload across body-group x mass dimensions, cross-producted with
    the velocity conditions from GridEvalVelocityCallback.

    Each entry in ``body_groups`` is a comma-separated group of body names that
    receive the force together (force is split evenly across bodies in the group).
    ``body_labels`` provides human-readable names for each group.

    Example:
        body_groups=("pelvis", "left_wrist_yaw_link,right_wrist_yaw_link")
        body_labels=("torso", "wrists")
        mass_kg=(0.0, 1.0, 2.0)

    This creates 2 body_groups x 3 masses = 6 payload sub-conditions, each
    cross-producted with the velocity grid.
    """

    enabled: bool = False
    """Whether to enable grid payload sweep."""

    body_groups: tuple[str, ...] = ()
    """Body groups for payload forces. Each entry is a comma-separated list of
    body names that receive force together. Creates one sweep dimension."""

    body_labels: tuple[str, ...] = ()
    """Human-readable labels for each body group (must match length of body_groups)."""

    mass_kg: tuple[float, ...] = (0.0,)
    """Payload masses to sweep in kg (0 = no payload)."""

    @model_validator(mode="after")
    def _validate(self) -> GridEvalPayloadConfig:
        if not self.enabled:
            return self
        if not self.body_groups:
            raise ValueError("GridEvalPayloadConfig: body_groups must not be empty when enabled")
        if self.body_labels and len(self.body_labels) != len(self.body_groups):
            raise ValueError(
                f"GridEvalPayloadConfig: body_labels length ({len(self.body_labels)}) "
                f"must match body_groups length ({len(self.body_groups)})"
            )
        if not self.body_labels:
            object.__setattr__(self, "body_labels", tuple(bg.replace(",", "+") for bg in self.body_groups))
        return self


@dataclass(frozen=True)
class GridEvalPayloadCallbackConfig:
    """Instantiation config for GridEvalPayloadCallback."""

    _target_: str = "holosoma.agents.callbacks.grid_eval_payload.GridEvalPayloadCallback"
    """Class to instantiate."""

    config: GridEvalPayloadConfig = GridEvalPayloadConfig()
    """Grid payload sweep settings."""


@dataclass(frozen=True)
class GridEvalPushConfig:
    """Settings for push perturbation sweep in grid evaluation.

    Sweeps pushes across body x direction x gait_phase dimensions,
    cross-producted with the velocity conditions from GridEvalVelocityCallback.

    Each condition receives exactly one deterministic push, triggered when the
    target gait phase is first detected after a gait-analysis warmup period.

    **Directions** are in the robot's local frame (rotated by the robot's yaw
    at push time):
        forward  = robot's heading direction
        backward = opposite to heading
        left     = 90 deg left of heading
        right    = 90 deg right of heading

    **Gait phases** are detected from left-foot-height oscillations. After warmup,
    the callback records foot heights for at least ``min_gait_cycles`` full gait
    cycles (detected dynamically from zero-crossings), then monitors foot state
    to trigger pushes at target phases:
        swing_to_stance = left foot touchdown (transition from air to ground)
        stance_to_swing = left foot liftoff (transition from ground to air)
        mid_stance      = left foot on ground, right foot at peak height
        mid_swing       = left foot at peak height (in air)

    Foot bodies are specified via ``left_foot_body`` / ``right_foot_body``.

    Example:
        body_names=("torso_link", "pelvis")
        body_labels=("torso", "pelvis")
        directions=("forward", "backward", "left", "right")
        gait_phases=("swing_to_stance", "stance_to_swing", "mid_stance", "mid_swing")
        force_n=150.0
        duration_s=0.2

    This creates 2 bodies x 4 directions x 4 phases = 32 push sub-conditions,
    each cross-producted with the velocity grid.
    """

    enabled: bool = False
    """Whether to enable grid push sweep."""

    body_names: tuple[str, ...] = ()
    """Body names to push. Each entry is a single body name."""

    body_labels: tuple[str, ...] = ()
    """Human-readable labels for each body (must match length of body_names)."""

    directions: tuple[str, ...] = ("forward", "backward", "left", "right")
    """Push directions in robot-local frame: forward, backward, left, right."""

    gait_phases: tuple[str, ...] = ("swing_to_stance", "stance_to_swing", "mid_stance", "mid_swing")
    """Gait phases at which to trigger pushes. Based on left foot state:
    swing_to_stance = touchdown, stance_to_swing = liftoff,
    mid_stance = right foot at peak, mid_swing = left foot at peak."""

    min_gait_cycles: int = 3
    """Minimum number of full gait cycles to observe before finalizing gait analysis.
    Analysis continues beyond gait_analysis_s if needed to reach this count."""

    gait_analysis_s: float = 2.0
    """Minimum duration (seconds) after warmup for gait analysis. Actual analysis
    may run longer if min_gait_cycles haven't been observed yet."""

    max_gait_analysis_s: float = 8.0
    """Maximum duration (seconds) for gait analysis. If min_gait_cycles aren't
    observed within this window, analysis finalizes anyway with a warning."""

    left_foot_body: str = "left_foot_contact_point"
    """Body name for left foot height detection."""

    right_foot_body: str = "right_foot_contact_point"
    """Body name for right foot height detection."""

    foot_contact_threshold: float = 0.5
    """Fraction of foot height range below which the foot is considered 'on ground'.
    E.g. 0.5 means the foot is on the ground when its height is in the lower 50%
    of its observed swing range."""

    force_n: float = 150.0
    """Push force magnitude in Newtons (used when force_magnitudes is empty)."""

    force_magnitudes: tuple[float, ...] = ()
    """Force magnitudes to sweep (Newtons). When non-empty, overrides force_n
    and creates an additional sweep dimension. E.g. (50, 100, 200, 300)."""

    duration_s: float = 0.2
    """Push duration in seconds."""

    DIRECTION_VECTORS: dict[str, tuple[float, float, float]] = dataclasses.field(
        default_factory=lambda: {
            "forward": (1.0, 0.0, 0.0),
            "backward": (-1.0, 0.0, 0.0),
            "left": (0.0, 1.0, 0.0),
            "right": (0.0, -1.0, 0.0),
        },
        repr=False,
    )
    """Mapping from direction name to robot-local unit vector."""

    _VALID_GAIT_PHASES: tuple[str, ...] = (
        "swing_to_stance",
        "stance_to_swing",
        "mid_stance",
        "mid_swing",
    )

    @model_validator(mode="after")
    def _validate(self) -> GridEvalPushConfig:
        if not self.enabled:
            return self
        if not self.body_names:
            raise ValueError("GridEvalPushConfig: body_names must not be empty when enabled")
        if self.body_labels and len(self.body_labels) != len(self.body_names):
            raise ValueError(
                f"GridEvalPushConfig: body_labels length ({len(self.body_labels)}) "
                f"must match body_names length ({len(self.body_names)})"
            )
        if not self.body_labels:
            object.__setattr__(self, "body_labels", self.body_names)
        for d in self.directions:
            if d not in self.DIRECTION_VECTORS:
                raise ValueError(f"GridEvalPushConfig: unknown direction '{d}'. Valid: {list(self.DIRECTION_VECTORS)}")
        for p in self.gait_phases:
            if p not in self._VALID_GAIT_PHASES:
                raise ValueError(
                    f"GridEvalPushConfig: unknown gait phase '{p}'. Valid: {list(self._VALID_GAIT_PHASES)}"
                )
        return self


@dataclass(frozen=True)
class GridEvalPushCallbackConfig:
    """Instantiation config for GridEvalPushCallback."""

    _target_: str = "holosoma.agents.callbacks.grid_eval_push.GridEvalPushCallback"
    """Class to instantiate."""

    config: GridEvalPushConfig = GridEvalPushConfig()
    """Grid push sweep settings."""


@dataclass(frozen=True)
class EvalCallbacksConfig:
    """Container for all eval callback configs.

    To add a new callback, add a field here with its config type.
    Each field's value is passed to instantiate() if it has a _target_.
    """

    recording: RecordingCallbackConfig = RecordingCallbackConfig()
    """Trajectory recording callback."""

    grid_velocity: GridEvalVelocityCallbackConfig = GridEvalVelocityCallbackConfig()
    """Grid-based velocity sweep callback."""

    grid_payload: GridEvalPayloadCallbackConfig = GridEvalPayloadCallbackConfig()
    """Grid-based payload sweep callback."""

    grid_push: GridEvalPushCallbackConfig = GridEvalPushCallbackConfig()
    """Grid-based push sweep callback."""

    @model_validator(mode="after")
    def _validate(self) -> EvalCallbacksConfig:
        if self.grid_payload.config.enabled and self.grid_push.config.enabled:
            raise ValueError(
                "grid_payload and grid_push cannot both be enabled: "
                "both inject external forces via IsaacSim and would overwrite each other"
            )
        any_grid = (
            self.grid_velocity.config.enabled or self.grid_payload.config.enabled or self.grid_push.config.enabled
        )
        if any_grid and not self.recording.config.enabled:
            raise ValueError("Grid eval callbacks require recording to be enabled. Set --recording.config.enabled=True")
        return self

    def collect_active_callbacks(self) -> dict:
        """Collect callback configs where config.enabled is True."""
        cb_configs = {}
        for f in dataclasses.fields(self):
            cfg = getattr(self, f.name)
            if not hasattr(cfg, "_target_"):
                raise ValueError(f"Callback config '{f.name}' missing _target_ field")
            if not hasattr(cfg.config, "enabled"):
                raise ValueError(f"Callback config '{f.name}' missing config.enabled field")
            if cfg.config.enabled:
                cb_configs[f.name] = cfg
        return cb_configs
