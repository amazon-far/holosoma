"""Task configuration types for holosoma_inference."""

from __future__ import annotations

from typing import Literal

from pydantic.dataclasses import dataclass

InputSource = Literal["keyboard", "interface", "joystick", "ros2", "injected"]

DEFAULT_VELOCITY_INPUT: InputSource = "keyboard"
DEFAULT_STATE_INPUT: InputSource = "keyboard"


@dataclass(frozen=True)
class DebugConfig:
    """Debug overrides for quick testing."""

    force_upright_imu: bool = False
    """Override projected_gravity with [0, 0, -1] (perfectly upright)."""

    force_zero_angular_velocity: bool = False
    """Override base_ang_vel with [0, 0, 0]."""

    force_zero_action: bool = False
    """Zero out the scaled policy action (robot holds default pose)."""

    record_rosbag: bool = False
    """Record a per-session rosbag for the lifetime of this run_policy process.
    Spawns ``ros2 bag record --all`` in the background when the policy starts and
    stops it on exit (Ctrl-C, normal completion, or error), then prints the bag's
    absolute path. Each run gets its own short, self-contained bag — no hunting
    for a timestamp inside a long fleet-telemetry recording."""

    rosbag_dir: str = "~/run_policy_sessions"
    """Directory for per-session rosbags recorded via ``record_rosbag``."""

    rosbag_cpu: int = 0
    """CPU core to pin ``ros2 bag record`` to (via ``taskset``) so it doesn't
    inherit the policy's affinity and add control-loop jitter. Default 0 exists
    everywhere and is never an isolated RT core. Negative disables pinning."""


@dataclass(frozen=True)
class OnnxRuntimeConfig:
    """ONNX Runtime session options for the policy model. Defaults are ORT's own."""

    intra_op_num_threads: int = 0
    """Threads per operator. 0 keeps ORT's default of one thread per core, which
    on a core-restricted target oversubscribes badly: 32 threads on three
    available cores measured 1.26 s of run-queue delay against 4.66 s of CPU."""

    inter_op_num_threads: int = 0
    """Threads across operators (parallel execution mode only). 0 keeps ORT's default."""

    enable_cpu_mem_arena: bool = True
    """ORT default. False makes the CPU allocator go to the system allocator instead."""

    enable_mem_pattern: bool = True
    """ORT default. False disables pre-planned memory reuse across nodes."""

    execution_mode: Literal["sequential", "parallel"] = "sequential"
    """Graph execution mode. "sequential" is ORT's default."""


@dataclass(frozen=True)
class Ros2DepthConsumerConfig:
    """Config for the ROS2 depth-image consumer (``Ros2DepthConsumer``).

    Defaults match the on-robot image_server preprocessing the policy was
    trained against. Leave ``topics`` empty to disable the depth sensor.
    """

    topics: tuple[str, ...] = ()
    """Raw depth topic(s) (``sensor_msgs/Image``, encoding ``32FC1``, metric
    meters). Empty disables the sensor. One per camera, in stack order (front
    first, back second). Multi-camera frames are time-synchronized; the policy
    reads a preprocessed ``(N, 1, resized_height, resized_width)`` stack via
    ``self._injected_sensors["depth"].get_latest()``."""

    resized_height: int = 27
    """Target height (bicubic resize) before clip+normalize."""

    resized_width: int = 48
    """Target width (bicubic resize) before clip+normalize."""

    near_clip: float = 0.1
    """Near clip (m); depth is normalized to [-0.5, 0.5] over [near, far]."""

    far_clip: float = 2.0
    """Far clip (m); depth is normalized to [-0.5, 0.5] over [near, far]."""

    frame_delay_ms: float = 0.0
    """Modeled depth latency to re-introduce, in absolute milliseconds.

    The ROS2 depth transport is effectively instantaneous (sub-1ms), but the
    policy was trained with the on-robot image_server's inherent capture/serve
    latency baked in. ``get_latest()`` holds back frames so it returns the
    freshest frame at least this old, reproducing that delay independent of
    publish rate (a fixed ms value is robust across fps, unlike a frame count).
    ``0.0`` (default) keeps freshest-frame behavior. Pin this per-policy from a
    preset to match the delay the model was trained with (e.g. ``200.0``)."""


@dataclass(frozen=True)
class TaskConfig:
    """Task execution configuration for policy inference."""

    model_path: str | list[str]
    """Path to ONNX model(s). Supports local paths and wandb:// URIs. Required field."""

    rl_rate: float = 50
    """Policy inference rate in Hz."""

    policy_action_scale: float = 0.25
    """Scaling factor applied to policy actions."""

    action_scales_by_effort_limit_over_p_gain: bool = False
    """Use per-joint scaling: ``policy_action_scale * effort_limit / p_gain``."""

    use_phase: bool = True
    """Whether to use gait phase observations."""

    gait_period: float = 1.0
    """Gait cycle period in seconds."""

    skip_stiff_prompt: bool = False
    """WBT: skip the blocking stdin prompt before entering stiff hold and enter
    immediately. Default False keeps the interactive 'Press Enter to continue'
    safety pause; non-interactive launches (e.g. the ROS2 service node) set True."""

    domain_id: int = 0
    """DDS domain ID for communication."""

    interface: str = "auto"
    """Network interface name. Use ``"auto"`` to auto-detect, or specify explicitly (e.g. ``"eth0"``)."""

    depth: Ros2DepthConsumerConfig = Ros2DepthConsumerConfig()
    """Depth-image consumer config (empty ``topics`` disables it)."""

    onnxruntime: OnnxRuntimeConfig = OnnxRuntimeConfig()
    """ONNX Runtime session options for the policy model."""

    fresh_state_timeout_s: float = 0.0
    """Block for a fresh state sample before reading state, up to this many seconds.

    The control loop is paced by ``RateLimiter`` on its own absolute schedule and
    then reads whatever the interface has cached, so the consumed sample can be
    older than the sensor period. When this is non-zero and the interface
    implements ``wait_fresh_state(timeout_s)``, the loop waits for the next
    sample first; the wait is wall time inside the cycle, not sensor→wire
    latency. 0 (default) reads the cache as before."""

    stats_log_interval: int = 50
    """Iterations between the one-line latency summary. 0 disables the summary."""

    latency_csv_path: str | None = None
    """Write a per-cycle latency CSV here. The logged stats are a rolling
    ``rl_rate``-wide window, which cannot show a p99 or a max over a whole run."""

    velocity_input: InputSource = DEFAULT_VELOCITY_INPUT
    """Source for velocity commands."""

    state_input: InputSource = DEFAULT_STATE_INPUT
    """Source for non-velocity inputs (start/stop, walk/stand, tuning)."""

    use_keyboard: bool = False
    """Shortcut: set both velocity_input and state_input to "keyboard".

    Cannot be combined with explicit input settings.
    """

    use_joystick: bool = False
    """Shortcut: set both velocity_input and state_input to "interface".

    Reads from the SDK's wireless controller (the dongle/controller shipped
    with Unitree G1, Booster T1, etc.). For host-side USB gamepads
    (Xbox/Logitech via /dev/input/event*), use ``use_usb_joystick`` instead.

    Cannot be combined with explicit input settings.
    """

    use_usb_joystick: bool = False
    """Shortcut: set both velocity_input and state_input to "joystick".

    Reads a USB gamepad on the host via evdev (``/dev/input/event*``).
    Linux-only. Cannot be combined with explicit input settings.
    """

    joystick_type: str = "xbox"
    """Joystick type."""

    joystick_device: int = 0
    """Joystick device index."""

    ros_cmd_vel_topic: str = "cmd_vel"
    """ROS2 topic name for velocity commands (used when velocity_input is "ros2")."""

    ros_state_input_topic: str = "holosoma/state_input"
    """ROS2 topic name for discrete commands (used when state_input is "ros2")."""

    ros_vel_timeout: float = 1.0
    """Seconds without a velocity message before zeroing commands. Set to 0 to disable."""

    auto_walk_on_vel_cmd: bool = False
    """Automatically enter walking mode when a non-zero velocity command is received."""

    use_sim_time: bool = False
    """Use synchronized simulation time for WBT policies."""

    # Deprecation candidates:
    desired_base_height: float = 0.75
    """Target base height in meters."""

    residual_upper_body_action: bool = False
    """Whether to use residual control for upper body."""

    print_observations: bool = False
    """Print observation vectors for debugging."""

    motion_start_timestep: int = 0
    """Starting timestep for motion clip playback."""

    motion_end_timestep: int | None = None
    """Ending timestep for motion clip playback. If None, plays until the end."""

    debug: DebugConfig = DebugConfig()
    """Debug overrides for quick testing."""

    def __post_init__(self):
        """Resolve use_keyboard/use_joystick/use_usb_joystick shortcuts into velocity_input/state_input."""
        active_shortcuts = [
            name
            for name, enabled in (
                ("keyboard", self.use_keyboard),
                ("joystick", self.use_joystick),
                ("usb-joystick", self.use_usb_joystick),
            )
            if enabled
        ]
        if len(active_shortcuts) > 1:
            joined = ", ".join(f"--task.use-{n}" for n in active_shortcuts)
            raise ValueError(
                f"Cannot combine multiple input shortcuts ({joined}). "
                "Use one shortcut or set --task.velocity-input and --task.state-input individually."
            )

        shortcut: InputSource | None = None
        flag_name: str | None = None
        if self.use_usb_joystick:
            shortcut = "joystick"
            flag_name = "usb-joystick"
        elif self.use_joystick:
            shortcut = "interface"
            flag_name = "joystick"
        elif self.use_keyboard:
            shortcut = "keyboard"
            flag_name = "keyboard"

        if shortcut is not None:
            has_custom_input = self.velocity_input != DEFAULT_VELOCITY_INPUT or self.state_input != DEFAULT_STATE_INPUT
            if has_custom_input:
                raise ValueError(
                    f"Cannot combine --task.use-{flag_name} with --task.velocity-input or "
                    "--task.state-input. Use either the shortcut flag or the individual "
                    "input settings, not both."
                )
            object.__setattr__(self, "velocity_input", shortcut)
            object.__setattr__(self, "state_input", shortcut)
