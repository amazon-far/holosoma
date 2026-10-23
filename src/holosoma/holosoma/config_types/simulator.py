from __future__ import annotations

from dataclasses import field
from enum import Enum
from typing import Any, Literal

from pydantic import field_validator, model_validator
from pydantic.dataclasses import dataclass

from holosoma.config_types.frequency import DecimationLike, resolve_decimation, validate_decimation_like
from holosoma.config_types.viewer import ViewerConfig


class MujocoBackend(str, Enum):
    """MuJoCo physics backend selection.

    Determines which MuJoCo backend to use for physics simulation.
    """

    CLASSIC = "classic"
    """CPU-based single environment backend."""

    WARP = "warp"
    """GPU-accelerated multi-environment backend."""


class IsaacPhysicsBackend(str, Enum):
    """Isaac Sim physics engine selection.

    Isaac Lab 2.3.2 (Isaac Sim 5.1) had exactly one engine, so this is meaningless there and the
    5.1 backend rejects any non-default value rather than ignoring it. Isaac Lab 3.0 (Isaac Sim
    6.1) resolves the engine from ``SimulationCfg.physics``, which is what this selects.

    ``PHYSX`` and ``NEWTON`` are supported. PhysX stays the default because it is the engine the
    5.1 backend uses; Newton is a separate physics target. ``OVPHYSX`` is rejected at startup
    because it cannot run under Kit. Unsupported capabilities fail loudly -- see
    ``simulator/isaacsim6/capabilities.py`` and ``docs/isaac-sim-backends.md``.
    """

    PHYSX = "physx"
    """PhysX via ``isaaclab_physx``. The default, and the engine 5.1 uses."""

    NEWTON = "newton"
    """NVIDIA Newton via ``isaaclab_newton``. Warp-native; different solver and actuator model."""

    OVPHYSX = "ovphysx"
    """OmniPhysX via ``isaaclab_ov``."""


@dataclass(frozen=True)
class IsaacNewtonConfig:
    """Tunables for the Isaac Lab 3.0 Newton physics backend.

    Deliberately *not* a reuse of :class:`PhysxConfig`. The two engines share no tuning surface:
    ``isaaclab_newton.physics.NewtonCfg`` has none of PhysX's ``solver_type`` /
    ``num_position_iterations`` / ``num_velocity_iterations`` / ``bounce_threshold_velocity``, and
    exposes substepping plus solver sub-configurations instead. Mapping one struct onto both would
    mean silently dropping whichever fields the selected engine does not understand, so each
    engine gets its own.

    Only used when ``isaac_physics_backend='newton'``. Fields left ``None`` keep Isaac Lab's own
    default rather than holosoma imposing one.
    """

    num_substeps: int | None = None
    """Physics substeps per simulation step. Isaac Lab default is 1."""

    collision_decimation: int | None = None
    """How many steps between collision detection passes. Isaac Lab default is 0 (every step)."""

    use_cuda_graph: bool | None = None
    """Capture the step as a CUDA graph. Isaac Lab default is True; disabling aids debugging."""

    deterministic_mode: str | None = None
    """Newton's determinism level, e.g. ``'not_guaranteed'`` (its default)."""


@dataclass(frozen=True)
class MujocoWarpConfig:
    """Configuration for MuJoCo Warp backend memory allocation.

    Controls GPU memory allocation for batched parallel simulation.
    Increase these values if you encounter overflow warnings during training.
    """

    nconmax_per_env: int = 96
    """Maximum contacts per environment (default: 96).

    Increase for:
    - Complex terrains with many contact points
    - Robots with numerous collision geometries
    - Multi-contact scenarios (manipulation, climbing)

    Memory scales as: num_envs x nconmax_per_env
    """

    njmax_per_env: int | None = None
    """Maximum constraints per environment (default: auto-calculated).

    If None (default), automatically calculated as: max(nconmax * 6, nv * 4)
    where nv is the model's velocity dimension.

    Constraints include:
    - Contact constraints (friction cones: ~6 per contact)
    - Joint limits
    - Equality constraints

    Only override if you know you need more constraint capacity.
    """


@dataclass(frozen=True)
class ResetManagerConfig:
    """Configuration for the reset event manager."""

    events: list[Any] = field(default_factory=list)
    """List of reset event configurations to be managed."""


@dataclass(frozen=True)
class PhysxConfig:
    """Low-level PhysX solver settings."""

    solver_type: int
    """Solver type identifier passed to PhysX."""

    num_position_iterations: int
    """Number of position iterations per solver step."""

    num_velocity_iterations: int
    """Number of velocity iterations per solver step."""

    num_threads: int = 4
    """Worker thread count used by PhysX."""

    enable_dof_force_sensors: bool = False
    """Whether to enable force sensors on individual DOFs."""

    bounce_threshold_velocity: float = 0.5
    """Velocity threshold below which bounce responses are suppressed."""


@dataclass(frozen=True)
class MujocoXMLFilterCfg:
    """Configuration for filtering MuJoCo MJCF/XML robot files.

    This configuration controls how robot MJCF files are processed and filtered
    when loaded into the MuJoCo simulator. It allows removal of specific elements
    that may conflict with the simulation environment or cause issues.
    """

    enable: bool = False
    """Whether to enable XML filtering."""

    remove_lights: bool = True
    """Whether to remove <light> elements from the MJCF file."""

    remove_ground: bool = True
    """Whether to remove ground/floor/plane geometries from the MJCF file.
    Assumes these are top-level worldbody geoms."""

    ground_names: list[str] = field(default_factory=lambda: ["floor", "ground", "plane"])
    """List of geometry names to identify and remove as ground elements."""


@dataclass(frozen=True)
class IsaacSimRenderConfig:
    """Isaac Sim / Kit AppLauncher options applied at app startup."""

    rendering_mode: Literal["performance", "balanced", "quality"] | None = None
    """AppLauncher rendering-mode preset (e.g. ``quality`` or ``performance``)."""

    kit_args: str = ""
    """Space-separated Kit/carb settings forwarded to AppLauncher — the way to set RTX/render options,
    e.g. ``--/rtx/rendermode=PathTracing --/rtx/pathtracing/spp=32``."""

    def app_launcher_args(self) -> dict[str, Any]:
        """The audited subset of settings accepted by Isaac Lab's AppLauncher."""
        launcher_args: dict[str, Any] = {}
        if self.rendering_mode is not None:
            launcher_args["rendering_mode"] = self.rendering_mode
        if self.kit_args:
            launcher_args["kit_args"] = self.kit_args
        return launcher_args


@dataclass(frozen=True)
class SimEngineConfig:
    """Top-level simulation engine settings."""

    fps: int
    """Physics rate in Hz: the sim integrates one physics step every ``1/fps`` seconds (the engine
    timestep ``dt``). The base frequency every other rate here is derived from."""

    control_decimation: DecimationLike
    """Physics steps per control step: sets the CONTROL (policy) rate to ``fps / control_decimation``
    Hz. Each control step the policy emits one action that is HELD while the sim runs this many
    physics steps, then the next observation/action is taken. Higher = cheaper, lower-frequency
    control. Int (every Nth physics step), or a frequency string expressed against ``fps``: a bare
    "50Hz" requires an exact decimation (e.g. fps=200 -> 4) and errors if not exactly achievable,
    while ">50Hz"/"<50Hz" round to the nearest rate at or above/below the target. Holds the value AS
    WRITTEN; read the resolved integer step count via :attr:`control_decimation_steps`."""

    substeps: int
    """Solver substeps WITHIN each physics step — subdivides the ``1/fps`` step for finer/stabler
    integration (contacts, stiff dynamics). Does NOT change ``fps`` or any rate; purely solver
    accuracy vs. cost per step."""

    physx: PhysxConfig
    """PhysX solver configuration."""

    render_mode: str = "human"
    """Rendering mode requested from the simulator."""

    render_interval: DecimationLike = 1
    """Physics steps per rendered frame: render rate = ``fps / render_interval`` Hz.
    Decouples rendering from physics so the sim can step faster than it draws; 1 = render every
    step. Int, or a frequency string vs ``fps``. Holds the value AS WRITTEN; read the resolved
    integer step count via :attr:`render_interval_steps`."""

    max_episode_length_s: float = 20.0
    """Maximum episode length in seconds."""

    kinematic_playback: bool = False
    """Disable physics: each step propagates externally written state through forward
    kinematics instead of integrating. No gravity, contacts, or actuation, for the WHOLE sim.
    For rendering-only replay of recorded trajectories. Mixing (kinematic robot, simulated
    objects) is not supported; that would be a per-actor flag (e.g. IsaacLab's per-prim
    ``kinematic_enabled``), not this global one."""

    @model_validator(mode="after")
    def _validate_rates(self) -> SimEngineConfig:
        """Validate (do NOT mutate) the rate fields: both must resolve to a valid step count against
        ``fps``. The fields keep what the user wrote; the resolved ints are exposed as the
        ``*_steps`` properties below, computed against ``fps`` on access (always consistent if ``fps``
        changes, and the input config is never rewritten)."""
        resolve_decimation(self.control_decimation, self.fps, field="control_decimation", log=True)
        resolve_decimation(self.render_interval, self.fps, field="render_interval", log=True)
        return self

    @property
    def control_decimation_steps(self) -> int:
        """Resolved physics-steps-per-control-step (int >= 1) — ``control_decimation`` evaluated
        against ``fps`` (a frequency string becomes ``fps/target``; an int passes through)."""
        return resolve_decimation(self.control_decimation, self.fps, field="control_decimation")

    @property
    def render_interval_steps(self) -> int:
        """Resolved physics-steps-per-rendered-frame (int >= 1) — ``render_interval`` evaluated
        against ``fps`` (frequency string -> ``fps/target``; int passes through)."""
        return resolve_decimation(self.render_interval, self.fps, field="render_interval")


@dataclass(frozen=True)
class VirtualGantryCfg:
    """Configuration parameters for virtual gantry system."""

    enabled: bool = False
    """Whether to enable the virtual gantry system."""

    attachment_body_names: list[str] = field(
        default_factory=lambda: ["Trunk", "torso_link", "torso", "base_link", "pelvis", "base"]
    )
    """List of body names to try for attachment (in preference order)."""

    stiffness: float = 200.0
    """Spring stiffness coefficient for elastic band force calculation."""

    damping: float = 100.0
    """Damping coefficient for velocity-based force damping."""

    height: float = 3.0
    """Default height for gantry anchor point in world coordinates."""

    point: list[float] | None = None
    """3D position of gantry anchor point [x, y, z]. If None, defaults to [0, 0, height]."""

    length: float = 0.0
    """Rest length of the elastic band (zero force distance)."""

    apply_force: float = 0.0
    """Additional force magnitude to apply (for manual force adjustment)."""

    apply_force_sign: int = -1
    """Sign multiplier for apply_force direction (-1 or 1)."""


@dataclass(frozen=True)
class BridgeConfig:
    """Configuration for robot SDK bridge integration.

    This configuration matches the parameters used in holosoma_inference's BaseSimulator
    for robot SDK communication and control.
    """

    enabled: bool = False
    """Whether to enable the bridge."""

    # Core bridge settings (from holosoma_inference BaseSimulator)
    use_joystick: bool = False
    """Whether to enable joystick/wireless controller support."""

    joystick_device: int = 0
    """Joystick device ID (Linux only)."""

    joystick_type: str = "xbox"
    """Type of joystick controller."""

    # SDK connection settings
    domain_id: int = 0
    """Domain ID for robot communication."""

    interface: str | None = None
    """Network interface for robot communication. Auto-detected if None."""

    transport_decimation: DecimationLike = 1
    """SDK transport cadence. ``1`` exchanges data every physics step; an integer exchanges it every
    Nth step. Frequency strings such as ``"50Hz"`` are resolved against the physics rate. This
    setting does not change the torque-control cadence."""

    # ROS settings
    use_ros: bool = False
    """Whether to use ROS for communication."""

    publish_odom: bool = True
    """Publish base odometry over the SDK (SportModeState on rt/odommodestate) each transport update.

    On by default so the sim feeds base odometry through the Unitree SDK bridge
    (so a downstream telemetry read_odom_state -> /telemetry/odom is identical to hardware). Only
    SDKs with a base-state channel act on it; others (booster) treat publish_odom as a no-op."""

    dds_config: str | None = None
    """Complete inline CycloneDDS XML for unitree_mp on SDK domain 0 only.

    Requires DDS_CONFIG_API_VERSION=1, available in far-unitree-sdk 0.1.8. XML overrides the interface;
    detailed XML validation belongs to the native SDK. None preserves NIC-only initialization.
    """

    @field_validator("dds_config")
    @classmethod
    def validate_dds_config(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("dds_config must be a non-empty inline XML string or None")
        return value

    @model_validator(mode="after")
    def validate_bridge(self) -> BridgeConfig:
        """Validate transport cadence form; the hook registry resolves it against the physics rate."""
        validate_decimation_like(self.transport_decimation, field="BridgeConfig.transport_decimation")
        return self


@dataclass(frozen=True)
class SimulatorInitConfig:
    """Top-level simulator initialisation configuration."""

    name: str
    """Name of the simulator backend (e.g. ``isaacsim``)."""

    sim: SimEngineConfig
    """Simulation engine configuration settings."""

    debug_viz: bool = True
    """Enable debug visualization (gantry lines, etc.)."""

    viewer: ViewerConfig = field(default_factory=ViewerConfig)
    """Interactive viewer camera configuration.

    Configures camera tracking for the interactive viewer with advanced features:
    - Multiple camera modes (Fixed, Spherical, Cartesian)
    - Camera smoothing for stable viewing
    - Robot tracking with configurable body attachment

    Example:
        viewer=ViewerConfig(
            enabled=True,
            camera=SphericalCameraConfig(
                distance=3.0,
                azimuth=45.0,
                elevation=30.0,
            ),
        )
    """

    reset_manager: ResetManagerConfig = field(default_factory=ResetManagerConfig)
    """Reset event manager configuration."""

    contact_sensor_history_length: int = 3
    """Number of frames of contact data retained for sensors."""

    robot_mjcf_filter: MujocoXMLFilterCfg = field(default_factory=MujocoXMLFilterCfg)
    """MuJoCo-specific XML filtering configuration for robot MJCF files."""

    isaacsim: IsaacSimRenderConfig = field(default_factory=IsaacSimRenderConfig)
    """Isaac Sim AppLauncher options (rendering mode + Kit/carb args)."""

    mujoco_backend: MujocoBackend = MujocoBackend.CLASSIC
    """MuJoCo physics backend selection.

    Determines which MuJoCo backend to use for physics simulation:
    - 'classic': CPU-based single environment (backward compatible, default)
    - 'warp': GPU-accelerated multi-environment with mujoco_warp

    This setting only applies when using the MuJoCo simulator (name='mujoco').
    For other simulators (isaacsim), this field is ignored.

    Command line usage:
        --simulator.config.mujoco-backend=warp
        --simulator.config.mujoco-backend=classic

    Or use the syntactic sugar configs:
        simulator:mujoco   (uses classic backend)
        simulator:mjwarp   (uses warp backend)
    """

    isaac_physics_backend: IsaacPhysicsBackend = IsaacPhysicsBackend.PHYSX
    """Isaac Sim physics engine selection.

    Only meaningful for the Isaac Sim 6.2 backend (``simulator:isaacsim6``), which resolves the
    engine from ``SimulationCfg.physics``. Isaac Sim 5.1 has one engine, so the 5.1 backend
    **rejects** any non-default value instead of ignoring it -- a silently ignored engine choice
    would be a config that lies about what ran.

    ``physx`` matches 5.1's numerics; ``newton`` is a separate supported physics target.
    ``ovphysx`` and unported capabilities raise at startup. See
    ``docs/isaac-sim-backends.md`` for the per-backend matrix.

    Command line usage:
        --simulator.config.isaac-physics-backend=newton
    """

    isaac_newton: IsaacNewtonConfig = field(default_factory=IsaacNewtonConfig)
    """Newton backend tunables. Only used when ``isaac_physics_backend='newton'``.

    Separate from ``physx`` because the engines share no tuning fields (see
    :class:`IsaacNewtonConfig`).
    """

    isaac_visualizers: tuple[str, ...] = ()
    """Isaac Lab 3.0 visualizers: kit, newton_gl, newton_rtx, rerun, or viser.

    An empty tuple follows training.headless: no viewer when headless, Kit otherwise.
    Camera rendering is configured independently and works without a viewer.
    Isaac Sim 5.1 rejects a non-empty value because it has a different launcher API.
    """

    mujoco_warp: MujocoWarpConfig = field(default_factory=MujocoWarpConfig)
    """MuJoCo Warp backend memory allocation configuration.

    Controls GPU memory allocation for the Warp backend. Only used when
    mujoco_backend='warp'. Allows tuning contact and constraint capacity
    for different scenarios.

    Command line usage:
        --simulator.config.mujoco-warp.nconmax-per-env=128
        --simulator.config.mujoco-warp.njmax-per-env=1024
    """

    bridge: BridgeConfig = field(default_factory=BridgeConfig)
    """Robot SDK bridge configuration."""

    virtual_gantry: VirtualGantryCfg = field(default_factory=VirtualGantryCfg)
    """Virtual gantry system configuration."""


@dataclass(frozen=True)
class SimulatorConfig:
    """Wrapper for simulator instantiation."""

    _target_: str
    """Fully-qualified simulator factory target."""

    _recursive_: bool
    """Recursive instantiation flag."""

    config: SimulatorInitConfig
    """Structured simulator configuration passed to the factory."""
