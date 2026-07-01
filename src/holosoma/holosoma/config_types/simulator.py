from __future__ import annotations

from dataclasses import field
from enum import Enum
from typing import Any

from pydantic import model_validator
from pydantic.dataclasses import dataclass

from holosoma.config_types.frequency import DecimationLike, resolve_decimation
from holosoma.config_types.viewer import ViewerConfig


class MujocoBackend(str, Enum):
    """MuJoCo physics backend selection.

    Determines which MuJoCo backend to use for physics simulation.
    """

    CLASSIC = "classic"
    """CPU-based single environment backend."""

    WARP = "warp"
    """GPU-accelerated multi-environment backend."""


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

    # Rate limiting
    rate_limit_dt: float | None = None
    """Rate limiting timestep. If None, uses simulation timestep."""

    # ROS settings
    use_ros: bool = False
    """Whether to use ROS for communication."""


@dataclass(frozen=True)
class SimulatorInitConfig:
    """Top-level simulator initialisation configuration."""

    name: str
    """Name of the simulator backend (e.g. ``isaacgym``)."""

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

    mujoco_backend: MujocoBackend = MujocoBackend.CLASSIC
    """MuJoCo physics backend selection.

    Determines which MuJoCo backend to use for physics simulation:
    - 'classic': CPU-based single environment (backward compatible, default)
    - 'warp': GPU-accelerated multi-environment with mujoco_warp

    This setting only applies when using the MuJoCo simulator (name='mujoco').
    For other simulators (isaacgym, isaacsim), this field is ignored.

    Command line usage:
        --simulator.config.mujoco-backend=warp
        --simulator.config.mujoco-backend=classic

    Or use the syntactic sugar configs:
        simulator:mujoco   (uses classic backend)
        simulator:mjwarp   (uses warp backend)
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
