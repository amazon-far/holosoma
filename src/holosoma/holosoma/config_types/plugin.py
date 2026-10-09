"""Config types for simulator plugins.

A *plugin* is a bundle of behavior — a set of lifecycle hooks plus whatever state and
side effects they need — that an extension attaches to a running simulator without
subclassing a backend. Plugins are built on the lifecycle hook system (``simulator.hooks``,
:class:`~holosoma.simulator.base_simulator.hooks.Phase`) and may depend on other simulator
contracts (the virtual gantry, the clock, etc.). Each plugin pairs:

- a ``PluginConfig`` subclass (the CLI-visible, serializable knobs), registered under a
  name in ``holosoma.config_values.plugin.PLUGIN_REGISTRY``, and
- a runtime plugin class, returned by the config's :meth:`PluginConfig.get_cls`. There is
  no base class to inherit: any class constructed as ``cls(cfg, simulator)`` that
  registers its hooks on ``simulator.hooks`` works (duck-typed).

The config is resolved on the CLI as a dynamic-dict field (see ``RunSimConfig.plugin``);
``BaseSimulator.install_plugins`` instantiates each ``get_cls()`` after backend setup.
"""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from pydantic import ConfigDict, field_validator, model_validator
from pydantic.dataclasses import dataclass as pydantic_dataclass

from holosoma.config_types.frequency import DecimationLike, validate_decimation_like
from holosoma.config_types.value_types import UnitQuaternionWXYZ

# Reject unknown fields on Pydantic plugin configs.
_FORBID_EXTRA = ConfigDict(extra="forbid")


@dataclass(frozen=True)
class PluginConfig(abc.ABC):
    """Base config for a simulator plugin.

    Subclass with the plugin's parameters as dataclass fields and implement
    :meth:`get_cls` to point at the runtime plugin class. Register an instance in
    ``PLUGIN_REGISTRY`` so it is selectable as ``plugin.<key>:<variant>`` on the CLI.
    """

    @abc.abstractmethod
    def get_cls(self) -> Callable[..., Any]:
        """Return the runtime plugin class this config configures.

        The class is constructed as ``cls(cfg, simulator)`` and is expected to register
        its hooks on ``simulator.hooks`` in ``__init__`` — no base class required.
        Import it lazily inside this method so that registering the config preset does
        not pull the (possibly heavy) runtime module at import time.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class NoOpPluginConfig(PluginConfig):
    """A plugin that does nothing, registered as the ``none`` preset.

    Selecting ``plugin.<key>:none`` disables that slot: its runtime class registers no
    hooks, so it is a genuine no-op.
    """

    def get_cls(self) -> Callable[..., Any]:
        from holosoma.simulator.shared.builtin_plugins import NoOpPlugin

        return NoOpPlugin


@dataclass(frozen=True)
class ClockPublishPluginConfig(PluginConfig):
    """Publish sim time as a ROS2 ``rosgraph_msgs/msg/Clock`` topic.

    A ROS2 example plugin. rclpy is an optional dependency (``holosoma[ros2]``); this
    config stays import-safe without ROS because :meth:`get_cls` defers the impl import.
    """

    topic: str = "/clock"
    """Topic to publish the clock on (ROS2 ``use_sim_time`` consumers expect ``/clock``)."""

    node_name: str = "holosoma_clock"
    """ROS2 node name for the publisher."""

    publish_every: DecimationLike = 1
    """How often to publish, on the PHYSICS rate (the clock is read right after each physics
    step). Either a decimation int (publish every Nth physics step) or a frequency string
    (``"100Hz"``, ``">100Hz"``, ``"<100Hz"``) resolved at install time against ``fps``."""

    def __post_init__(self) -> None:
        validate_decimation_like(self.publish_every, field="publish_every")

    def get_cls(self) -> Callable[..., Any]:
        from holosoma.simulator.shared.ros2_plugins import ClockPublishPlugin

        return ClockPublishPlugin


@dataclass(frozen=True)
class GantryControlPluginConfig(PluginConfig):
    """Control and monitor the virtual gantry over independent ROS2 topics.

    Position / length / enabled each have a command subscription and a state publisher.
    Readbacks publish once per control frame after pending commands are applied. Set any
    topic to ``None`` to disable that individual subscription or publisher; CLI overrides
    accept the literal ``None`` (for example, ``--plugin.g.position-topic=None``). A ROS2
    example plugin; rclpy is optional (``holosoma[ros2]``), imported lazily via
    :meth:`get_cls`.
    """

    position_topic: str | None = "/gantry/position"
    """``geometry_msgs/msg/Point`` — new gantry anchor point ``(x, y, z)`` in world frame."""

    length_topic: str | None = "/gantry/length"
    """``std_msgs/msg/Float64`` — new elastic-band rest length."""

    enabled_topic: str | None = "/gantry/enabled"
    """``std_msgs/msg/Bool`` — enable (True) or disable (False) the gantry."""

    position_readback_topic: str | None = "/gantry/position/readback"
    """``geometry_msgs/msg/Point`` — current gantry anchor point in world frame."""

    length_readback_topic: str | None = "/gantry/length/readback"
    """``std_msgs/msg/Float64`` — current elastic-band rest length."""

    enabled_readback_topic: str | None = "/gantry/enabled/readback"
    """``std_msgs/msg/Bool`` — whether the gantry is currently enabled."""

    node_name: str = "holosoma_gantry_control"
    """ROS2 node name for the subscriptions and publishers."""

    def get_cls(self) -> Callable[..., Any]:
        from holosoma.simulator.shared.ros2_plugins import GantryControlPlugin

        return GantryControlPlugin


@pydantic_dataclass(frozen=True, config=_FORBID_EXTRA)
class ROS2OdometryPluginConfig(PluginConfig):
    """Publish a robot-attached frame as ``nav_msgs/Odometry`` over ROS2.

    ``body_name`` selects any rigid body in the robot and ``position``/``orientation`` define the
    published child frame relative to that body. Leaving ``body_name=None`` and using the identity
    transform preserves the original behavior of publishing ``simulator.robot_root_states``.
    rclpy is optional (``holosoma[ros2]``), imported lazily via :meth:`get_cls`.
    """

    node_name: str = "sim_odometry"
    """ROS2 node name created for this sink."""

    topic: str = "/odom"
    """Topic to publish the ``nav_msgs/Odometry`` on."""

    frame_id: str = "odom"
    """``header.frame_id``: the fixed frame the pose is expressed in (odometry origin)."""

    child_frame_id: str = "base_link"
    """``child_frame_id``: the published moving frame. Its twist is expressed in this frame."""

    body_name: str | None = None
    """Robot body to attach ``child_frame_id`` to. ``None`` uses the robot root state for backward
    compatibility; otherwise the name is resolved through ``simulator.find_rigid_body_indice``."""

    position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """Child-frame position ``[x, y, z]`` in the selected body's frame, in meters."""

    orientation: UnitQuaternionWXYZ = field(
        default_factory=lambda: [1.0, 0.0, 0.0, 0.0], metadata={"validate_default": True}
    )
    """Child-frame unit quaternion ``[w, x, y, z]`` in the selected body's frame.
    Norm deviations up to 1e-4 are normalized."""

    qos: str = "best_effort"
    """QoS profile: ``best_effort`` (default) or ``reliable``."""

    env_id: int = 0
    """Which environment's robot frame to publish. Default 0 (the single real-time robot)."""

    publish_every: DecimationLike = 1
    """How often to publish, on the CONTROL rate (body state is fresh after each frame's tensor
    refresh). Either a decimation int (publish every Nth control step) or a frequency string
    (``"50Hz"``, ``">50Hz"``, ``"<50Hz"``) resolved at install time against the control rate."""

    @field_validator("publish_every", mode="before")
    @classmethod
    def validate_publish_every(cls, value: Any) -> Any:
        # Preserve the rate contract before Pydantic can coerce bools/floats to ints.
        validate_decimation_like(value, field="publish_every")
        return value

    def __post_init__(self) -> None:
        if self.qos not in ("best_effort", "reliable"):
            raise ValueError(f"ROS2OdometryPluginConfig.qos must be 'best_effort' or 'reliable', got '{self.qos}'.")
        if not self.topic:
            raise ValueError("ROS2OdometryPluginConfig.topic must be a non-empty topic.")
        if self.env_id < 0:
            raise ValueError(f"ROS2OdometryPluginConfig.env_id must be >= 0, got {self.env_id}.")
        if self.body_name == "":
            raise ValueError("ROS2OdometryPluginConfig.body_name must be non-empty or None.")
        if len(self.position) != 3 or not all(math.isfinite(value) for value in self.position):
            raise ValueError(
                f"ROS2OdometryPluginConfig.position must contain 3 finite values [x, y, z], got {self.position}."
            )

    def get_cls(self) -> Callable[..., Any]:
        from holosoma.simulator.shared.ros2_plugins import ROS2OdometryPlugin

        return ROS2OdometryPlugin


@dataclass(frozen=True)
class MotionPlaybackPluginConfig(PluginConfig):
    """Kinematically drive the robot (and optionally scene objects) from a recorded motion clip.

    Rendering-only playback. Pair with ``--simulator.config.sim.kinematic_playback True``: the
    sim then propagates the plugin's per-frame state writes with forward kinematics only — no
    dynamics at all — and the loop ticks once per control frame. Without the flag (any dynamics
    loop, including training/eval), the plugin still works by re-pinning state every
    ``PRE_STEP``, so physics never accumulates more than one substep against the clip.

    Pair with a mounted camera and a frame consumer (``frame-writer``, ``viz-record``,
    ``ros2-image``) to render recorded joint angles + object motion.

    Clip format: any 2D matrix (``.npy`` / ``.npz`` with one array / ``.csv`` / ``.txt``), one
    row per frame. Columns left to right are the robot block then one 7-column ``[x, y, z,
    quat]`` block per object in ``objects`` order. The robot block is ``root_pose(7) +
    joint_pos(num_dof)`` (floating base) or ``joint_pos(num_dof)`` (fixed base), inferred from
    the width; joints are in the sim's DOF order. A ``.npz`` with a ``qpos`` key uses ``qpos`` as
    the robot block and ``object_motion_pos``/``object_motion_rot_wxyz`` as the optional object
    block.

    Replay is single-trajectory across envs: at any instant every env receives the same clip state
    (a vectorized run renders identical copies across envs) — there is no per-env clip assignment.
    This is the spatial axis and is independent of the episodic playlist below, which sequences
    clips over *time*: all envs advance through the same episode together.

    Playlist: ``motion_files`` is one or more clips played in order, each clip one episode. Between
    episodes the plugin rewinds to the next clip's first frame and emits ``EPISODE_END``/
    ``EPISODE_START`` so downstream plugins (lighting/material/camera/scene randomizers) re-roll
    their per-episode settings. The playlist plays ``n_run`` times in order (``A B C | A B C``);
    total episodes = ``len(motion_files) * n_run``, after which the run shuts down (counting clips
    after glob entries are expanded). ``n_run == 0`` plays the playlist forever until the
    run is stopped. A single clip played once emits no episode boundaries.
    """

    motion_files: list[str] = field(default_factory=list)
    """Playlist of motion matrices (.npy/.npz/.csv/.txt), played in order, one episode per clip.
    Accepts the same path forms as motion training data (absolute, relative, ``holosoma/...``
    package paths, ``s3://``). At least one clip is required.

    An entry may also be a glob pattern, so a large playlist need not be spelled out path by path::

        --plugin.play.motion_files=["1.npz"]                 1 clip
        --plugin.play.motion_files=["1.npz","2.npz"]         2 clips, in that order
        --plugin.play.motion_files=["motions/*.npz"]         pattern matches, sorted
        --plugin.play.motion_files=["s3://bucket/motions/*.npz"]     same, over a bucket prefix
        --plugin.play.motion_files=["warmup.npz","motions/*.npz"]   any mix; expands in place, entry order kept

    A pattern contributes only non-dot ``.npy``/``.npz``/``.csv``/``.txt`` files, and one matching
    nothing is an error rather than a silently shorter playlist. Matches sort lexicographically, so
    zero-pad numbered clips (``clip09.npz``) when they must play in numeric order — otherwise
    ``clip10`` plays before ``clip2``. Any entry that is not a pattern is used exactly as written."""

    n_run: int = 1
    """How many times to play the whole playlist. ``0`` plays it forever until the run is stopped;
    ``>= 1`` plays it that many times and then shuts down. Total episodes =
    ``len(motion_files) * n_run`` (unbounded when 0)."""

    settle_frames: int = 0
    """Frames to hold the clip at its first frame after each episode boundary before the clip
    starts advancing (>= 0, default 0 = no hold). Gives Isaac time to materialize per-episode
    changes applied on ``EPISODE_START`` (MDL/texture compile, prim activation, dome/HDR swap) and
    lets RTX path-tracing reconverge before each episode's rendered frames begin."""

    objects: list[str] = field(default_factory=list)
    """Scene-object names (``--scene.rigid_objects`` keys) to drive, in the same order as their
    7-column pose blocks appear after the robot block. Empty (default) drives only the robot."""

    fps: float = 50.0
    """Frame rate of the clip (rows per second), for mapping playback time to a row index."""

    quat_order: Literal["xyzw", "wxyz"] = "wxyz"
    """Quaternion column order in the file's pose blocks (root and objects). Written to the sim
    as xyzw regardless."""

    speed: float = 1.0
    """Playback speed factor in sim time: 2.0 plays the clip twice as fast."""

    start_time: float = 0.0
    """Clip time (seconds) to start playback from."""

    interpolate: bool = True
    """Linearly interpolate (slerp for quaternions) between clip frames when the control rate
    does not line up with the clip fps. False snaps to the nearest earlier frame."""

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError(f"MotionPlaybackPluginConfig.fps must be > 0, got {self.fps}.")
        if self.speed <= 0:
            raise ValueError(f"MotionPlaybackPluginConfig.speed must be > 0, got {self.speed}.")
        if self.start_time < 0:
            raise ValueError(f"MotionPlaybackPluginConfig.start_time must be >= 0, got {self.start_time}.")
        if self.n_run < 0:
            raise ValueError(f"MotionPlaybackPluginConfig.n_run must be >= 0, got {self.n_run}.")
        if self.settle_frames < 0:
            raise ValueError(f"MotionPlaybackPluginConfig.settle_frames must be >= 0, got {self.settle_frames}.")
        # A clip need not be set here: the default preset is registered as an empty template and the
        # clip requirement is enforced at plugin construction (MotionPlaybackPlugin.__init__).
        dupes = {o for o in self.objects if self.objects.count(o) > 1}
        if dupes:
            raise ValueError(f"MotionPlaybackPluginConfig.objects has duplicates: {sorted(dupes)}.")

    def get_cls(self) -> Callable[..., Any]:
        from holosoma.simulator.plugins.playback import MotionPlaybackPlugin

        return MotionPlaybackPlugin


# ----------------------------------------------------------------------------------------------- #
# Camera-frame egress plugins
#
# These plugins consume the sim's rendered camera frames and push them to a sink: ROS2 topics, a
# live cv2 window, or an mp4. Each is a ``PluginConfig`` selected as ``plugin.<key>:<variant>`` just
# like the plugins above; ``get_cls`` returns a
# :class:`~holosoma.simulator.plugins.camera_consumer.CameraConsumerPlugin` subclass, imported lazily
# so the transport dependency (rclpy, cv2, …) loads only when the sink is selected. A consumer reads
# the raw rendered buffer from ``BaseSimulator.get_camera_data`` (rgb ``uint8`` R,G,B; depth
# ``float32`` meters), passing ``device="cpu"`` so the first reader per (camera, modality) pays the
# one device->host copy and the rest share it.
# ----------------------------------------------------------------------------------------------- #

# Depth colormap names accepted by the viz plugin (mapped to cv2.COLORMAP_* there).
_DEPTH_COLORMAPS = ("inferno", "turbo", "viridis", "magma", "jet", "gray")

# Modalities an egress route can carry.
EgressModality = Literal["rgb", "depth"]

# Wire encodings a ROS2 image route may request:
#   - "rgb8"  : raw sensor_msgs/Image, R,G,B (no BGR swap; that is a cv2/JPEG artifact only).
#   - "jpeg"  : sensor_msgs/CompressedImage, lossy (teleop/viz, not depth or training datasets).
#   - "png"   : sensor_msgs/CompressedImage, lossless RGB.
#   - "32FC1" : raw sensor_msgs/Image depth, float32 meters (matches get_camera_data).
#   - "16UC1" : raw sensor_msgs/Image depth, uint16 millimeters.
#
# A ``depth`` route MAY pick an rgb format (rgb8/jpeg/png): the depth map is then COLORIZED to RGB
# (same colormap the viz plugin uses) before encoding, for a human-viewable stream. A ``depth`` route
# with a depth format (32FC1/16UC1) publishes the raw metric depth. An ``rgb`` route may only use an
# rgb format (there is nothing to colorize).
ROS2ImageFormat = Literal["rgb8", "jpeg", "png", "32FC1", "16UC1"]
_RGB_FORMATS = ("rgb8", "jpeg", "png")
_DEPTH_FORMATS = ("32FC1", "16UC1")


@pydantic_dataclass(frozen=True, config=_FORBID_EXTRA)
class ROS2ImageRoute:
    """One camera-stream to ROS2-topic mapping within a :class:`ROS2ImagePluginConfig`."""

    camera: str
    """Camera name; must match a key in the active ``--sensor`` camera dict."""

    topic: str
    """ROS2 topic to publish on, used verbatim (no auto-suffixing). For a CompressedImage
    (``jpeg``/``png``) the ROS convention is a ``/compressed`` suffix, e.g.
    ``/sim_cameras/head/image/compressed``; spell it out here if you want it."""

    modality: EgressModality = "rgb"
    """Which rendered modality to publish; must be in the camera's ``data_types``."""

    format: ROS2ImageFormat = "jpeg"
    """Wire encoding (see :data:`ROS2ImageFormat`). An ``rgb`` route needs an rgb format; a ``depth``
    route may pick a depth format (raw metric) OR an rgb format (colorized to RGB before encoding)."""

    depth_colormap: str = "inferno"
    """Colormap used when a ``depth`` route is colorized to RGB (rgb format): inferno (default),
    turbo, viridis, magma, jet, or gray. Ignored for raw-depth and rgb routes."""

    depth_range: list[float] | None = None
    """Fixed ``[min_m, max_m]`` depth range (meters) for stable colorization of a colorized ``depth``
    route; ``None`` means ``[0.01, 5.0]``. ``+inf`` (no hit) maps to the far end. Ignored otherwise."""

    @model_validator(mode="after")
    def validate_route(self) -> ROS2ImageRoute:
        if not self.camera:
            raise ValueError("ROS2ImageRoute.camera must be a non-empty camera name.")
        if not self.topic:
            raise ValueError(f"ROS2ImageRoute for camera '{self.camera}' needs a non-empty topic.")
        rgb_fmt = self.format in _RGB_FORMATS
        # rgb modality must use an rgb format. depth modality accepts either: a depth format (raw) or
        # an rgb format (colorized to RGB before encoding) — so only rgb+depth-format is rejected.
        if self.modality == "rgb" and not rgb_fmt:
            raise ValueError(
                f"ROS2ImageRoute camera '{self.camera}': modality 'rgb' needs an rgb format "
                f"{_RGB_FORMATS}, got '{self.format}'."
            )
        # depth is colorized only when the format is an rgb one; the colormap/range knobs are used
        # then. Validate the colormap and range regardless so a misconfig fails loud at construction.
        if self.depth_colormap not in _DEPTH_COLORMAPS:
            raise ValueError(
                f"ROS2ImageRoute camera '{self.camera}': depth_colormap '{self.depth_colormap}' "
                f"unknown; allowed: {sorted(_DEPTH_COLORMAPS)}."
            )
        if self.depth_range is not None and (len(self.depth_range) != 2 or self.depth_range[0] >= self.depth_range[1]):
            raise ValueError(
                f"ROS2ImageRoute camera '{self.camera}': depth_range must be [min_m, max_m] with "
                f"min<max, got {self.depth_range}."
            )
        return self


@pydantic_dataclass(frozen=True, config=_FORBID_EXTRA)
class ROS2ImagePluginConfig(PluginConfig):
    """One ROS2 image-publishing sink: a single node fanning out to the cameras in ``routes``."""

    node_name: str = "sim_cameras"
    """ROS2 node name created for this sink."""

    qos: str = "best_effort"
    """QoS profile: ``best_effort`` (default, matches ZED/sensor drivers) or ``reliable``."""

    async_publish: bool = False
    """False (default) encodes and publishes inline on the simulation thread so no frames are
    skipped. True uses per-route worker threads and drops the oldest frame under backpressure."""

    queue_maxlen: int = 2
    """Per-route bounded queue depth when ``async_publish``. Drop-oldest beyond this (latest wins):
    1 keeps the freshest frame only, 2 gives one frame of jitter tolerance. Ignored when not async."""

    publish_camera_info: bool = True
    """Also publish a latched ``sensor_msgs/CameraInfo``. Pinhole intrinsics are exact; native Isaac
    Sim fisheye calibration is approximated with ROS's ``equidistant`` distortion model."""

    jpeg_quality: int = 50
    """JPEG encode quality 1-100 for ``jpeg`` routes (ignored by other formats). Default 50."""

    env_id: int = 0
    """Which environment's view to publish. Default 0 (the single real-time robot); set higher to
    stream a specific env of a vectorized run. One env per node; all routes share it."""

    routes: dict[str, ROS2ImageRoute] = field(default_factory=dict)
    """Camera-to-topic routes this node publishes, keyed by an arbitrary label. The key is a
    CLI handle only (like a list index was) — it does not affect publishing; the route's
    ``camera``/``topic`` fields do."""

    def get_cls(self) -> Callable[..., Any]:
        # Deferred import: keeps rclpy out of CLI-build import.
        from holosoma.simulator.plugins.ros2.ros2_image_plugin import ROS2ImagePlugin

        return ROS2ImagePlugin

    @model_validator(mode="after")
    def validate_egress(self) -> ROS2ImagePluginConfig:
        if self.qos not in ("best_effort", "reliable"):
            raise ValueError(f"ROS2ImagePluginConfig.qos must be 'best_effort' or 'reliable', got '{self.qos}'.")
        if self.queue_maxlen < 1:
            raise ValueError(f"ROS2ImagePluginConfig.queue_maxlen must be >= 1, got {self.queue_maxlen}.")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError(f"ROS2ImagePluginConfig.jpeg_quality must be in [1, 100], got {self.jpeg_quality}.")
        if self.env_id < 0:
            raise ValueError(f"ROS2ImagePluginConfig.env_id must be >= 0, got {self.env_id}.")
        topics = [r.topic for r in self.routes.values()]
        dupes = {t for t in topics if topics.count(t) > 1}
        if dupes:
            raise ValueError(f"ROS2ImagePluginConfig node '{self.node_name}' has duplicate topics: {sorted(dupes)}.")
        return self


@pydantic_dataclass(frozen=True, config=_FORBID_EXTRA)
class ROS2PointCloudRoute:
    """One LiDAR-to-PointCloud2 topic mapping."""

    lidar: str
    """Configured LiDAR sensor name whose scans this route publishes."""

    topic: str
    """PointCloud2 topic name."""

    frame_id: str | None = None
    """ROS frame for the sensor-local points. ``None`` uses the LiDAR sensor name."""

    point_translation: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """Translation ``[x, y, z]`` in meters, expressed in ``frame_id`` coordinates. The publisher
    transforms every point as ``p_frame = R * p_sensor + translation``."""

    point_rotation: UnitQuaternionWXYZ = field(
        default_factory=lambda: [1.0, 0.0, 0.0, 0.0], metadata={"validate_default": True}
    )
    """Unit quaternion ``[w, x, y, z]`` rotating LiDAR optical-frame points into ``frame_id``.
    Norm deviations up to 1e-4 are normalized; leave it as the identity when both frames use the
    same axes."""

    @model_validator(mode="after")
    def validate_route(self) -> ROS2PointCloudRoute:
        if not self.lidar:
            raise ValueError("ROS2PointCloudRoute.lidar must be a non-empty sensor name.")
        if not self.topic:
            raise ValueError(f"ROS2PointCloudRoute for LiDAR '{self.lidar}' needs a non-empty topic.")
        if self.frame_id == "":
            raise ValueError("ROS2PointCloudRoute.frame_id must be non-empty or None.")
        if len(self.point_translation) != 3 or not all(math.isfinite(value) for value in self.point_translation):
            raise ValueError(
                "ROS2PointCloudRoute.point_translation must contain 3 finite values "
                f"[x, y, z], got {self.point_translation}."
            )
        return self


@pydantic_dataclass(frozen=True, config=_FORBID_EXTRA)
class ROS2PointCloudPluginConfig(PluginConfig):
    """Publish mounted LiDAR scans as shape-preserving ``sensor_msgs/PointCloud2`` messages."""

    node_name: str = "sim_lidar"
    """ROS2 node name created for this point-cloud publisher."""

    qos: str = "best_effort"
    """QoS profile: ``"best_effort"`` (default) or ``"reliable"``."""

    async_publish: bool = False
    """False (default) encodes and publishes on the simulation thread so no scans are skipped.
    True uses bounded worker queues and drops the oldest scan under backpressure."""

    queue_maxlen: int = 2
    """Per-route queue depth for asynchronous publishing; the oldest scan is dropped when full."""

    env_id: int = 0
    """Environment index to publish from in a vectorized simulation."""

    routes: dict[str, ROS2PointCloudRoute] = field(default_factory=dict)
    """LiDAR-to-topic routes, keyed by arbitrary configuration labels."""

    def get_cls(self) -> Callable[..., Any]:
        from holosoma.simulator.plugins.ros2.ros2_pointcloud_plugin import ROS2PointCloudPlugin

        return ROS2PointCloudPlugin

    @model_validator(mode="after")
    def validate_egress(self) -> ROS2PointCloudPluginConfig:
        if not self.node_name.strip():
            raise ValueError("ROS2PointCloudPluginConfig.node_name must be non-empty.")
        if self.qos not in ("best_effort", "reliable"):
            raise ValueError(f"ROS2PointCloudPluginConfig.qos must be 'best_effort' or 'reliable', got '{self.qos}'.")
        if self.queue_maxlen < 1:
            raise ValueError(f"ROS2PointCloudPluginConfig.queue_maxlen must be >= 1, got {self.queue_maxlen}.")
        if self.env_id < 0:
            raise ValueError(f"ROS2PointCloudPluginConfig.env_id must be >= 0, got {self.env_id}.")
        topics = [route.topic for route in self.routes.values()]
        duplicates = {topic for topic in topics if topics.count(topic) > 1}
        if duplicates:
            raise ValueError(
                f"ROS2PointCloudPluginConfig node '{self.node_name}' has duplicate topics: {sorted(duplicates)}."
            )
        return self


@pydantic_dataclass(frozen=True, config=_FORBID_EXTRA)
class FrameWriterPluginConfig(PluginConfig):
    """Write every watched camera stream's frames to disk as an image sequence (dataset capture).

    Inline on the sim thread: lossless and every-frame (the sim waits on the disk write). Each
    ``(camera, modality, env)`` stream gets its own directory of numbered frames; an
    ``index.jsonl`` records intrinsics and one line per frame (path, camera, modality, env,
    frame index, sim time).
    """

    output_dir: str = "logs/frames"
    """Directory for the frame directories and ``index.jsonl``. Created if missing."""

    cameras: list[str] | None = None
    """Camera names to capture; ``None`` (default) means all configured cameras."""

    modalities: list[EgressModality] | None = None
    """Modalities to capture; ``None`` (default) means every modality each camera produces."""

    env_ids: list[int] | None = None
    """Environments to capture; ``None`` (default) means all environments."""

    rgb_format: Literal["png", "jpeg", "mp4"] = "png"
    """RGB output: ``png`` (default, lossless) or ``jpeg`` image sequences (names match the ROS2
    egress), or ``mp4`` for one H.264 video per stream (buffered, encoded at teardown; fps from
    the frames' sim-time stamps)."""

    jpeg_quality: int = 95
    """JPEG encode quality 1-100 for ``jpeg`` (ignored for ``png``). Same knob as the ROS2 egress;
    default 95 here (dataset capture) vs 50 there (streaming)."""

    depth_format: Literal["png16", "npy"] = "png16"
    """Depth file format: ``png16`` (uint16 millimeters, +inf saturates — the ROS2 egress's
    ``16UC1``) or ``npy`` (float32 meters, exact — the ROS2 egress's ``32FC1``)."""

    def get_cls(self) -> Callable[..., Any]:
        # Deferred import: keeps cv2 out of CLI-build import.
        from holosoma.simulator.plugins.frame_writer import FrameWriterPlugin

        return FrameWriterPlugin

    @model_validator(mode="after")
    def validate_writer(self) -> FrameWriterPluginConfig:
        if not self.output_dir:
            raise ValueError("FrameWriterPluginConfig.output_dir must be non-empty.")
        if self.env_ids is not None and (not self.env_ids or any(e < 0 for e in self.env_ids)):
            raise ValueError(
                f"FrameWriterPluginConfig.env_ids must be None (all envs) or a non-empty list of "
                f"ids >= 0, got {self.env_ids}."
            )
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError(f"FrameWriterPluginConfig.jpeg_quality must be in [1, 100], got {self.jpeg_quality}.")
        return self


@pydantic_dataclass(frozen=True, config=_FORBID_EXTRA)
class CameraVizPluginConfig(PluginConfig):
    """Local visualization plugin: tile mounted-camera views into a live cv2 window and/or an mp4.

    Tiles all watched (camera, modality, env) panels into one grid per step. Inline only:
    ``publish`` composes the grid and shows or buffers it on the calling thread.
    """

    live_window: bool = False
    """Show a live ``cv2`` window of the camera view(s). Needs a display; ignored headless."""

    record_video: bool = False
    """Buffer frames and encode an mp4 (H.264) at stop."""

    env_ids: list[int] = field(default_factory=lambda: [0])
    """Environments to visualize (default ``[0]``). Multiple env ids tile as a grid: one row per
    env, one column per (camera, modality) panel."""

    cameras: list[str] | None = None
    """Camera names to show; ``None`` (default) means all configured cameras."""

    modalities: list[EgressModality] | None = None
    """Modalities to show; ``None`` (default) means every modality each selected camera produces."""

    depth_range: list[float] | None = None
    """Fixed ``[min_m, max_m]`` depth range (meters) for stable colorization; ``None`` means
    ``[0.01, 5.0]``. ``+inf`` (no hit) maps to the far end."""

    depth_colormap: str = "inferno"
    """OpenCV colormap for depth: inferno (default), turbo, viridis, magma, jet, or gray."""

    update_decimation: DecimationLike = 1
    """Int visualizes every Nth rendered frame of the fastest watched camera; a frequency string
    ("10Hz") is a target against the control rate, converted to a frame-decimation by the recorder."""

    playback_rate: float = 1.0
    """Video playback speed factor; 1.0 plays back at true wall-clock speed."""

    save_dir: str | None = None
    """Output directory for the video; ``None`` derives it from the experiment/video dir."""

    def get_cls(self) -> Callable[..., Any]:
        # Deferred import: keeps cv2/video utils out of CLI-build import.
        from holosoma.simulator.plugins.viz.viz_plugin import CameraVizPlugin

        return CameraVizPlugin

    @model_validator(mode="after")
    def validate_recorder(self) -> CameraVizPluginConfig:
        validate_decimation_like(self.update_decimation, field="CameraVizPluginConfig.update_decimation")
        if not self.env_ids:
            raise ValueError("CameraVizPluginConfig.env_ids must be non-empty (default [0]).")
        if any(e < 0 for e in self.env_ids):
            raise ValueError(f"CameraVizPluginConfig.env_ids must all be >= 0, got {self.env_ids}.")
        if self.depth_range is not None and (len(self.depth_range) != 2 or self.depth_range[0] >= self.depth_range[1]):
            raise ValueError(
                f"CameraVizPluginConfig.depth_range must be [min_m, max_m] with min<max, got {self.depth_range}."
            )
        if self.depth_colormap not in _DEPTH_COLORMAPS:
            raise ValueError(
                f"CameraVizPluginConfig.depth_colormap '{self.depth_colormap}' unknown; "
                f"allowed: {sorted(_DEPTH_COLORMAPS)}."
            )
        return self


@pydantic_dataclass(frozen=True, config=_FORBID_EXTRA)
class LidarVizPluginConfig(PluginConfig):
    """Locally inspect fresh LiDAR scans in a live 3D view, PNG sequence, and/or MP4.

    This is a debugging sink: it consumes already-produced sensor-local XYZ scans and does not
    change LiDAR ray casting, cadence, or ROS output. A saved sequence contains one range-colored
    3D point-cloud PNG per fresh scan. Video frames are rendered from the same 3D plots and encoded
    at teardown, one MP4 per selected ``(LiDAR, environment)`` stream.
    """

    live_window: bool = False
    """Show one live OpenCV window per selected ``(LiDAR, environment)`` stream. Each window
    displays the range-colored 3D view configured below. Requires a non-headless simulator and a
    usable ``DISPLAY``; otherwise it is disabled."""

    save_scans: bool = False
    """Save one numbered range-colored 3D point-cloud PNG for every fresh selected scan."""

    record_video: bool = False
    """Buffer one rendered 3D frame per fresh selected scan and encode an H.264 MP4 at teardown.
    One video is written for each selected ``(LiDAR, environment)`` stream."""

    playback_rate: float = 1.0
    """Video playback-speed multiplier. ``1.0`` preserves the scan cadence derived from simulation
    timestamps; values above 1.0 speed the MP4 up and values below 1.0 slow it down."""

    env_ids: list[int] = field(default_factory=lambda: [0])
    """Environment indices to visualize. The default ``[0]`` targets the real-time environment;
    each selected LiDAR/environment pair has its own live window and saved-scan directory."""

    lidars: list[str] | None = None
    """LiDAR sensor names to visualize. ``None`` (default) selects every configured LiDAR."""

    point_size: float = 2.0
    """Matplotlib marker area in points squared. Increase for sparse scans or decrease for dense scans."""

    max_points: int | None = 100_000
    """Maximum finite points rendered per scan. ``None`` renders every finite point; otherwise the
    renderer takes a deterministic evenly spaced subset to keep debugging responsive."""

    view_elevation: float = 24.0
    """Initial 3D camera elevation in degrees for live windows and saved figures."""

    view_azimuth: float = -64.0
    """Initial 3D camera azimuth in degrees for live windows and saved figures."""

    save_dir: str | None = None
    """Directory for PNG sequences and per-stream MP4 files. ``None`` uses the experiment video
    directory when configured, otherwise ``logs/lidar_sensors``."""

    def get_cls(self) -> Callable[..., Any]:
        # Deferred import: Matplotlib is loaded only when this optional debug sink is selected.
        from holosoma.simulator.plugins.viz.lidar_viz_plugin import LidarVizPlugin

        return LidarVizPlugin

    @model_validator(mode="after")
    def validate_viz(self) -> LidarVizPluginConfig:
        if not self.env_ids:
            raise ValueError("LidarVizPluginConfig.env_ids must be non-empty (default [0]).")
        if any(env_id < 0 for env_id in self.env_ids):
            raise ValueError(f"LidarVizPluginConfig.env_ids must all be >= 0, got {self.env_ids}.")
        if not math.isfinite(self.point_size) or self.point_size <= 0.0:
            raise ValueError(f"LidarVizPluginConfig.point_size must be finite and > 0, got {self.point_size}.")
        if self.max_points is not None and self.max_points < 1:
            raise ValueError(f"LidarVizPluginConfig.max_points must be >= 1 or None, got {self.max_points}.")
        if not math.isfinite(self.view_elevation):
            raise ValueError(f"LidarVizPluginConfig.view_elevation must be finite degrees, got {self.view_elevation}.")
        if not math.isfinite(self.view_azimuth):
            raise ValueError(f"LidarVizPluginConfig.view_azimuth must be finite degrees, got {self.view_azimuth}.")
        if not math.isfinite(self.playback_rate) or self.playback_rate <= 0.0:
            raise ValueError(f"LidarVizPluginConfig.playback_rate must be finite and > 0, got {self.playback_rate}.")
        if self.save_dir == "":
            raise ValueError("LidarVizPluginConfig.save_dir must be a non-empty directory or None.")
        return self
