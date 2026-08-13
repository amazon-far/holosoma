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
``BaseSimulator.__init__`` then instantiates each ``get_cls()`` against itself.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, Callable, Literal

from pydantic import ConfigDict, model_validator
from pydantic.dataclasses import dataclass as pydantic_dataclass

from holosoma.config_types.frequency import DecimationLike, validate_decimation_like


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
    """Control the virtual gantry over ROS2 via three independent standard-message topics.

    Each of position / length / enabled is its own subscription, so publishing to one
    topic changes only that property (the others are left untouched). A ROS2 example
    plugin; rclpy is optional (``holosoma[ros2]``), imported lazily via :meth:`get_cls`.
    """

    position_topic: str = "/gantry/position"
    """``geometry_msgs/msg/Point`` — new gantry anchor point ``(x, y, z)`` in world frame."""

    length_topic: str = "/gantry/length"
    """``std_msgs/msg/Float64`` — new elastic-band rest length."""

    enabled_topic: str = "/gantry/enabled"
    """``std_msgs/msg/Bool`` — enable (True) or disable (False) the gantry."""

    node_name: str = "holosoma_gantry_control"
    """ROS2 node name for the subscriber."""

    def get_cls(self) -> Callable[..., Any]:
        from holosoma.simulator.shared.ros2_plugins import GantryControlPlugin

        return GantryControlPlugin


@dataclass(frozen=True)
class ROS2OdometryPluginConfig(PluginConfig):
    """Publish the robot base pose/velocity as ``nav_msgs/Odometry`` over ROS2.

    A self-sourced (non-camera) egress plugin: it reads base pose/velocity straight off
    ``simulator.robot_root_states`` each control step (the sim analog of the robot's onboard
    sport/odom estimate) — so it uses no camera base class, just registers a ``FRAME_END`` publish
    like :class:`ClockPublishPluginConfig`. rclpy is optional (``holosoma[ros2]``), imported lazily
    via :meth:`get_cls`.
    """

    node_name: str = "sim_odometry"
    """ROS2 node name created for this sink."""

    topic: str = "/odom"
    """Topic to publish the ``nav_msgs/Odometry`` on."""

    frame_id: str = "odom"
    """``header.frame_id``: the fixed frame the pose is expressed in (odometry origin)."""

    child_frame_id: str = "base_link"
    """``child_frame_id``: the moving body frame the twist is expressed in."""

    qos: str = "best_effort"
    """QoS profile: ``best_effort`` (default) or ``reliable``."""

    env_id: int = 0
    """Which environment's base state to publish. Default 0 (the single real-time robot)."""

    publish_every: DecimationLike = 1
    """How often to publish, on the CONTROL rate (base state is fresh after each frame's tensor
    refresh). Either a decimation int (publish every Nth control step) or a frequency string
    (``"50Hz"``, ``">50Hz"``, ``"<50Hz"``) resolved at install time against the control rate."""

    def __post_init__(self) -> None:
        validate_decimation_like(self.publish_every, field="publish_every")
        if self.qos not in ("best_effort", "reliable"):
            raise ValueError(f"ROS2OdometryPluginConfig.qos must be 'best_effort' or 'reliable', got '{self.qos}'.")
        if not self.topic:
            raise ValueError("ROS2OdometryPluginConfig.topic must be a non-empty topic.")
        if self.env_id < 0:
            raise ValueError(f"ROS2OdometryPluginConfig.env_id must be >= 0, got {self.env_id}.")

    def get_cls(self) -> Callable[..., Any]:
        from holosoma.simulator.shared.ros2_plugins import ROS2OdometryPlugin

        return ROS2OdometryPlugin


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

# Reject unknown fields on every egress config.
_FORBID_EXTRA = ConfigDict(extra="forbid")

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

    async_publish: bool = True
    """True (default): snapshot on the sim thread, encode and publish on per-route worker threads
    (drop-oldest under backpressure). False: encode and publish inline on the sim thread, lossless
    and every-frame (dataset capture); the sim waits."""

    queue_maxlen: int = 2
    """Per-route bounded queue depth when ``async_publish``. Drop-oldest beyond this (latest wins):
    1 keeps the freshest frame only, 2 gives one frame of jitter tolerance. Ignored when not async."""

    publish_camera_info: bool = True
    """Also publish a latched ``sensor_msgs/CameraInfo`` per camera (static K from intrinsics)."""

    jpeg_quality: int = 50
    """JPEG encode quality 1-100 for ``jpeg`` routes (ignored by other formats). Default 50."""

    env_id: int = 0
    """Which environment's view to publish. Default 0 (the single real-time robot); set higher to
    stream a specific env of a vectorized run. One env per node; all routes share it."""

    routes: dict[str, ROS2ImageRoute] = dc_field(default_factory=dict)
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
class CameraVizPluginConfig(PluginConfig):
    """Local visualization plugin: tile mounted-camera views into a live cv2 window and/or an mp4.

    Tiles all watched (camera, modality, env) panels into one grid per step. Inline only:
    ``publish`` composes the grid and shows or buffers it on the calling thread.
    """

    live_window: bool = False
    """Show a live ``cv2`` window of the camera view(s). Needs a display; ignored headless."""

    record_video: bool = False
    """Buffer frames and encode an mp4 (H.264) at stop."""

    env_ids: list[int] = dc_field(default_factory=lambda: [0])
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
