"""Sensor configuration types.

A sensor is a read-only producer mounted onto an existing body (a robot link or a
spawned scene actor) that it follows.

The camera optical frame is ``-Z`` forward, ``+Y`` up (OpenGL/USD/MuJoCo-native). Each
backend converts this convention to its native basis.
"""

from __future__ import annotations

from dataclasses import field
from typing import Literal, cast, get_args

from pydantic import ConfigDict, model_validator
from pydantic.dataclasses import dataclass

from holosoma.config_types.frequency import DecimationLike, validate_decimation_like

# Reject unknown fields on every sensor config (a typo'd field fails at construction).
_FORBID_EXTRA = ConfigDict(extra="forbid")

# Camera data types. "rgb" and "depth" are implemented.
CameraDataType = Literal["rgb", "depth"]
CAMERA_DATA_TYPES = cast("tuple[CameraDataType, ...]", get_args(CameraDataType))

# Depth-clipping behavior for a depth camera: what the sensor reports for rays that hit nothing
# within the far plane (IsaacSim's ``distance_to_image_plane`` is ``+inf`` there). These are the
# exact values of IsaacLab's ``TiledCameraCfg.depth_clipping_behavior`` — passed through natively:
#   - "none"  : keep ``+inf`` for no-hit (raw sensor output; consumers must handle non-finite).
#   - "max"   : clip ``+inf`` (and anything beyond ``far``) to the ``far`` clip distance in meters.
#   - "zero"  : set no-hit to ``0.0`` (the ROS ``16UC1``/OpenNI no-return convention).
DepthClippingBehavior = Literal["none", "max", "zero"]

# Which body a sensor mounts on:
#   - "robot_link" resolves through the robot-only body index (find_rigid_body_indice).
#   - "actor" resolves a spawned scene/individual actor through the ObjectRegistry.
#   - "world" is a free-floating camera fixed in each env's frame (no body to follow); ``target``
#     is unused. position/orientation are the pose in the per-env frame (env origin + offset), so
#     every env gets its own fixed camera at the same relative spot. Useful for logging/overview.
MountKind = Literal["robot_link", "actor", "world"]

# Pixel memory layout of a transformed image. ``get_camera_data`` returns HWC
# (row-major, channel-last); "CHW" emits channel-first for torch vision policies.
ImageLayout = Literal["HWC", "CHW"]

# Output dtype/range of a transformed image:
#   - "native"     : passthrough, no scaling (rgb uint8 [0,255]; depth float32 meters).
#   - "float01"    : float32 in [0,1]  (rgb /255; depth normalized via depth_range).
#   - "float_pm1"  : float32 in [-1,1] (rgb /127.5-1; depth normalized then mapped to [-1,1]).
ImageScale = Literal["native", "float01", "float_pm1"]


@dataclass(frozen=True, config=_FORBID_EXTRA)
class ImageTransformConfig:
    """Transform applied to a camera obs term's frame, for visual policies.

    Applied after the ``get_camera_data`` read by ``apply_image_transform``, in a fixed order
    (resize, scale, layout, flatten). All defaults are no-ops, leaving the ``[N, H, W, C]`` frame
    untouched.
    """

    resize: list[int] | None = None
    """Target ``[H, W]`` in pixels; ``None`` keeps the native resolution. RGB uses bilinear,
    depth uses nearest."""

    layout: ImageLayout = "HWC"
    """Axis order of the (un-flattened) output (see :data:`ImageLayout`). Defaults to HWC.
    ``CHW`` for torch vision policies; also the serialization order when ``flatten``."""

    scale: ImageScale = "native"
    """Output dtype/range (see :data:`ImageScale`). Defaults to passthrough (rgb uint8,
    depth float32 meters)."""

    flatten: bool = False
    """Flatten the per-env image to a 1-D ``[N, C*H*W]`` vector (consumed by ``CNNWrapper`` via
    ``view(N, C, H, W)``). Requires ``layout="CHW"`` to match that reshape; ``False`` keeps the
    4-D image."""

    depth_range: list[float] | None = None
    """For depth under a float ``scale``: ``[min_m, max_m]`` mapped to the output range (``+inf``
    no-hit maps to the far end). Required when scaling depth to float; ignored for rgb and ``native``."""

    @model_validator(mode="after")
    def validate_transform(self) -> ImageTransformConfig:
        if self.resize is not None and (len(self.resize) != 2 or self.resize[0] <= 0 or self.resize[1] <= 0):
            raise ValueError(f"ImageTransformConfig.resize must be positive [H, W], got {self.resize}.")
        if self.flatten and self.layout != "CHW":
            raise ValueError(
                f"ImageTransformConfig.flatten requires layout='CHW' (the order CNNWrapper reshapes "
                f"back via view(N, C, H, W)); got layout='{self.layout}'."
            )
        if self.depth_range is not None and (len(self.depth_range) != 2 or self.depth_range[0] >= self.depth_range[1]):
            raise ValueError(
                f"ImageTransformConfig.depth_range must be [min_m, max_m] with min<max, got {self.depth_range}."
            )
        return self


@dataclass(frozen=True, config=_FORBID_EXTRA)
class IsaacSimCameraConfig:
    """IsaacSim-only camera knobs (USD ``PinholeCameraCfg``) beyond the agnostic core.

    All ``None`` (default) derive from the core fields.
    """

    focal_length: float | None = None
    """Lens focal length in USD units (the aperture pair is derived from it to hit ``vertical_fov``).
    ``None`` means 24.0."""

    f_stop: float | None = None
    """Aperture f-stop. ``0.0`` (default when None) is pinhole / no depth-of-field; ``>0`` enables
    defocus blur (IsaacSim-only)."""

    focus_distance: float | None = None
    """Focus distance in meters (only meaningful when ``f_stop`` > 0). ``None`` keeps the default."""

    depth_clipping_behavior: DepthClippingBehavior = "none"
    """For a depth camera, what the TiledCamera reports where a ray hits nothing within ``far``
    (``distance_to_image_plane`` is ``+inf`` there). Passed through to IsaacLab's
    ``TiledCameraCfg.depth_clipping_behavior``: ``"none"`` (default) keeps the raw ``+inf``;
    ``"max"`` clips no-hit and anything beyond ``far`` to the ``far`` distance; ``"zero"`` maps
    no-hit to ``0.0``. Only affects the ``depth`` modality on IsaacSim."""


@dataclass(frozen=True, config=_FORBID_EXTRA)
class IsaacGymCameraConfig:
    """IsaacGym-only camera knobs (``gymapi.CameraProperties``) beyond the agnostic core."""

    supersampling_horizontal: int | None = None
    """Horizontal supersampling factor (anti-aliasing); ``None`` keeps IsaacGym's default (1)."""

    supersampling_vertical: int | None = None
    """Vertical supersampling factor (anti-aliasing); ``None`` keeps IsaacGym's default (1)."""

    use_collision_geometry: bool | None = None
    """Render collision geometry instead of visual meshes; ``None`` keeps the default (False)."""


@dataclass(frozen=True, config=_FORBID_EXTRA)
class MujocoCameraConfig:
    """MuJoCo-only camera knobs (``<camera>`` + Warp render context) beyond the agnostic core.

    The three appearance flags below are global to the Warp render context, not per-camera; cameras
    that set a given flag must agree (validated in :func:`validate_camera_dict`).
    ``None`` is unset (renderer default)."""

    use_shadows: bool | None = None
    """(Warp, global) Render shadows; ``None`` keeps the renderer default (off). Ignored by classic."""

    use_textures: bool | None = None
    """(Warp, global) Apply textures; ``None`` keeps the renderer default (on). Ignored by classic."""

    use_precomputed_rays: bool | None = None
    """(Warp, global) Precompute camera rays; set ``False`` to allow per-step intrinsics
    domain-randomization. ``None`` keeps the renderer default (True). Ignored by classic."""


@dataclass(frozen=True, config=_FORBID_EXTRA)
class SensorMountConfig:
    """Where a sensor is mounted and its fixed offset (on that body, or in the per-env frame)."""

    target_kind: MountKind = "robot_link"
    """Which namespace ``target`` is resolved in (see :data:`MountKind`)."""

    target: str = ""
    """Body/actor name. Required and non-empty for ``robot_link`` (a robot link name; use the root
    link, e.g. ``"pelvis"``, to mount on the base) and ``actor`` (an ObjectRegistry actor name).
    Must be empty for ``world`` (a free-floating camera anchors to no body)."""

    position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """Offset position ``[x, y, z]`` in meters. In the mount-body frame for ``robot_link``/``actor``;
    in the per-env frame (env origin + offset) for ``world``. Defaults to origin."""

    orientation: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])  # [w,x,y,z]
    """Offset orientation quaternion ``[w, x, y, z]`` (w-first). In the mount-body frame for
    ``robot_link``/``actor``; in the per-env frame for ``world``. Identity means the camera optical
    axis is ``-Z`` forward / ``+Y`` up."""

    @model_validator(mode="after")
    def validate_mount(self) -> SensorMountConfig:
        """``position``/``orientation`` arity, and the target/kind contract."""
        if len(self.position) != 3:
            raise ValueError(f"SensorMountConfig.position must have 3 elements [x,y,z], got {self.position}.")
        if len(self.orientation) != 4:
            raise ValueError(f"SensorMountConfig.orientation must have 4 elements [w,x,y,z], got {self.orientation}.")
        if self.target_kind == "world":
            if self.target:
                raise ValueError(
                    f"SensorMountConfig.target must be empty for target_kind='world' (a free-floating "
                    f"camera anchors to no body); got '{self.target}'."
                )
        elif not self.target:
            raise ValueError(
                f"SensorMountConfig.target must be a non-empty body/actor name when target_kind='{self.target_kind}'."
            )
        return self


@dataclass(frozen=True, config=_FORBID_EXTRA)
class CameraSensorConfig:
    """One mounted camera. The core fields mean the same thing on every backend.

    ``width``/``height``/``vertical_fov``/``near``/``far``/``data_types`` and the ``mount``
    are backend-agnostic; the optional ``isaacsim``/``isaacgym``/``mujoco`` sub-configs hold
    only the few intrinsics a given engine interprets in its own way.

    Keyed by its sensor name in the ``--sensor`` dict — the handle for
    ``get_camera_data(name, ...)`` and the observation-term parameter.
    """

    mount: SensorMountConfig
    """Body the camera is mounted on and follows, plus the fixed offset. Required (``target`` must
    be non-empty)."""

    width: int = 128
    """Rendered image width in pixels. Defaults to 128."""

    height: int = 128
    """Rendered image height in pixels. Defaults to 128."""

    vertical_fov: float = 45.0
    """Vertical field of view in degrees (the single shared FOV convention). Defaults to 45."""

    near: float = 0.01
    """Near clipping plane in meters. Defaults to 0.01.

    Per-camera on IsaacSim/IsaacGym. MuJoCo's clip is global (``model.vis.map.znear``), so across
    multiple cameras the shared range widens to ``min(near)``; a camera may then see nearer than
    its own value."""

    far: float = 1000.0
    """Far clipping plane in meters. Defaults to 1000.0.

    Per-camera on IsaacSim/IsaacGym; global on MuJoCo, where the shared range widens to ``max(far)``
    across all cameras. MuJoCo's far-clip also doubles as the depth no-hit boundary."""

    data_types: list[CameraDataType] = field(default_factory=lambda: ["rgb"])
    """Modalities to produce: ``"rgb"`` and/or ``"depth"``, e.g. ``["rgb", "depth"]``. The public
    ``get_camera_data(name, data_type)`` accessor is keyed by these."""

    update_decimation: DecimationLike = 1
    """Render every Nth control step (1 = every step). Int, or a frequency string ("20Hz") resolved
    against the control rate (fps/control_decimation) at the simulator. Lets slow cameras skip steps."""

    isaacsim: IsaacSimCameraConfig | None = None
    """IsaacSim-specific intrinsics overrides. Defaults to None (derive from core fields)."""

    isaacgym: IsaacGymCameraConfig | None = None
    """IsaacGym-specific intrinsics overrides. Defaults to None (derive from core fields)."""

    mujoco: MujocoCameraConfig | None = None
    """MuJoCo-specific intrinsics overrides. Defaults to None (derive from core fields)."""

    @model_validator(mode="after")
    def validate_camera(self) -> CameraSensorConfig:
        """Positive resolution, known data types, at least one modality, valid mount target."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"CameraSensorConfig needs positive width/height, got {self.width}x{self.height}.")
        if not self.data_types:
            raise ValueError("CameraSensorConfig must request at least one data_type.")
        validate_decimation_like(self.update_decimation, field="CameraSensorConfig update_decimation")
        # The robot is addressed via the ``robot_link`` kind; an ``actor`` mount must not name it.
        if self.mount.target_kind == "actor" and self.mount.target == "robot":
            raise ValueError(
                "Camera mounts on actor 'robot'; use target_kind='robot_link' "
                "(name the robot link, e.g. 'pelvis') for the robot."
            )
        return self


def validate_camera_dict(cameras: dict[str, CameraSensorConfig]) -> None:
    """Validate cross-camera constraints on a mounted-camera dict (keyed by sensor name).

    Individual ``CameraSensorConfig`` validation runs at construction; this covers only the checks
    that span multiple cameras. Call at the boundary that assembles the per-key ``--sensor`` dict
    (or any code that builds one directly) before handing it to a backend.

    Rejects conflicting MuJoCo-Warp appearance flags: ``use_shadows`` / ``use_textures`` /
    ``use_precomputed_rays`` are global to the shared Warp render context, so cameras that set a
    given flag must agree (``None`` imposes no constraint).
    """
    for attr in ("use_shadows", "use_textures", "use_precomputed_rays"):
        setters = {name: getattr(c.mujoco, attr) for name, c in cameras.items() if c.mujoco is not None}
        distinct = {v for v in setters.values() if v is not None}
        if len(distinct) > 1:
            conflicting = {n: v for n, v in setters.items() if v is not None}
            raise ValueError(
                f"MuJoCo-Warp render flag '{attr}' is GLOBAL to the shared render context but "
                f"cameras set conflicting values {conflicting}. All cameras that set it must agree "
                f"(or leave it None)."
            )
