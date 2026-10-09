# Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""Mounted camera and LiDAR configuration types.

A sensor is a read-only producer mounted onto an existing body (a robot link or a
spawned scene actor) that it follows.

Every mounted sensor uses the OpenGL/USD/MuJoCo optical frame: ``-Z`` forward and ``+Y`` up.
Isaac Gym converts this convention to its native sensor frame internally.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import field
from typing import TYPE_CHECKING, Callable, Literal, Union, cast, get_args

import tyro
from pydantic import ConfigDict, model_validator
from pydantic.dataclasses import dataclass

from holosoma.config_types.frequency import DecimationLike, validate_decimation_like
from holosoma.config_types.value_types import UnitQuaternionWXYZ

if TYPE_CHECKING:
    from holosoma.simulator.shared.lidar_sensor import LidarPatternProvider

# Reject unknown fields on every sensor config (a typo'd field fails at construction).
_FORBID_EXTRA = ConfigDict(extra="forbid")
_FORBID_EXTRA_WITH_PATTERN = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

# Camera data types. "rgb" and "depth" are implemented.
CameraDataType = Literal["rgb", "depth"]
CAMERA_DATA_TYPES = cast("tuple[CameraDataType, ...]", get_args(CameraDataType))

# Isaac Sim camera projections exposed by Isaac Lab's PinholeCameraCfg/FisheyeCameraCfg.
IsaacSimCameraProjectionType = Literal[
    "pinhole",
    "fisheyePolynomial",
    "fisheyeSpherical",
    "fisheyeKannalaBrandtK3",
    "fisheyeRadTanThinPrism",
    "omniDirectionalStereo",
]
ISAACSIM_FISHEYE_PROJECTION_TYPES = cast(
    "tuple[IsaacSimCameraProjectionType, ...]",
    tuple(value for value in get_args(IsaacSimCameraProjectionType) if value != "pinhole"),
)
_ISAACSIM_FISHEYE_POLYNOMIAL_FIELDS = (
    "polynomial_a",
    "polynomial_b",
    "polynomial_c",
    "polynomial_d",
    "polynomial_e",
    "polynomial_f",
)
ISAACSIM_FISHEYE_CONFIG_FIELDS = (
    "nominal_width",
    "nominal_height",
    "optical_centre_x",
    "optical_centre_y",
    "max_fov",
    *_ISAACSIM_FISHEYE_POLYNOMIAL_FIELDS,
)

# How a sensor represents a miss or a return outside its configured range.
NoReturnClippingBehavior = Literal["none", "max", "zero"]
DepthClippingBehavior = NoReturnClippingBehavior
LidarRangeClippingBehavior = NoReturnClippingBehavior

# MuJoCo exposes a six-entry mask; default LiDAR queries include authored visual groups 0-2.
DEFAULT_MUJOCO_LIDAR_GEOM_GROUPS = (True, True, True, False, False, False)

# Which body a sensor mounts on:
#   - "robot_link" resolves through the robot-only body index (find_rigid_body_indice).
#   - "actor" resolves a spawned scene/individual actor through the ObjectRegistry.
#   - "world" is a free-floating sensor fixed in each env's frame (no body to follow); ``target``
#     is unused. position/orientation are the pose in the per-env frame (env origin + offset), so
#     every env gets its own fixed sensor at the same relative spot.
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
class IsaacSimFisheyeConfig:
    r"""Native Isaac Sim f-theta lens calibration.

    ``nominal_width`` / ``nominal_height`` define the pixel coordinate system for the optical
    center and radial polynomial. The coefficients map nominal pixel radius :math:`r` to ray angle
    :math:`\theta` in radians:

    .. math::
        \theta(r) = a + br + cr^2 + dr^3 + er^4 + fr^5

    ``max_fov`` clips the rendered angular extent. The entire calibration is ignored when
    :attr:`IsaacSimCameraConfig.projection_type` is ``"pinhole"``.
    """

    nominal_width: float = 1936.0
    """F-theta calibration width in pixels. Defaults to Isaac Lab's 1936."""

    nominal_height: float = 1216.0
    """F-theta calibration height in pixels. Defaults to Isaac Lab's 1216."""

    optical_centre_x: float = 970.94244
    """F-theta optical center x-coordinate in nominal pixels. Defaults to Isaac Lab's 970.94244."""

    optical_centre_y: float = 600.37482
    """F-theta optical center y-coordinate in nominal pixels. Defaults to Isaac Lab's 600.37482."""

    max_fov: float = 200.0
    """Maximum fisheye field of view in degrees. Defaults to Isaac Lab's 200."""

    polynomial_a: float = 0.0
    """Constant native f-theta polynomial coefficient."""

    polynomial_b: float = 0.00245
    """Linear native f-theta polynomial coefficient. Defaults to Isaac Lab's 0.00245."""

    polynomial_c: float = 0.0
    """Quadratic native f-theta polynomial coefficient."""

    polynomial_d: float = 0.0
    """Cubic native f-theta polynomial coefficient."""

    polynomial_e: float = 0.0
    """Quartic native f-theta polynomial coefficient."""

    polynomial_f: float = 0.0
    """Quintic native f-theta polynomial coefficient."""

    @model_validator(mode="after")
    def validate_calibration(self) -> IsaacSimFisheyeConfig:
        """Validate the native f-theta calibration."""
        positive_fields = ("nominal_width", "nominal_height")
        for attr in positive_fields:
            value = getattr(self, attr)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"IsaacSimFisheyeConfig.{attr} must be finite and > 0, got {value}.")
        for attr in ("optical_centre_x", "optical_centre_y"):
            value = getattr(self, attr)
            if not math.isfinite(value):
                raise ValueError(f"IsaacSimFisheyeConfig.{attr} must be finite, got {value}.")
        if not math.isfinite(self.max_fov) or not 0.0 < self.max_fov <= 360.0:
            raise ValueError(f"IsaacSimFisheyeConfig.max_fov must be finite and in (0, 360], got {self.max_fov}.")
        for attr in _ISAACSIM_FISHEYE_POLYNOMIAL_FIELDS:
            value = getattr(self, attr)
            if not math.isfinite(value):
                raise ValueError(f"IsaacSimFisheyeConfig.{attr} must be finite, got {value}.")
        return self


@dataclass(frozen=True, config=_FORBID_EXTRA)
class IsaacSimCameraConfig:
    """Isaac Sim camera projection and lens settings.

    With ``projection_type="pinhole"``, :attr:`CameraSensorConfig.vertical_fov` determines the
    aperture. Native fisheye projections ignore ``CameraSensorConfig.vertical_fov`` and use
    :attr:`fisheye` for their angular calibration.
    """

    projection_type: IsaacSimCameraProjectionType = "pinhole"
    """Native USD/RTX camera projection. Defaults to ``"pinhole"``."""

    focal_length: float = 24.0
    """Lens focal length in Isaac Lab camera units (centimeters). For pinhole, the aperture pair is
    derived from it to hit ``CameraSensorConfig.vertical_fov``. Defaults to Isaac Lab's 24.0."""

    f_stop: float = 0.0
    """Aperture f-stop. ``0.0`` disables depth of field; ``>0`` enables defocus blur."""

    focus_distance: float = 400.0
    """Focus distance in meters (only meaningful when ``f_stop`` > 0). Defaults to Isaac Lab's 400."""

    fisheye: IsaacSimFisheyeConfig = field(default_factory=IsaacSimFisheyeConfig)
    """F-theta calibration used by native fisheye projections and ignored for ``"pinhole"``."""

    @model_validator(mode="after")
    def validate_lens(self) -> IsaacSimCameraConfig:
        """Validate native lens values."""
        for attr in ("focal_length", "focus_distance"):
            value = getattr(self, attr)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"IsaacSimCameraConfig.{attr} must be finite and > 0, got {value}.")
        if not math.isfinite(self.f_stop) or self.f_stop < 0.0:
            raise ValueError(f"IsaacSimCameraConfig.f_stop must be finite and >= 0, got {self.f_stop}.")
        return self


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
class LidarBodyFilterConfig:
    """Select robot geometry or one body to omit from LiDAR ray queries.

    Keep ``target_kind="mount"`` to prevent the body carrying the LiDAR from enclosing the
    ray origin, use ``target_kind="robot"`` to omit every robot-owned ray target, select a
    ``robot_link`` or registered scene ``actor`` explicitly, or use ``"none"`` to include every
    query target. Backends resolve a selected link to the native rigid body that owns it after
    import or compilation; fixed or fused links therefore follow their owning body. An ``actor``
    selects that registered actor's root body. The filter only affects ray queries and does not
    remove visual or collision geometry. Isaac Gym's terrain-only compatibility caster has no
    robot or body targets, so every filter kind is intentionally a no-op there.
    """

    target_kind: Literal["mount", "robot", "robot_link", "actor", "none"] = "mount"
    """How ``target`` is resolved. ``"mount"`` uses the sensor's mount body, ``"robot"`` omits
    every robot-owned target, ``"robot_link"`` resolves an unprefixed robot link, ``"actor"``
    resolves a registered scene actor, and ``"none"`` disables the filter."""

    target: str = ""
    """Body or actor name for ``"robot_link"`` and ``"actor"``. Leave empty for ``"mount"``
    ``"robot"``, and ``"none"``."""

    @model_validator(mode="after")
    def validate_target(self) -> LidarBodyFilterConfig:
        if self.target_kind in ("mount", "robot", "none"):
            if self.target:
                raise ValueError(
                    f"LidarBodyFilterConfig.target must be empty for "
                    f"target_kind='{self.target_kind}', got '{self.target}'."
                )
        elif not self.target or self.target != self.target.strip():
            raise ValueError(
                "LidarBodyFilterConfig.target must be a non-empty name without surrounding whitespace "
                f"for target_kind='{self.target_kind}'."
            )
        return self


@dataclass(frozen=True, config=_FORBID_EXTRA)
class MujocoLidarConfig:
    """MuJoCo geom-group selection shared by classic MuJoCo and MuJoCo Warp."""

    geom_groups: list[bool] = field(default_factory=lambda: list(DEFAULT_MUJOCO_LIDAR_GEOM_GROUPS))
    """Six inclusion flags for authored MuJoCo geom groups 0 through 5. Defaults to visual groups
    0 through 2 while excluding collision/debug groups 3 through 5; explicit masks may enable any
    group. Holosoma compiles robot-owned geometry into reserved visual and collision layers:
    enabling any group in 0-2 includes robot visuals, and enabling any group in 3-5 includes robot
    collision/debug geometry. ``body_filter.target_kind="robot"`` disables both reserved robot
    layers. Compiled groups 4 and 5 are reserved for robot geometry, so scene geoms cannot author
    either group. Applied identically to ``mj_multiRay`` and ``mujoco_warp.rays``."""

    @model_validator(mode="after")
    def validate_geom_groups(self) -> MujocoLidarConfig:
        if len(self.geom_groups) != 6:
            raise ValueError("MujocoLidarConfig.geom_groups must contain exactly six group flags.")
        return self


@dataclass(frozen=True, config=_FORBID_EXTRA)
class SensorMountConfig:
    """Where a sensor is mounted and its fixed offset (on that body, or in the per-env frame)."""

    target_kind: MountKind = "robot_link"
    """Which namespace ``target`` is resolved in (see :data:`MountKind`)."""

    target: str = ""
    """Body/actor name. Required and non-empty for ``robot_link`` (a robot link name; use the root
    link, e.g. ``"pelvis"``, to mount on the base) and ``actor`` (an ObjectRegistry actor name).
    Must be empty for ``world`` (a free-floating sensor anchors to no body)."""

    position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """Offset position ``[x, y, z]`` in meters. In the mount-body frame for ``robot_link``/``actor``;
    in the per-env frame (env origin + offset) for ``world``. Defaults to origin."""

    orientation: UnitQuaternionWXYZ = field(
        default_factory=lambda: [1.0, 0.0, 0.0, 0.0], metadata={"validate_default": True}
    )
    """Offset orientation quaternion ``[w, x, y, z]`` (w-first). In the mount-body frame for
    ``robot_link``/``actor``; in the per-env frame for ``world``. Identity uses the shared
    optical convention: ``-Z`` forward and ``+Y`` up. Must be unit length; norm deviations
    up to 1e-4 are normalized."""

    @model_validator(mode="after")
    def validate_mount(self) -> SensorMountConfig:
        """Validate the mount position and target contract."""
        if len(self.position) != 3:
            raise ValueError(f"SensorMountConfig.position must have 3 elements [x,y,z], got {self.position}.")
        if not all(math.isfinite(value) for value in self.position):
            raise ValueError(f"SensorMountConfig.position must contain only finite values, got {self.position}.")
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


class LidarRayPatternConfig(ABC):
    """Base class for a LiDAR ray pattern in the shared optical sensor frame.

    Choose :class:`GridLidarRayPatternConfig` for channel/azimuth scanners,
    or :class:`CustomLidarRayPatternConfig` for supplied directions. Extensions can add
    device-specific or time-varying patterns by subclassing this type. Each configuration selects
    its pattern provider lazily, keeping configuration serializable and independent of
    device-resident tensors and simulation-time state.
    """

    @abstractmethod
    def get_provider_cls(self) -> Callable[..., LidarPatternProvider]:
        """Return the pattern provider class for this ray layout.

        The sensor manager constructs this class with the pattern config, a target device, and the
        effective LiDAR publication rate. Each concrete implementation imports its provider locally
        so configuration loading stays independent of simulator dependencies.
        """
        raise NotImplementedError


@dataclass(frozen=True, config=_FORBID_EXTRA)
class GridLidarRayPatternConfig(LidarRayPatternConfig):
    """A uniform or explicitly sampled azimuth/elevation LiDAR pattern.

    Directions use ``-Z`` as forward, ``+Y`` as up, and ``+X`` as right. Zero azimuth points
    forward; positive azimuth rotates toward the sensor's left (``-X``); positive elevation rotates
    toward ``+Y``. Each axis may be generated from a field of view and resolution or supplied as
    explicit angles.
    """

    horizontal_fov: list[float] = field(default_factory=lambda: [-180.0, 180.0])
    """Generated azimuth range ``[min_deg, max_deg]``. The upper endpoint is excluded."""

    vertical_fov: list[float] = field(default_factory=lambda: [0.0, 0.0])
    """Generated elevation range ``[min_deg, max_deg]``. The upper endpoint is excluded.

    Equal endpoints produce one row at that angle.
    """

    horizontal_resolution: float = 1.0
    """Uniform azimuth spacing in degrees. Ignored when ``horizontal_angles`` is provided."""

    vertical_resolution: float = 1.0
    """Uniform elevation spacing in degrees. Ignored when ``vertical_angles`` is provided."""

    horizontal_angles: list[float] | None = None
    """Explicit azimuth angles in degrees. ``None`` uses ``horizontal_fov`` and ``horizontal_resolution``."""

    vertical_angles: list[float] | None = None
    """Explicit elevation angles in degrees. ``None`` uses ``vertical_fov`` and ``vertical_resolution``."""

    @model_validator(mode="after")
    def validate_pattern(self) -> GridLidarRayPatternConfig:
        for axis in ("horizontal", "vertical"):
            fov = getattr(self, f"{axis}_fov")
            resolution = getattr(self, f"{axis}_resolution")
            angles = getattr(self, f"{axis}_angles")
            if len(fov) != 2 or not all(math.isfinite(value) for value in fov):
                raise ValueError(f"GridLidarRayPatternConfig.{axis}_fov must contain two finite angles.")
            span = fov[1] - fov[0]
            if span < 0.0 or span > 360.0:
                raise ValueError(
                    f"GridLidarRayPatternConfig.{axis}_fov must have a span in [0, 360] degrees, got {fov}."
                )
            if not math.isfinite(resolution) or resolution <= 0.0:
                raise ValueError(f"GridLidarRayPatternConfig.{axis}_resolution must be finite and > 0.")
            if angles is not None and (not angles or not all(math.isfinite(value) for value in angles)):
                raise ValueError(f"GridLidarRayPatternConfig.{axis}_angles must be None or a non-empty finite list.")
        return self

    def get_provider_cls(self) -> Callable[..., LidarPatternProvider]:
        from holosoma.simulator.shared.lidar_sensor import GridLidarPatternProvider

        return GridLidarPatternProvider


@dataclass(frozen=True, config=_FORBID_EXTRA)
class CustomLidarRayPatternConfig(LidarRayPatternConfig):
    """An ordered list of user-supplied ray directions in the shared optical sensor frame."""

    ray_directions: tyro.conf.UsePythonSyntaxForLiteralCollections[list[list[float]]]
    """Non-zero ``[x, y, z]`` direction vectors. They are normalized when the sensor is created.
    From the CLI, pass the complete pattern as one Python literal, for example
    ``--sensor.scan.pattern.ray-directions '[[-1, 0, -3], [4, 5, -6]]'``."""

    organized_shape: list[int] | None = None
    """Optional ``[height, width]`` whose product equals the number of directions. ``None`` is one row."""

    @model_validator(mode="after")
    def validate_pattern(self) -> CustomLidarRayPatternConfig:
        if not self.ray_directions:
            raise ValueError("CustomLidarRayPatternConfig.ray_directions must not be empty.")
        for direction in self.ray_directions:
            if len(direction) != 3 or not all(math.isfinite(value) for value in direction):
                raise ValueError("CustomLidarRayPatternConfig.ray_directions entries must be finite [x, y, z] vectors.")
            if math.sqrt(sum(value * value for value in direction)) < 1e-8:
                raise ValueError("CustomLidarRayPatternConfig.ray_directions must not contain a zero vector.")
        if self.organized_shape is not None and (
            len(self.organized_shape) != 2
            or self.organized_shape[0] <= 0
            or self.organized_shape[1] <= 0
            or self.organized_shape[0] * self.organized_shape[1] != len(self.ray_directions)
        ):
            raise ValueError(
                "CustomLidarRayPatternConfig.organized_shape must be positive [height, width] "
                f"whose product is {len(self.ray_directions)}, got {self.organized_shape}."
            )
        return self

    def get_provider_cls(self) -> Callable[..., LidarPatternProvider]:
        from holosoma.simulator.shared.lidar_sensor import CustomLidarPatternProvider

        return CustomLidarPatternProvider


@dataclass(frozen=True, config=_FORBID_EXTRA)
class IsaacSimLidarConfig:
    """IsaacLab MultiMeshRayCaster options.

    Explicit expressions identify USD prims or prim-path expressions. The ray caster can include
    arbitrary supported geometry and tracks non-ground transforms as the scene changes.
    """

    mesh_prim_paths: list[str] | None = None
    """USD prim paths or expressions to ray cast against.

    ``None`` automatically includes visible ``default``/``render``-purpose geometry from the global
    ground, every spawned scene object, and the robot, then applies
    :attr:`LidarSensorConfig.body_filter`. Default discovery omits ``proxy``/``guide`` purpose and
    inherited-invisible geometry. USD material opacity is not interpreted, so transparent helper
    geometry must instead be hidden, assigned non-render purpose, or removed from the asset. A
    non-empty list is authoritative in Isaac Sim and bypasses discovery and ``body_filter``
    unchanged. Other backends still apply the shared body filter normally. Each expression must
    identify at most one target below each native rigid-body owner per environment, as required by
    IsaacLab 2.3.2 transform tracking; list exact mesh paths when siblings share one owner.
    """

    @model_validator(mode="after")
    def validate_mesh_paths(self) -> IsaacSimLidarConfig:
        if self.mesh_prim_paths is not None and (
            not self.mesh_prim_paths or any(not path.strip() for path in self.mesh_prim_paths)
        ):
            raise ValueError("IsaacSimLidarConfig.mesh_prim_paths must be None or a non-empty list of paths.")
        return self


@dataclass(frozen=True, config=_FORBID_EXTRA)
class CameraSensorConfig:
    """One mounted camera. The core fields mean the same thing on every backend.

    ``width``/``height``/``vertical_fov``/``near``/``far``/``data_types`` and the ``mount``
    are backend-agnostic. The ``isaacsim``/``isaacgym``/``mujoco`` sub-configs expose additional
    engine-specific controls. All backends use a pinhole projection unless the Isaac Sim config
    selects one of its native fisheye models.

    Keyed by its sensor name in the ``--sensor`` dict — the handle for
    ``get_camera_data(name, ...)`` and the observation-term parameter.
    """

    mount: SensorMountConfig
    """Body or world frame the camera follows, plus its fixed optical-frame offset."""

    width: int = 128
    """Rendered image width in pixels. Defaults to 128."""

    height: int = 128
    """Rendered image height in pixels. Defaults to 128."""

    vertical_fov: float = 45.0
    """Pinhole vertical field of view in degrees. Defaults to 45. Used on every backend except
    when Isaac Sim selects a native fisheye projection."""

    near: float = 0.01
    """Near clipping plane in meters. Defaults to 0.01.

    Per-camera on IsaacSim/IsaacGym. MuJoCo's clip is global (``model.vis.map.znear``), so across
    multiple cameras the shared range widens to ``min(near)``; a camera may then see nearer than
    its own value."""

    far: float = 1000.0
    """Far clipping plane in meters. Defaults to 1000.0.

    Per-camera on IsaacSim/IsaacGym; global on MuJoCo, where the shared range widens to ``max(far)``
    across all cameras. MuJoCo's far-clip also doubles as the depth no-hit boundary."""

    depth_clipping_behavior: DepthClippingBehavior = "none"
    """How depth-camera misses and returns outside ``[near, far]`` are reported on every backend:

    - ``"none"`` keeps ``+inf`` for no return.
    - ``"max"`` emits ``far``.
    - ``"zero"`` emits ``0.0``.

    Applies only when ``data_types`` includes ``"depth"``. Backends first provide raw metric depth;
    Holosoma then applies this policy consistently before consumers read the frame."""

    data_types: list[CameraDataType] = field(default_factory=lambda: ["rgb"])
    """Modalities to produce: ``"rgb"`` and/or ``"depth"``, e.g. ``["rgb", "depth"]``. The public
    ``get_camera_data(name, data_type)`` accessor is keyed by these."""

    update_decimation: DecimationLike = 1
    """Render every Nth control step (1 = every step). Int, or a frequency string ("20Hz") resolved
    against the control rate (fps/control_decimation) at the simulator. Lets slow cameras skip steps."""

    isaacsim: IsaacSimCameraConfig = field(default_factory=IsaacSimCameraConfig)
    """Isaac Sim projection and lens settings. Defaults to the pinhole model."""

    isaacgym: IsaacGymCameraConfig = field(default_factory=IsaacGymCameraConfig)
    """Isaac Gym camera settings. Unset fields retain native defaults."""

    mujoco: MujocoCameraConfig = field(default_factory=MujocoCameraConfig)
    """MuJoCo camera settings. Unset fields retain native defaults."""

    @model_validator(mode="after")
    def validate_camera(self) -> CameraSensorConfig:
        """Validate resolution, clipping range, modalities, cadence, and mount target."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"CameraSensorConfig needs positive width/height, got {self.width}x{self.height}.")
        if not math.isfinite(self.vertical_fov) or not 0.0 < self.vertical_fov < 180.0:
            raise ValueError(
                f"CameraSensorConfig.vertical_fov must be finite and in (0, 180), got {self.vertical_fov}."
            )
        if not math.isfinite(self.near) or self.near < 0.0:
            raise ValueError(f"CameraSensorConfig.near must be finite and >= 0, got {self.near}.")
        if not math.isfinite(self.far) or self.far <= self.near:
            raise ValueError(f"CameraSensorConfig.far must be finite and > near, got {self.near}..{self.far}.")
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


@dataclass(frozen=True, config=_FORBID_EXTRA_WITH_PATTERN)
class LidarSensorConfig:
    """One mounted ray-cast LiDAR. The core fields mean the same thing on every backend.

    ``mount``/``pattern``/``near``/``far``/``range_clipping_behavior``/``body_filter`` and
    ``update_decimation`` are backend-agnostic. The optional ``isaacsim`` and ``mujoco``
    sub-configs hold query options that only those engines interpret. Point outputs use the shared
    optical sensor frame: ``-Z`` forward and ``+Y`` up.

    Keyed by its sensor name in the ``--sensor`` dict, which is also the handle for
    ``get_lidar_data(name, ...)`` and LiDAR plugin routes.
    """

    mount: SensorMountConfig
    """Body or world frame the LiDAR follows, plus its fixed optical-frame offset."""

    body_filter: LidarBodyFilterConfig = field(default_factory=LidarBodyFilterConfig)
    """Robot geometry or one body omitted from scene queries on supporting backends. Defaults to
    the body carrying the LiDAR; use ``target_kind="none"`` to include every target, or select one
    robot link or scene actor."""

    pattern: LidarRayPatternConfig = field(default_factory=GridLidarRayPatternConfig)
    """Ray layout and time behavior for every measurement."""

    near: float = 0.05
    """Near clipping distance in meters. Shorter returns use the configured no-return behavior."""

    far: float = 100.0
    """Far clipping distance in meters. Longer returns use the configured no-return behavior."""

    range_clipping_behavior: LidarRangeClippingBehavior = "none"
    """How no-return rays are reported consistently on every backend:

    - ``"none"`` keeps ``+inf`` ranges and XYZ NaNs.
    - ``"max"`` emits ``far`` and a point at the ray endpoint.
    - ``"zero"`` emits a zero range and a point at the sensor origin.
    """
    update_decimation: DecimationLike = 1
    """Capture every Nth control step, or use a frequency string such as ``"10Hz"``."""

    isaacsim: IsaacSimLidarConfig | None = None
    """Isaac Sim mesh-query options. ``None`` uses :class:`IsaacSimLidarConfig` defaults."""

    mujoco: MujocoLidarConfig | None = None
    """MuJoCo classic and Warp geom-group selection. ``None`` includes visual groups 0 through 2."""

    @model_validator(mode="after")
    def validate_lidar(self) -> LidarSensorConfig:
        if not math.isfinite(self.near) or self.near < 0.0:
            raise ValueError(f"LidarSensorConfig.near must be finite and >= 0, got {self.near}.")
        if not math.isfinite(self.far) or self.far <= self.near:
            raise ValueError(f"LidarSensorConfig.far must be finite and > near, got {self.near}..{self.far}.")
        validate_decimation_like(self.update_decimation, field="LidarSensorConfig update_decimation")
        if self.mount.target_kind == "actor" and self.mount.target == "robot":
            raise ValueError(
                "LiDAR mounts on actor 'robot'; use target_kind='robot_link' "
                "(name the robot link, e.g. 'pelvis') for the robot."
            )
        return self


SensorConfig = Union[CameraSensorConfig, LidarSensorConfig]


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
        setters = {name: getattr(c.mujoco, attr) for name, c in cameras.items()}
        distinct = {v for v in setters.values() if v is not None}
        if len(distinct) > 1:
            conflicting = {n: v for n, v in setters.items() if v is not None}
            raise ValueError(
                f"MuJoCo-Warp render flag '{attr}' is GLOBAL to the shared render context but "
                f"cameras set conflicting values {conflicting}. All cameras that set it must agree "
                f"(or leave it None)."
            )
