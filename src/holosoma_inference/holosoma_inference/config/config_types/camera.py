"""Camera configuration types for holosoma_inference.

Depth-camera geometry and image properties for perception policies (e.g. the
depth-distillation stair policy). These describe the camera the policy was
*trained* against — the numbers must match training, because the resized
``(resized_height, resized_width)`` image is what the depth backbone consumes.

Distinct from :class:`~holosoma_inference.config.config_types.task.Ros2DepthConsumerConfig`,
which configures a ROS2 *transport* for depth. This type describes the camera
itself and is transport-agnostic.
"""

from __future__ import annotations

from dataclasses import field

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class CameraPose:
    """Mounting pose of a single camera relative to a parent link."""

    parent_link: str
    """Link the camera is rigidly attached to (e.g. ``"robot/torso_link"``)."""

    camera_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Translation (x, y, z) in meters relative to ``parent_link``."""

    camera_rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Rotation (roll, pitch, yaw) in degrees relative to ``parent_link``."""


@dataclass(frozen=True)
class CameraProps:
    """Image properties shared by all cameras in a :class:`CameraConfig`."""

    image_type: str = "depth"
    """Image modality (``"depth"`` or ``"rgb"``)."""

    width: int = 240
    """Native capture width in pixels."""

    height: int = 135
    """Native capture height in pixels."""

    resized_width: int = 87
    """Width after resize — must match the depth backbone's input width."""

    resized_height: int = 58
    """Height after resize — must match the depth backbone's input height."""

    horizontal_fov: float = 101.41
    """Horizontal field of view in degrees."""

    vertical_fov: float = 69.00
    """Vertical field of view in degrees."""

    near_clip: float = 0.1
    """Near clip in meters. Depth is clipped to [near_clip, far_clip]."""

    far_clip: float = 2.0
    """Far clip in meters. Depth is clipped to [near_clip, far_clip]."""

    frame_rate: int = 10
    """Publish rate in Hz on the producer side."""

    image_show: bool = False
    """Display captured frames for debugging."""

    depth_delay: int = 0
    """Modeled sensor delay in frames (total delay = ``depth_delay / frame_rate`` s)."""


@dataclass(frozen=True)
class CameraConfig:
    """A set of cameras (by name) sharing one set of image properties."""

    poses: dict[str, CameraPose] = field(default_factory=dict)
    """Camera name -> mounting pose. Iteration order is the depth stack order."""

    props: CameraProps = CameraProps()
    """Image properties shared by every camera in ``poses``."""

    @property
    def num_cameras(self) -> int:
        """Number of cameras in this configuration."""
        return len(self.poses)
