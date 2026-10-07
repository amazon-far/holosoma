"""Default camera configurations for holosoma_inference.

These describe the camera each perception policy was *trained* with. The
resized dimensions in particular are part of the model's input contract — the
depth backbone's input tensor is exactly
``(resized_height, resized_width)``.
"""

from __future__ import annotations

from holosoma_inference.config.config_types.camera import CameraConfig, CameraPose, CameraProps
from holosoma_inference.utils.config_registry import ConfigRegistry

CAMERA_REGISTRY = ConfigRegistry(CameraConfig, group="holosoma.config.camera")

# Single forward-facing ZED 2i depth camera on the torso, pitched down to see
# the ground ahead — the setup used for the stair/rough-terrain policies.
single_zed2i_depth = CameraConfig(
    poses={
        "cam_front_depth": CameraPose(
            parent_link="robot/torso_link",
            camera_offset=(0.1, 0.0, 0.1),
            camera_rotation=(0.0, 75.0, 0.0),
        ),
    },
    props=CameraProps(
        image_type="depth",
        width=240,
        height=135,
        resized_width=87,
        resized_height=58,
        horizontal_fov=101.41,
        vertical_fov=69.00,
        near_clip=0.1,
        far_clip=2.0,
        frame_rate=10,
        image_show=False,
        depth_delay=0,
    ),
)

CAMERA_REGISTRY.add("single-zed2i-depth", single_zed2i_depth)

# Single forward-facing RealSense D435i depth camera on the torso, pitched 27deg down — the rig the
# D435i depth-distillation checkpoints were trained against (training-side ``G1FlatRsD435iConfig``).
#
# ``width``/``height`` are the training raycast resolution (106x60), and ``vertical_fov`` is derived
# from that aspect rather than a native 848x480 one so the projection matches training's K matrix:
#   f = (106/2) / tan(89.5deg/2);  vfov = 2*atan((60/2)/f) ~= 58.5953deg
# The clip range is the D435i config's [min_range, max_range] = [0.3, 3.0], which differs from the
# ZED rig's 2.0 far clip and must match the sim-side ``--plugin.<key>.far-clip``.
single_d435i_depth = CameraConfig(
    poses={
        "cam_front_depth": CameraPose(
            parent_link="robot/torso_link",
            camera_offset=(0.01, 0.01, 0.44),
            camera_rotation=(1.0, 27.0, 1.0),
        ),
    },
    props=CameraProps(
        image_type="depth",
        width=106,
        height=60,
        resized_width=87,
        resized_height=58,
        horizontal_fov=89.5,
        vertical_fov=58.5953,
        near_clip=0.3,
        far_clip=3.0,
        frame_rate=10,
        image_show=False,
        depth_delay=0,
    ),
)

CAMERA_REGISTRY.add("single-d435i-depth", single_d435i_depth)
