"""Individual camera building blocks for the G1 robot.

Each value is one :class:`CameraSensorConfig`; compose them per-key on the CLI, e.g.
``--sensor.my_head:g1-head --sensor.my_left_wrist:g1-left-wrist``. Mounts on ``g1_29dof`` body
names and frames (see ``data/robots/g1/g1_29dof.xml``):

- Head: the ``torso_link`` frame origin is at the waist-pitch joint; the head and ``mid360``
  Livox site sit at ``z ~= 0.41`` in that frame.
- Wrists: in the ``*_wrist_yaw_link`` frame the hand points along +X (``*_palm`` site at
  ``[0.08,0,0]``).

The mount orientations below reorient the camera's optical -Z to look along body +X.
"""

from __future__ import annotations

from holosoma.config_types.sensor import CameraSensorConfig, SensorMountConfig

# Egocentric head camera (torso-frame z~=0.41, by the mid360 mount), looking forward and upright.
head_camera = CameraSensorConfig(
    mount=SensorMountConfig(
        target_kind="robot_link",
        target="torso_link",
        position=[0.05, 0.0, 0.41],
        # forward = body +X, up = body +Z (level, forward-facing)
        orientation=[0.5, 0.5, -0.5, -0.5],
    ),
    width=128,
    height=128,
    data_types=["rgb"],
    update_decimation="50Hz",
)


def _stereo_head_camera(sign: int) -> CameraSensorConfig:
    """One eye of a stereo head pair, offset along body +Y by half the interpupillary distance."""
    return CameraSensorConfig(
        mount=SensorMountConfig(
            target_kind="robot_link",
            target="torso_link",
            # Default eye separation (meters) along body +Y (human interpupillary distance) is 0.063
            position=[0.05, sign * 0.063 / 2, 0.41],
            # forward = body +X, up = body +Z (level, forward-facing)
            orientation=[0.5, 0.5, -0.5, -0.5],
        ),
        width=128,
        height=128,
        vertical_fov=45,
        data_types=["rgb"],
        update_decimation="50Hz",
    )


stereo_head_camera_left = _stereo_head_camera(+1)
stereo_head_camera_right = _stereo_head_camera(-1)


def _wrist_camera(side: str) -> CameraSensorConfig:
    """One grasp camera per wrist: offset up off the back of the wrist (+Z, clear of the hand)
    and pitched down to watch the palm/grasp region ahead."""
    return CameraSensorConfig(
        mount=SensorMountConfig(
            target_kind="robot_link",
            target=f"{side}_wrist_yaw_link",
            position=[0.03, 0.0, 0.06],
            # forward = +X pitched 30deg toward -Z, up near +Z,
            orientation=[0.61237244, 0.35355339, -0.35355339, -0.61237244],
        ),
        width=128,
        height=128,
        near=0.02,
        data_types=["rgb"],
        update_decimation="50Hz",
    )


left_wrist_camera = _wrist_camera("left")
right_wrist_camera = _wrist_camera("right")

# Waist-height depth cameras (torso_link frame origin sits at the waist-pitch joint, z~=0 => waist
# height). Depth-only, for near-body obstacle sensing / whole-body awareness fore and aft.

# forward = body +X pitched 70deg down toward -Z, up in the +X/+Z plane; just ahead of the torso,
# looking down at the ground ahead of the feet.
waist_front_camera = CameraSensorConfig(
    mount=SensorMountConfig(
        target_kind="robot_link",
        target="torso_link",
        position=[0.1, 0.0, 0.0],
        orientation=[0.69636424, 0.1227878, -0.1227878, -0.69636424],
    ),
    width=128,
    height=128,
    data_types=["depth"],
    update_decimation="50Hz",
)

# forward = body -X, up = body +Z (level), just behind the torso (180deg yaw of the front).
waist_back_camera = CameraSensorConfig(
    mount=SensorMountConfig(
        target_kind="robot_link",
        target="torso_link",
        position=[-0.1, 0.0, 0.0],
        orientation=[0.5, 0.5, 0.5, 0.5],
    ),
    width=128,
    height=128,
    data_types=["depth"],
    update_decimation="50Hz",
)

# Forward-facing depth camera pitched 71deg down, mounted on the torso — the rig the
# depth-distillation stair policies were trained against. Renders at 240x135 (16:9,
# matching the training raycast resolution); the depth-shm plugin resizes to the
# backbone's 58x87 input. The orientation places the optical axis (-Z) along body +X
# rotated 71deg toward -Z, with image-up perpendicular to it in the X-Z plane, so the
# camera sees the ground a short distance ahead of the feet.
stair_front_depth_camera = CameraSensorConfig(
    mount=SensorMountConfig(
        target_kind="robot_link",
        target="torso_link",
        position=[0.125, 0.06, 0.02],
        orientation=[-0.69740924, -0.11670628, 0.11670628, 0.69740924],
    ),
    width=240,
    height=135,
    vertical_fov=69.0,
    data_types=["depth"],
    # ">50Hz" rather than "50Hz": the rate resolves against the CONTROL rate, and the sim2sim
    # scripts run mujoco at 500Hz/4 = 125Hz, where a bare "50Hz" is not exactly achievable and raises.
    update_decimation=">50Hz",
)

# Forward-facing RealSense D435i depth camera pitched 27deg down on the torso — the rig the
# *D435i* depth-distillation checkpoints were trained against (training-side
# `G1FlatRsD435iConfig`: offset_pos=(0.01, 0.01, 0.44), offset_rot=(1.0, 27.0, 1.0) roll/pitch/yaw
# deg, 106x60, hfov 89.5deg, range [0.3, 3.0]m).
#
# The orientation is that training (roll, pitch, yaw) triple converted into MuJoCo's camera frame:
# warp's camera looks along +Z in its data frame and reaches the sensor frame via
# offset_rot_base = (-90, 0, -90), while MuJoCo's camera looks along -Z, so
# ``q = q_user * q_base * Rx(180deg)`` (all wxyz).
#
# vertical_fov is deliberately derived from the 106x60 *raycast* aspect rather than the render
# aspect, so the projection matches training's K matrix:
#   f = (106/2) / tan(89.5deg/2);  vfov = 2*atan((60/2)/f) ~= 58.5953deg
# Rendering at 106x60 keeps that consistent and lets the depth-shm plugin apply training's
# [2:, 4:-4] crop before resizing to the backbone's 58x87 input.
d435i_front_depth_camera = CameraSensorConfig(
    mount=SensorMountConfig(
        target_kind="robot_link",
        target="torso_link",
        position=[0.01, 0.01, 0.44],
        orientation=[0.60290764, 0.37585401, -0.362958, -0.60290764],
    ),
    width=106,
    height=60,
    vertical_fov=58.5953,
    # NOT near=0.3 / far=3.0, even though that is the training clamp range: MuJoCo's clip is a
    # GLOBAL frustum (``model.vis.map.{znear,zfar}``) shared with the viewer, and it *removes*
    # geometry rather than saturating it. A near plane at 0.3 would make an obstacle at 0.2m
    # invisible and let the camera see the background behind it, whereas training clamps such a hit
    # to 0.3 (a very close surface). So keep a permissive frustum and let the depth-shm plugin do
    # the [near_clip, far_clip] clamp — that reproduces training's ``torch.clamp`` exactly.
    near=0.1,
    data_types=["depth"],
    update_decimation=">50Hz",
)
