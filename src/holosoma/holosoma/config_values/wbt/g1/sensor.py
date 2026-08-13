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
