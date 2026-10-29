"""Individual camera building blocks for the G1 robot.

Each value is one :class:`CameraSensorConfig`; compose them per-key on the CLI, e.g.
``--sensor.my_head:g1-head --sensor.my_left_wrist:g1-left-wrist``. Mounts on ``g1_29dof`` body
names and frames (see ``data/robots/g1/g1_29dof.xml``):

- Head: the ``torso_link`` frame origin is at the waist-pitch joint; the head and upper sensor
  mount sit at ``z ~= 0.41`` in that frame.
- Wrists: in the ``*_wrist_yaw_link`` frame the hand points along +X (``*_palm`` site at
  ``[0.08,0,0]``).

The mount orientations below reorient the camera's optical -Z to look along body +X.
"""

from __future__ import annotations

from holosoma.config_types.sensor import (
    CameraSensorConfig,
    IsaacSimCameraConfig,
    SensorMountConfig,
)

# Egocentric head camera (torso-frame z~=0.41), looking forward and upright.
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


_G1_MESHES = "holosoma/data/robots/g1/meshes"
# Links that can occlude the head camera, with their link-frame meshes. The pelvis contour shell is
# fixed to the pelvis frame; the wrists omit the rubber hands, which sit in a separate mesh.
_G1_LINK_MESHES: dict[str, str | list[str]] = {
    "pelvis": [f"{_G1_MESHES}/pelvis.STL", f"{_G1_MESHES}/pelvis_contour_link.STL"],
    **{
        f"{side}_{link}": f"{_G1_MESHES}/{side}_{link}.STL"
        for side in ("left", "right")
        for link in (
            "hip_pitch_link",
            "hip_roll_link",
            "hip_yaw_link",
            "knee_link",
            "ankle_pitch_link",
            "ankle_roll_link",
            "shoulder_pitch_link",
            "shoulder_roll_link",
            "shoulder_yaw_link",
            "elbow_link",
            "wrist_roll_link",
            "wrist_pitch_link",
            "wrist_yaw_link",
        )
    },
    "waist_yaw_link": f"{_G1_MESHES}/waist_yaw_link_rev_1_0.STL",
    "waist_roll_link": f"{_G1_MESHES}/waist_roll_link_rev_1_0.STL",
}

# RealSense D435i depth in the G1 head, as used for perceptive locomotion: 106x60 at 89.5deg
# horizontal FOV (58.6deg vertical), 0.44 m above the torso_link origin, looking along body +X
# pitched 27deg down. Isaac Sim ray casts it against the terrain and the robot links below; other
# backends render it natively. The opt-in ``g1_head_d435i_mount_setup_terms`` perturb its mount
# per env.
# ``near`` keeps returns closer than the D435i's 0.3 m working range so preprocessing can clamp
# them to it, as the hardware pipeline does.
g1_head_d435i_depth = CameraSensorConfig(
    mount=SensorMountConfig(
        target_kind="robot_link",
        target="torso_link",
        position=[0.01, 0.01, 0.44],
        orientation=[0.6029076421, 0.3758540132, -0.3629580015, -0.6029076421],
    ),
    width=106,
    height=60,
    vertical_fov=58.595269992389284,
    near=0.01,
    far=3.0,
    depth_clipping_behavior="max",
    data_types=["depth"],
    isaacsim=IsaacSimCameraConfig(renderer="raycast", raycast_link_meshes=_G1_LINK_MESHES),
)
