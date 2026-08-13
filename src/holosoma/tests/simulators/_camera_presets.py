"""Test-only scene and sensor presets for the camera assertion harness.

Scene presets are registered into the production ``scene:`` CLI menu at test time (so
``scene:camera-target`` resolves on the lazy tyro path). Sensor presets are plain camera dicts
(``dict[str, CameraSensorConfig]``, keyed by sensor name) injected directly into the parsed config
by the harness — they exercise camera *behavior*, not the per-key ``--sensor`` CLI (covered
elsewhere), and a multi-camera rig can't be named by a single CLI token.

Geometry: robot base frame is +X forward, +Z up. An identity mount orientation looks down the
optical axis (-Z). A -90 deg pitch about body Y rotates -Z to +X (look forward); quaternion
(w,x,y,z) = (cos45, 0, -sin45, 0).
"""

from __future__ import annotations

import math

from holosoma.config_types.scene import PhysicsConfig, RigidObjectConfig, SceneConfig
from holosoma.config_types.sensor import CameraSensorConfig, SensorMountConfig

_SMALL_BOX = "holosoma/data/scene_objects/boxes/small_box.urdf"
_SMALL_BOX_USD = "holosoma/data/scene_objects/boxes/small_box.usda"

# -90 deg about body +Y: rotates the camera's optical -Z (down) to point along body +X (forward).
_C = math.cos(-math.pi / 4)
_S = math.sin(-math.pi / 4)
_LOOK_FORWARD_WXYZ = [_C, 0.0, _S, 0.0]  # (w, x, y, z)

# Bright static box 0.5 m in front of the robot at camera height; fills the center of a
# forward-looking camera. urdf and usd for the respective backends.
_BASE_Z = 0.79  # g1 base/pelvis spawn height (init_state.pos z); camera mounts here.
camera_target = SceneConfig(
    rigid_objects={
        "target": RigidObjectConfig(
            urdf_file=_SMALL_BOX,
            usd_file=_SMALL_BOX_USD,
            position=[0.5, 0.0, _BASE_Z],
            fixed=True,
            physics=PhysicsConfig(),
        )
    }
)

# One camera on the robot base, pitched to look forward (+X), framing the target box at base
# height. Mount pushed 0.1 m forward to clear the robot torso.
front_cam = {
    "front_cam": CameraSensorConfig(
        mount=SensorMountConfig(
            target_kind="robot_link", target="pelvis", position=[0.1, 0.0, 0.0], orientation=_LOOK_FORWARD_WXYZ
        ),
        width=64,
        height=64,
        vertical_fov=60.0,
        data_types=["rgb"],
    )
}

# Forward camera producing both rgb and depth, used by the depth test.
front_cam_depth = {
    "front_cam": CameraSensorConfig(
        mount=SensorMountConfig(
            target_kind="robot_link", target="pelvis", position=[0.1, 0.0, 0.0], orientation=_LOOK_FORWARD_WXYZ
        ),
        width=64,
        height=64,
        vertical_fov=60.0,
        data_types=["rgb", "depth"],
    )
}

# Forward camera with a non-square resolution (width 96 != height 64) to catch a width/height
# transpose. width > height widens the horizontal FOV (from vertical_fov).
_WIDE_W = 96
_WIDE_H = 64
front_cam_wide = {
    "front_cam": CameraSensorConfig(
        mount=SensorMountConfig(
            target_kind="robot_link", target="pelvis", position=[0.1, 0.0, 0.0], orientation=_LOOK_FORWARD_WXYZ
        ),
        width=_WIDE_W,
        height=_WIDE_H,
        vertical_fov=60.0,
        data_types=["rgb"],
    )
}

# Two cameras at the same mount: fast_cam renders every control step (decimation=1), slow_cam every
# 3rd (decimation=3). Exercises the per-camera update_decimation gate.
slow_fast_cam = {
    "fast_cam": CameraSensorConfig(
        mount=SensorMountConfig(
            target_kind="robot_link", target="pelvis", position=[0.1, 0.0, 0.0], orientation=_LOOK_FORWARD_WXYZ
        ),
        width=64,
        height=64,
        vertical_fov=60.0,
        data_types=["rgb"],
        update_decimation=1,
    ),
    "slow_cam": CameraSensorConfig(
        mount=SensorMountConfig(
            target_kind="robot_link", target="pelvis", position=[0.1, 0.0, 0.0], orientation=_LOOK_FORWARD_WXYZ
        ),
        width=64,
        height=64,
        vertical_fov=60.0,
        data_types=["rgb"],
        update_decimation=3,
    ),
}

# Flat bright-red panel (0.4 m x 0.4 m, thin) facing the camera, centered on the forward axis at a
# known distance. The geometry test measures its silhouette (centroid, pixel extent vs FOV).
_PANEL = "holosoma/data/scene_objects/panels/red_panel.urdf"
_PANEL_XML = "holosoma/data/scene_objects/panels/red_panel.xml"
_PANEL_USD = "holosoma/data/scene_objects/panels/red_panel.usda"
_PANEL_HALF_SIZE = 0.2  # meters (half of the 0.4 m panel height/width)
_PANEL_DISTANCE = 1.0  # meters from the robot base (camera mounts ~0.1 m forward of base)

panel_target = SceneConfig(
    rigid_objects={
        "panel": RigidObjectConfig(
            urdf_file=_PANEL,
            xml_file=_PANEL_XML,
            usd_file=_PANEL_USD,
            position=[_PANEL_DISTANCE, 0.0, _BASE_Z],
            fixed=True,
            physics=PhysicsConfig(),
        )
    }
)

# Panel offset off the forward axis: to the robot's left (+Y) and up (+Z), so the red silhouette
# lands in one image quadrant. The orientation test uses this to catch a mirror, flip, or H/W
# transpose.
_PANEL_OFFSET_Y = 0.20  # meters to the robot's left (+Y body)
_PANEL_OFFSET_Z = 0.20  # meters up (+Z body)
panel_offaxis = SceneConfig(
    rigid_objects={
        "panel": RigidObjectConfig(
            urdf_file=_PANEL,
            xml_file=_PANEL_XML,
            usd_file=_PANEL_USD,
            position=[_PANEL_DISTANCE, _PANEL_OFFSET_Y, _BASE_Z + _PANEL_OFFSET_Z],
            fixed=True,
            physics=PhysicsConfig(),
        )
    }
)

# Two cameras on the robot base seeing different content: "front_cam" frames the panel ahead (+X),
# "down_cam" (identity mount) looks straight down. Used to assert per-camera attribution.
dual_cam = {
    "front_cam": CameraSensorConfig(
        mount=SensorMountConfig(
            target_kind="robot_link", target="pelvis", position=[0.1, 0.0, 0.0], orientation=_LOOK_FORWARD_WXYZ
        ),
        width=64,
        height=64,
        vertical_fov=60.0,
        data_types=["rgb"],
    ),
    "down_cam": CameraSensorConfig(
        mount=SensorMountConfig(target_kind="robot_link", target="pelvis", position=[0.1, 0.0, 0.0]),
        width=64,
        height=64,
        vertical_fov=60.0,
        data_types=["rgb"],
    ),
}

# One camera mounted on the scene object "panel" (target_kind="actor"), exercising the actor-mount
# resolver. Identity mount looks down the panel's local -Z.
actor_cam = {
    "panel_cam": CameraSensorConfig(
        mount=SensorMountConfig(target_kind="actor", target="panel", position=[0.0, 0.0, 0.0]),
        width=64,
        height=64,
        vertical_fov=60.0,
        data_types=["rgb"],
    )
}

# One FREE-FLOATING camera (target_kind="world", no body to follow), fixed in each env's frame at
# the robot's spawn spot, pitched to look forward (+X) at the panel. It anchors to the env frame,
# NOT the robot, so moving/yawing the robot must NOT change what it sees. Mirrors the `panel-target`
# scene geometry (panel 1.0 m ahead at base height); mounted like `front_cam` but on the world.
world_cam = {
    "world_cam": CameraSensorConfig(
        mount=SensorMountConfig(target_kind="world", position=[0.1, 0.0, _BASE_Z], orientation=_LOOK_FORWARD_WXYZ),
        width=64,
        height=64,
        vertical_fov=60.0,
        data_types=["rgb", "depth"],
    )
}

SCENE_PRESETS = {
    "camera-target": camera_target,
    "panel-target": panel_target,
    "panel-offaxis": panel_offaxis,
}
# Named camera-dict presets, for harnesses that take a ``--sensors`` NAME on the command line
# (e.g. camera_assert.py) and inject the resolved dict onto the config's ``sensor`` field.
SENSOR_PRESETS = {
    "front-cam": front_cam,
    "front-cam-depth": front_cam_depth,
    "front-cam-wide": front_cam_wide,
    "dual-cam": dual_cam,
    "actor-cam": actor_cam,
    "slow-fast-cam": slow_fast_cam,
    "world-cam": world_cam,
}


def register() -> None:
    """Merge the camera test SCENE presets into the production ``scene:`` CLI menu (idempotent).

    Sensor presets are NOT registered: they are plain camera dicts injected directly by the harness
    (see module docstring), so there is nothing to add to a CLI menu.
    """
    from holosoma.config_values import scene

    scene.SCENE_REGISTRY.update(SCENE_PRESETS)


def as_mjwarp(config):
    """Switch a MuJoCo ``RunSimConfig`` to the Warp backend (GPU, multi-env capable)."""
    import dataclasses

    from holosoma.config_types.simulator import MujocoBackend

    sim_cfg = dataclasses.replace(
        config.simulator,
        config=dataclasses.replace(config.simulator.config, mujoco_backend=MujocoBackend.WARP),
    )
    return dataclasses.replace(config, simulator=sim_cfg)


# Register on import so `import _camera_presets` installs the presets without an explicit call.
register()
