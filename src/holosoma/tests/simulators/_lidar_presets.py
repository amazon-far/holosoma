# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""Test-only LiDAR presets shared by the live backend harness."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable

from holosoma.config_types.sensor import (
    CustomLidarRayPatternConfig,
    IsaacSimLidarConfig,
    LidarBodyFilterConfig,
    LidarRangeClippingBehavior,
    LidarRayPatternConfig,
    LidarSensorConfig,
    MujocoLidarConfig,
    SensorMountConfig,
)
from holosoma.simulator.shared.lidar_sensor import CyclicLidarPatternProvider, LidarPatternProvider
from holosoma.utils.safe_torch_import import torch

ACTOR_MOUNT_HEIGHT = 1.21
ROBOT_LINK_MOUNT_HEIGHT = 1.2
ROBOT_FILTERED_MOUNT_HEIGHT = 0.0
ROLL_RADIANS = math.radians(30.0)
PLANE_NORMAL_SENSOR = [0.0, math.sin(ROLL_RADIANS), math.cos(ROLL_RADIANS)]
DOWNWARD_DIRECTIONS = [
    [0.0, 0.0, -1.0],
    [0.2, 0.0, -1.0],
    [-0.2, 0.0, -1.0],
    [0.0, 0.2, -1.0],
    [0.0, -0.2, -1.0],
    [1.0, 0.0, -0.1],  # Intersects beyond far and must be reported as a miss.
]
_ACTOR_MOUNT = SensorMountConfig(
    target_kind="actor",
    target="target",
    # camera-target's fixed actor is at [0.5, 0, 0.79], yielding sensor position [3, 0, 2].
    position=[2.5, 0.0, ACTOR_MOUNT_HEIGHT],
    orientation=[math.cos(ROLL_RADIANS / 2.0), math.sin(ROLL_RADIANS / 2.0), 0.0, 0.0],
)


def _downward_lidar(range_clipping_behavior: LidarRangeClippingBehavior) -> LidarSensorConfig:
    """A small deterministic pattern with one ray beyond its configured maximum range."""
    return LidarSensorConfig(
        mount=_ACTOR_MOUNT,
        pattern=CustomLidarRayPatternConfig(
            ray_directions=DOWNWARD_DIRECTIONS,
            organized_shape=[1, len(DOWNWARD_DIRECTIONS)],
        ),
        near=0.1,
        far=10.0,
        range_clipping_behavior=range_clipping_behavior,
    )


downward_lidar = _downward_lidar("none")
downward_max_lidar = _downward_lidar("max")
downward_zero_lidar = _downward_lidar("zero")
robot_link_lidar = LidarSensorConfig(
    mount=SensorMountConfig(
        target_kind="robot_link",
        target="pelvis",
        position=[2.5, 0.0, ROBOT_LINK_MOUNT_HEIGHT],
    ),
    pattern=CustomLidarRayPatternConfig(ray_directions=[[0.0, 0.0, -1.0]]),
    near=0.1,
    far=10.0,
)
robot_filtered_lidar = LidarSensorConfig(
    mount=SensorMountConfig(
        target_kind="robot_link",
        target="pelvis",
        position=[0.0, 0.0, ROBOT_FILTERED_MOUNT_HEIGHT],
    ),
    pattern=CustomLidarRayPatternConfig(ray_directions=[[0.0, 0.0, -1.0]]),
    near=0.1,
    far=10.0,
    body_filter=LidarBodyFilterConfig(target_kind="robot"),
)
# MuJoCo-only filter oracle: disabling every geom group must make every native ray miss. The
# live classic and Warp harnesses add this scanner, so both adapters prove they honor the config.
downward_filtered_lidar = LidarSensorConfig(
    mount=_ACTOR_MOUNT,
    pattern=CustomLidarRayPatternConfig(
        ray_directions=DOWNWARD_DIRECTIONS,
        organized_shape=[1, len(DOWNWARD_DIRECTIONS)],
    ),
    near=0.1,
    far=10.0,
    mujoco=MujocoLidarConfig(geom_groups=[False] * 6),
)


@dataclass(frozen=True)
class _CyclicTestPatternConfig(LidarRayPatternConfig):
    """Test-only time-varying pattern that keeps live backend coverage device-independent."""

    rays_per_snapshot: int

    def get_provider_cls(self) -> Callable[..., LidarPatternProvider]:
        return _CyclicTestPatternProvider


class _CyclicTestPatternProvider(CyclicLidarPatternProvider):
    """Small repeating sequence that points down in the test sensor's optical frame."""

    def __init__(
        self,
        config: _CyclicTestPatternConfig,
        *,
        device: torch.device | str,
        publish_hz: float,
    ) -> None:
        # IsaacLab represents USD Plane primitives with a 2e6 m float32 mesh. Keep this exact-plane
        # oracle near normal incidence while the separate cube oracle exercises arbitrary rays.
        azimuth = torch.linspace(-math.pi / 18.0, math.pi / 18.0, 128, device=device)
        sequence = torch.stack(
            [
                torch.sin(azimuth),
                torch.zeros_like(azimuth),
                -torch.cos(azimuth),
            ],
            dim=1,
        )
        super().__init__(
            sequence,
            height=1,
            width=config.rays_per_snapshot,
            publish_hz=publish_hz,
        )


# The generic cyclic fixture exercises runtime direction replacement while every sequence ray
# continues to face the same analytic plane as the static oracle.
CYCLIC_PLANE_NORMAL_SENSOR = PLANE_NORMAL_SENSOR
_CYCLIC_ACTOR_MOUNT = _ACTOR_MOUNT
cyclic_lidar = LidarSensorConfig(
    mount=_CYCLIC_ACTOR_MOUNT,
    pattern=_CyclicTestPatternConfig(rays_per_snapshot=64),
    near=0.1,
    far=10.0,
    update_decimation="10Hz",
    isaacsim=IsaacSimLidarConfig(mesh_prim_paths=["/World/ground"]),
)

# One-ray source for the mutable-direction moving-object oracle. Isaac Gym is excluded because
# its terrain-only compatibility caster cannot return scene objects.
CYCLIC_TARGET_MOUNT_POSITION = [2.5, -3.0, ACTOR_MOUNT_HEIGHT]
CYCLIC_TARGET_MOUNT_ORIENTATION_WXYZ = _ACTOR_MOUNT.orientation
cyclic_target_lidar = LidarSensorConfig(
    mount=SensorMountConfig(
        target_kind="actor",
        target="target",
        position=CYCLIC_TARGET_MOUNT_POSITION,
        orientation=CYCLIC_TARGET_MOUNT_ORIENTATION_WXYZ,
    ),
    pattern=_CyclicTestPatternConfig(rays_per_snapshot=1),
    near=0.1,
    far=5.0,
    update_decimation="10Hz",
    isaacsim=IsaacSimLidarConfig(mesh_prim_paths=["/World/envs/env_.*/scan_target/geom"]),
)

# MuJoCo and Isaac Sim use this world-fixed ray to verify that their native scene queries observe a
# kinematically moved target cube as well as the ground. Isaac Sim intentionally uses default mesh
# discovery here. Isaac Gym is excluded because its compatibility caster is terrain-only.
tracked_target_lidar = LidarSensorConfig(
    mount=SensorMountConfig(target_kind="world", position=[0.5, 0.0, 2.0]),
    pattern=CustomLidarRayPatternConfig(ray_directions=[[0.0, 0.0, -1.0]]),
    near=0.1,
    far=5.0,
)
tracked_target_actor_filtered_lidar = replace(
    tracked_target_lidar,
    body_filter=LidarBodyFilterConfig(target_kind="actor", target="target"),
)

# Isaac Sim visual-target discovery oracle. Both scanners share one downward ray from z=2 m.
# Default discovery must ignore the closer default-purpose CollisionAPI proxy under Collisions
# (top z=1.1) and return the visible, collision-enabled cube (top z=0.6). The explicit scanner
# proves that callers can still deliberately target the proxy and receive its closer return.
VISUAL_COLLISION_TARGET_POSITION = [3.0, 3.0, 0.0]
VISUAL_COLLISION_SENSOR_POSITION = [3.0, 3.0, 2.0]
VISUAL_TARGET_EXPECTED_RANGE = 1.4
EXPLICIT_COLLISION_EXPECTED_RANGE = 0.9
isaacsim_visual_target_lidar = LidarSensorConfig(
    mount=SensorMountConfig(target_kind="world", position=VISUAL_COLLISION_SENSOR_POSITION),
    pattern=CustomLidarRayPatternConfig(ray_directions=[[0.0, 0.0, -1.0]]),
    near=0.1,
    far=5.0,
)
isaacsim_explicit_collision_target_lidar = replace(
    isaacsim_visual_target_lidar,
    isaacsim=IsaacSimLidarConfig(mesh_prim_paths=["/World/envs/env_.*/visual_collision_target/Collisions/proxy"]),
)
isaacsim_ground_visual_lidar = LidarSensorConfig(
    mount=SensorMountConfig(target_kind="world", position=[5.0, 5.0, 2.0]),
    pattern=CustomLidarRayPatternConfig(ray_directions=[[0.0, 0.0, -1.0]]),
    near=0.1,
    far=5.0,
)

# This ray starts above the G1 and points through its opaque upper-body visuals toward the ground.
# A return substantially nearer than the known ground proves that default discovery retains
# legitimate robot visual self-hits.
ROBOT_VISUAL_SENSOR_HEIGHT = 2.5
isaacsim_robot_visual_lidar = LidarSensorConfig(
    mount=SensorMountConfig(target_kind="world", position=[0.0, 0.0, ROBOT_VISUAL_SENSOR_HEIGHT]),
    pattern=CustomLidarRayPatternConfig(ray_directions=[[0.0, 0.0, -1.0]]),
    near=0.05,
    far=5.0,
)

# MuJoCo-only selective filter oracle. In the test scene the target cube belongs to geom group 1
# while the ground belongs to group 0. This scanner must see the cube before it moves, then miss
# the ground after the move because it includes group 1 and excludes group 0.
tracked_target_group_one_lidar = LidarSensorConfig(
    mount=SensorMountConfig(target_kind="world", position=[0.5, 0.0, 2.0]),
    pattern=CustomLidarRayPatternConfig(ray_directions=[[0.0, 0.0, -1.0]]),
    near=0.1,
    far=5.0,
    mujoco=MujocoLidarConfig(geom_groups=[False, True, False, False, False, False]),
)

LIDAR_RIG = {
    "downward": downward_lidar,
    "downward_max": downward_max_lidar,
    "downward_zero": downward_zero_lidar,
    "robot_link": robot_link_lidar,
    "robot_filtered": robot_filtered_lidar,
    "cyclic": cyclic_lidar,
    "cyclic_target": cyclic_target_lidar,
}
