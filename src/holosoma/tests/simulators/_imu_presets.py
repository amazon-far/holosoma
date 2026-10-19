# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""Test-only IMU presets shared by the live backend harness.

Three of the four sit on the SAME body (``pelvis``) so their readings can be compared against each
other exactly: at rest a lever arm contributes nothing, so ``offset`` must equal ``identity``, and
``rotated`` must equal ``identity`` rotated by the mount quaternion. The fourth is on a different
link, purely to prove mount-target resolution works for something other than the root.
"""

from __future__ import annotations

import math

from holosoma.config_types.sensor import ImuSensorConfig, SensorMountConfig

# 90 deg about the mount's +X as [w, x, y, z]. Chosen because it moves gravity off the sensor's
# z-axis entirely, so a dropped or transposed mount rotation cannot pass by coincidence.
_ROT_90_X_WXYZ = [math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]

# Lever arm for the offset mount, in metres along the body's +X.
LEVER_ARM = 0.25

MOUNT_ROTATION_WXYZ = list(_ROT_90_X_WXYZ)

# Zero offset, identity rotation, on the root link: the sensor frame IS the base frame, so this one
# is directly comparable to robot_root_states.
identity_imu = ImuSensorConfig(mount=SensorMountConfig(target_kind="robot_link", target="pelvis"))

# Same body and orientation, translated along +X.
offset_imu = ImuSensorConfig(
    mount=SensorMountConfig(target_kind="robot_link", target="pelvis", position=[LEVER_ARM, 0.0, 0.0])
)

# Same body and origin, rotated.
rotated_imu = ImuSensorConfig(
    mount=SensorMountConfig(target_kind="robot_link", target="pelvis", orientation=list(_ROT_90_X_WXYZ))
)

# A non-root link, to prove mount-target resolution is not special-cased to the base.
torso_imu = ImuSensorConfig(mount=SensorMountConfig(target_kind="robot_link", target="torso_link"))

IMU_RIG = {
    "identity": identity_imu,
    "offset": offset_imu,
    "rotated": rotated_imu,
    "torso": torso_imu,
}
