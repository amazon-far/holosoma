"""Frame conversion for H1 firmware exposing an uncompensated torso IMU."""

import numpy as np


def torso_imu_to_pelvis(
    quat_wxyz: np.ndarray,
    gyro_torso: np.ndarray,
    waist_yaw: float,
    waist_yaw_velocity: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove waist yaw from orientation and body-local angular velocity.

    Assumes the IMU axes align with torso_link and the waist rotates around
    pelvis Z, as in the H1 model. R_world_pelvis = R_world_torso R_z(-yaw).
    The gyro must also be rotated, then have the joint velocity subtracted.
    """
    w, x, y, z = quat_wxyz
    c, s = np.cos(waist_yaw / 2), -np.sin(waist_yaw / 2)
    quat_pelvis = np.array([w*c - z*s, x*c + y*s, y*c - x*s, z*c + w*s])
    c, s = np.cos(waist_yaw), np.sin(waist_yaw)
    gx, gy, gz = gyro_torso
    gyro_pelvis = np.array([c*gx - s*gy, s*gx + c*gy, gz - waist_yaw_velocity])
    return quat_pelvis, gyro_pelvis
