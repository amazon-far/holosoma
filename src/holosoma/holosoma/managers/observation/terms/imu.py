# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""IMU-sensor observation terms.

Read a mounted IMU's latest sample via the simulator's ``get_imu_data`` accessor. All three return
flat ``[num_envs, k]`` tensors, so unlike the camera terms they are legal in a ``concatenate=True``
policy group.

Values are in the sensor's mount frame — not the optical frame the camera and LiDAR terms use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from holosoma.envs.base_task.base_task import BaseTask


def imu_angular_velocity(env: BaseTask, sensor: str) -> torch.Tensor:
    """Body rates for one mounted IMU: ``[num_envs, 3]`` in rad/s, sensor frame.

    Parameters
    ----------
    env : BaseTask
        The task (provides ``env.simulator``).
    sensor : str
        IMU name (the ``--sensor`` dict key).
    """
    return env.simulator.get_imu_data(sensor, "angular_velocity")


def imu_linear_acceleration(env: BaseTask, sensor: str) -> torch.Tensor:
    """Specific force for one mounted IMU: ``[num_envs, 3]`` in m/s², sensor frame.

    Reads ``+9.81`` along the sensor's up axis at rest and zero in free fall, like a real
    accelerometer. See :func:`imu_angular_velocity` for parameters.
    """
    return env.simulator.get_imu_data(sensor, "linear_acceleration")


def imu_orientation(env: BaseTask, sensor: str) -> torch.Tensor:
    """World attitude of the sensor frame: ``[num_envs, 4]`` as ``xyzw``.

    This is ground truth from the simulator, not an estimate — a real IMU driver reports no
    absolute orientation, so a policy trained on it will not have the same input on hardware.
    See :func:`imu_angular_velocity` for parameters.
    """
    return env.simulator.get_imu_data(sensor, "orientation")
