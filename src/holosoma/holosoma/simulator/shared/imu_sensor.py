# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""IMU output finalization shared by every backend's native IMU adapter.

MuJoCo's ``accelerometer`` / ``gyro`` / ``framequat`` triple and IsaacLab's ``Imu`` happen to agree:
both give the two rates in the sensor frame and the attitude as world-frame ``wxyz``. That is a
convenience, not a contract, so the output convention is pinned down here and nowhere else —
backend adapters hand over native values and hold no convention logic.

A function rather than an :class:`ImuRecord` ``set_buffer`` override (as ``CameraRecord`` uses for
depth clipping) because quaternion order fails silently: the ``orientation_wxyz`` keyword states
the expected input order at each call site, and one call means a backend cannot forget an output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from holosoma.utils.safe_torch_import import torch

if TYPE_CHECKING:
    from holosoma.simulator.shared.sensor_manager import ImuRecord


def finalize_imu_outputs(
    sensor_record: ImuRecord,
    *,
    linear_acceleration: torch.Tensor,
    angular_velocity: torch.Tensor,
    orientation_wxyz: torch.Tensor,
) -> None:
    """Store backend-neutral output buffers for one IMU sample.

    ``linear_acceleration`` and ``angular_velocity`` are ``[num_envs, 3]`` in the sensor frame:
    specific force in m/s² (``+9.81`` along the sensor's up axis at rest, zero in free fall) and
    body rates in rad/s. ``orientation_wxyz`` is ``[num_envs, 4]`` world attitude, buffered as
    ``xyzw`` to match ``rigid_body_quat_w``.

    Shapes are checked every sample because a backend slicing the wrong columns out of a packed
    native buffer yields plausible numbers.

    Inputs are always copied. A backend typically passes column slices of a live native buffer
    (MuJoCo's ``sensordata``), and those are already float32 on the Warp path — so casting alone
    would hand consumers a view that mutates under them on the next step.
    """
    for output, tensor, width in (
        ("linear_acceleration", linear_acceleration, 3),
        ("angular_velocity", angular_velocity, 3),
        ("orientation", orientation_wxyz, 4),
    ):
        if tensor.ndim != 2 or tensor.shape[-1] != width:
            raise ValueError(
                f"IMU '{sensor_record.name}' {output} must be [num_envs, {width}], got {tuple(tensor.shape)}."
            )

    sensor_record.set_buffer("linear_acceleration", linear_acceleration.to(torch.float32, copy=True))
    sensor_record.set_buffer("angular_velocity", angular_velocity.to(torch.float32, copy=True))
    # Advanced indexing already copies; [w,x,y,z] -> [x,y,z,w].
    sensor_record.set_buffer("orientation", orientation_wxyz[:, [1, 2, 3, 0]].to(torch.float32))
