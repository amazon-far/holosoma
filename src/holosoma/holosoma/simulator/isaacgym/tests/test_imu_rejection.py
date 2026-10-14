# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""Isaac Gym rejects configured IMUs instead of silently creating none.

Preview 4 exposes no IMU or accelerometer, and a synthesized one would omit the centripetal term
the other two backends measure natively. ``create_sensors`` must therefore fail at startup rather
than leave the sensor absent, which would only surface later as ``has no IMU`` from a consumer.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.isaacgym

pytest.importorskip("isaacgym")

from holosoma.config_types.sensor import (  # noqa: E402
    CameraSensorConfig,
    ImuSensorConfig,
    LidarSensorConfig,
    SensorMountConfig,
)
from holosoma.simulator.isaacgym import sensor_setup  # noqa: E402

_MOUNT = SensorMountConfig(target_kind="robot_link", target="pelvis")


def _sim(sensor_config: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(sensor_config=sensor_config)


def test_configured_imu_is_rejected_by_name() -> None:
    sim = _sim({"pelvis_imu": ImuSensorConfig(mount=_MOUNT), "head": CameraSensorConfig(mount=_MOUNT)})
    with pytest.raises(NotImplementedError, match="Isaac Gym has no IMU sensor") as excinfo:
        sensor_setup.reject_imus(sim)  # type: ignore[arg-type]
    message = str(excinfo.value)
    assert "pelvis_imu" in message  # the error names the offending sensor key
    assert "MuJoCo or Isaac Sim" in message  # ...and where it does work


def test_camera_and_lidar_only_configs_are_untouched() -> None:
    sim = _sim({"head": CameraSensorConfig(mount=_MOUNT), "scan": LidarSensorConfig(mount=_MOUNT)})
    sensor_setup.reject_imus(sim)  # type: ignore[arg-type]


def test_empty_sensor_config_is_untouched() -> None:
    sensor_setup.reject_imus(_sim({}))  # type: ignore[arg-type]
