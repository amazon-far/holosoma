"""Unit tests for camera-config validators (pure, no simulator).

Per-camera checks run at ``CameraSensorConfig`` construction; the cross-camera Warp render-flag
check is :func:`validate_camera_dict`, called at the CLI boundary that assembles the ``--sensor``
dict.
"""

from __future__ import annotations

import pytest

from holosoma.config_types.sensor import (
    CameraSensorConfig,
    MujocoCameraConfig,
    SensorMountConfig,
    validate_camera_dict,
)

pytestmark = pytest.mark.no_sim


def _cam(*, target_kind="robot_link", target="pelvis", mujoco=None):
    return CameraSensorConfig(
        mount=SensorMountConfig(target_kind=target_kind, target=target),
        data_types=["rgb"],
        mujoco=mujoco,
    )


def test_actor_mount_named_robot_rejected():
    # The robot is addressed via target_kind="robot_link", never as an actor named "robot".
    with pytest.raises(ValueError, match="use target_kind='robot_link'"):
        _cam(target_kind="actor", target="robot")


def test_actor_mount_other_name_allowed():
    _cam(target_kind="actor", target="panel")  # must not raise


def test_conflicting_warp_render_flag_rejected():
    # use_shadows is global to the shared Warp render context; cameras setting it differently are
    # rejected.
    a = _cam(mujoco=MujocoCameraConfig(use_shadows=True))
    b = _cam(mujoco=MujocoCameraConfig(use_shadows=False))
    with pytest.raises(ValueError, match="render flag 'use_shadows'"):
        validate_camera_dict({"a": a, "b": b})


def test_agreeing_warp_render_flag_allowed():
    a = _cam(mujoco=MujocoCameraConfig(use_shadows=True))
    b = _cam(mujoco=MujocoCameraConfig(use_shadows=True))
    validate_camera_dict({"a": a, "b": b})  # agreement is fine


def test_none_warp_render_flag_imposes_no_constraint():
    # One camera sets the flag, the other leaves it None: no conflict.
    a = _cam(mujoco=MujocoCameraConfig(use_textures=False))
    b = _cam()  # mujoco=None
    validate_camera_dict({"a": a, "b": b})  # must not raise


# The camera-frame sink fields (cameras/modalities/record_video/...) live on CameraVizPluginConfig;
# their validation is covered in config_types/tests/test_plugin_egress_config.py.
