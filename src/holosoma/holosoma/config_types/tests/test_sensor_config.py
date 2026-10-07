# Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""Unit tests for camera-config validators (pure, no simulator).

Per-camera checks run at ``CameraSensorConfig`` construction; the cross-camera Warp render-flag
check is :func:`validate_camera_dict`, called at the CLI boundary that assembles the ``--sensor``
dict.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from holosoma.config_types.sensor import (
    ISAACSIM_FISHEYE_PROJECTION_TYPES,
    CameraSensorConfig,
    CustomLidarRayPatternConfig,
    GridLidarRayPatternConfig,
    IsaacGymCameraConfig,
    IsaacSimCameraConfig,
    IsaacSimFisheyeConfig,
    IsaacSimLidarConfig,
    LidarBodyFilterConfig,
    LidarRayPatternConfig,
    LidarSensorConfig,
    MountKind,
    MujocoCameraConfig,
    MujocoLidarConfig,
    SensorMountConfig,
    validate_camera_dict,
)
from holosoma.simulator.shared.lidar_sensor import (
    CustomLidarPatternProvider,
    GridLidarPatternProvider,
)

pytestmark = pytest.mark.no_sim


def _cam(
    *,
    target_kind: MountKind = "robot_link",
    target: str = "pelvis",
    mujoco: MujocoCameraConfig | None = None,
) -> CameraSensorConfig:
    backend_config: dict[str, Any] = {} if mujoco is None else {"mujoco": mujoco}
    return CameraSensorConfig(
        mount=SensorMountConfig(target_kind=target_kind, target=target),
        data_types=["rgb"],
        **backend_config,
    )


def test_actor_mount_named_robot_rejected() -> None:
    # The robot is addressed via target_kind="robot_link", never as an actor named "robot".
    with pytest.raises(ValueError, match="use target_kind='robot_link'"):
        _cam(target_kind="actor", target="robot")


def test_actor_mount_other_name_allowed() -> None:
    _cam(target_kind="actor", target="panel")  # must not raise


def test_mount_transform_is_finite_and_quaternion_is_normalized() -> None:
    mount = SensorMountConfig(
        target_kind="robot_link",
        target="pelvis",
        position=[1.0, 2.0, 3.0],
        orientation=[1.000001, 0.0, 0.0, 0.0],
    )
    assert mount.orientation == [1.0, 0.0, 0.0, 0.0]

    with pytest.raises(ValueError, match="position must contain only finite"):
        SensorMountConfig(target_kind="robot_link", target="pelvis", position=[math.inf, 0.0, 0.0])
    with pytest.raises(ValueError, match=r"(?s)orientation.*only finite"):
        SensorMountConfig(target_kind="robot_link", target="pelvis", orientation=[math.nan, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="non-zero quaternion"):
        SensorMountConfig(target_kind="robot_link", target="pelvis", orientation=[0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="Quaternion norm"):
        SensorMountConfig(target_kind="robot_link", target="pelvis", orientation=[2.0, 0.0, 0.0, 0.0])


def test_conflicting_warp_render_flag_rejected() -> None:
    # use_shadows is global to the shared Warp render context; cameras setting it differently are
    # rejected.
    a = _cam(mujoco=MujocoCameraConfig(use_shadows=True))
    b = _cam(mujoco=MujocoCameraConfig(use_shadows=False))
    with pytest.raises(ValueError, match="render flag 'use_shadows'"):
        validate_camera_dict({"a": a, "b": b})


def test_agreeing_warp_render_flag_allowed() -> None:
    a = _cam(mujoco=MujocoCameraConfig(use_shadows=True))
    b = _cam(mujoco=MujocoCameraConfig(use_shadows=True))
    validate_camera_dict({"a": a, "b": b})  # agreement is fine


def test_none_warp_render_flag_imposes_no_constraint() -> None:
    # One camera sets the flag, the other leaves it None: no conflict.
    a = _cam(mujoco=MujocoCameraConfig(use_textures=False))
    b = _cam()  # default sub-config leaves use_textures unset
    validate_camera_dict({"a": a, "b": b})  # must not raise


def test_camera_backend_configs_default_to_concrete_structs() -> None:
    camera = _cam()
    assert isinstance(camera.isaacsim, IsaacSimCameraConfig)
    assert isinstance(camera.isaacgym, IsaacGymCameraConfig)
    assert isinstance(camera.mujoco, MujocoCameraConfig)


def test_isaacsim_defaults_match_native_lens_defaults() -> None:
    config = IsaacSimCameraConfig()
    assert config.focal_length == 24.0
    assert config.f_stop == 0.0
    assert config.focus_distance == 400.0
    assert config.fisheye.nominal_width == 1936.0
    assert config.fisheye.nominal_height == 1216.0
    assert config.fisheye.optical_centre_x == 970.94244
    assert config.fisheye.optical_centre_y == 600.37482
    assert config.fisheye.max_fov == 200.0
    assert config.fisheye.polynomial_b == 0.00245


def test_camera_backend_configs_reject_none() -> None:
    mount = SensorMountConfig(target_kind="robot_link", target="pelvis")
    with pytest.raises(ValueError, match="isaacsim"):
        CameraSensorConfig(mount=mount, isaacsim=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="isaacgym"):
        CameraSensorConfig(mount=mount, isaacgym=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="mujoco"):
        CameraSensorConfig(mount=mount, mujoco=None)  # type: ignore[arg-type]


@pytest.mark.parametrize("projection_type", ISAACSIM_FISHEYE_PROJECTION_TYPES)
def test_isaacsim_fisheye_projection_and_calibration_allowed(projection_type: str) -> None:
    config = IsaacSimCameraConfig(
        projection_type=projection_type,  # type: ignore[arg-type]
        fisheye=IsaacSimFisheyeConfig(
            nominal_width=1920.0,
            nominal_height=1080.0,
            optical_centre_x=960.0,
            optical_centre_y=540.0,
            max_fov=220.0,
            polynomial_a=0.01,
            polynomial_b=0.002,
        ),
    )
    assert config.projection_type == projection_type
    assert config.fisheye.max_fov == 220.0


def test_isaacsim_pinhole_allows_dormant_fisheye_calibration() -> None:
    config = IsaacSimCameraConfig(fisheye=IsaacSimFisheyeConfig(max_fov=180.0, polynomial_b=0.25))
    assert config.projection_type == "pinhole"
    assert config.fisheye.max_fov == 180.0
    assert config.fisheye.polynomial_b == 0.25


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("focal_length", 0.0),
        ("focus_distance", math.inf),
        ("f_stop", -1.0),
    ],
)
def test_isaacsim_lens_values_must_be_valid(field: str, value: float) -> None:
    kwargs: dict[str, Any] = {field: value}
    with pytest.raises(ValueError, match=field):
        IsaacSimCameraConfig(**kwargs)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("nominal_width", -1.0),
        ("nominal_height", 0.0),
        ("optical_centre_x", math.nan),
        ("max_fov", 361.0),
        ("polynomial_f", math.inf),
    ],
)
def test_isaacsim_fisheye_values_must_be_valid(field: str, value: float) -> None:
    kwargs: dict[str, Any] = {field: value}
    with pytest.raises(ValueError, match=field):
        IsaacSimFisheyeConfig(**kwargs)


def test_lidar_range_and_pattern_validation() -> None:
    mount = SensorMountConfig(target_kind="robot_link", target="pelvis")
    assert CameraSensorConfig(mount=mount).depth_clipping_behavior == "none"
    assert CameraSensorConfig(mount=mount, depth_clipping_behavior="max").depth_clipping_behavior == "max"
    assert CameraSensorConfig(mount=mount, depth_clipping_behavior="zero").depth_clipping_behavior == "zero"
    assert LidarSensorConfig(mount=mount).range_clipping_behavior == "none"
    lidar = LidarSensorConfig(
        mount=mount,
        near=0.2,
        far=20.0,
        range_clipping_behavior="max",
        pattern=GridLidarRayPatternConfig(
            horizontal_angles=[-90.0, 0.0, 90.0],
            vertical_angles=[-10.0, 10.0],
        ),
    )
    validate_camera_dict({"camera": _cam()})
    assert lidar.range_clipping_behavior == "max"
    assert LidarSensorConfig(mount=mount, range_clipping_behavior="zero").range_clipping_behavior == "zero"

    with pytest.raises(ValueError, match="far"):
        LidarSensorConfig(mount=mount, near=1.0, far=1.0)
    with pytest.raises(ValueError, match="range_clipping_behavior"):
        LidarSensorConfig(mount=mount, range_clipping_behavior="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="depth_clipping_behavior"):
        CameraSensorConfig(mount=mount, depth_clipping_behavior="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="horizontal_fov"):
        GridLidarRayPatternConfig(horizontal_fov=[0.0, 361.0])
    with pytest.raises(ValueError, match="zero vector"):
        CustomLidarRayPatternConfig(ray_directions=[[0.0, 0.0, 0.0]])


def test_isaacsim_depth_clipping_must_use_backend_independent_field() -> None:
    mount = SensorMountConfig(target_kind="robot_link", target="pelvis")
    with pytest.raises(ValueError, match="depth_clipping_behavior"):
        CameraSensorConfig(
            mount=mount,
            data_types=["depth"],
            isaacsim={"depth_clipping_behavior": "max"},  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("near", "far"),
    [
        (-0.1, 1.0),
        (1.0, 1.0),
        (2.0, 1.0),
        (math.nan, 1.0),
        (0.1, math.inf),
    ],
)
def test_camera_range_bounds_must_be_finite_and_ordered(near: float, far: float) -> None:
    with pytest.raises(ValueError, match="near|far"):
        CameraSensorConfig(mount=SensorMountConfig(target_kind="robot_link", target="pelvis"), near=near, far=far)


@pytest.mark.parametrize("vertical_fov", [0.0, 180.0, -1.0, math.nan, math.inf])
def test_camera_vertical_fov_must_be_finite_and_perspective_safe(vertical_fov: float) -> None:
    with pytest.raises(ValueError, match="vertical_fov"):
        CameraSensorConfig(
            mount=SensorMountConfig(target_kind="robot_link", target="pelvis"),
            vertical_fov=vertical_fov,
        )


def test_explicit_lidar_pattern_organized_shape_validation() -> None:
    pattern = CustomLidarRayPatternConfig(
        ray_directions=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        organized_shape=[1, 2],
    )
    assert pattern.organized_shape == [1, 2]
    with pytest.raises(ValueError, match="organized_shape"):
        CustomLidarRayPatternConfig(
            ray_directions=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            organized_shape=[2, 2],
        )


@pytest.mark.parametrize(
    ("pattern", "provider_cls", "publish_hz", "expected_shape"),
    [
        (
            GridLidarRayPatternConfig(horizontal_angles=[0.0, 90.0], vertical_angles=[-10.0, 10.0]),
            GridLidarPatternProvider,
            20.0,
            (2, 2),
        ),
        (
            CustomLidarRayPatternConfig(ray_directions=[[0.0, 0.0, -1.0]], organized_shape=[1, 1]),
            CustomLidarPatternProvider,
            20.0,
            (1, 1),
        ),
    ],
)
def test_lidar_pattern_config_selects_its_runtime_provider(
    pattern: LidarRayPatternConfig,
    provider_cls: Any,
    publish_hz: float,
    expected_shape: tuple[int, int],
) -> None:
    selected_cls = pattern.get_provider_cls()
    assert selected_cls is provider_cls
    provider = selected_cls(pattern, device="cpu", publish_hz=publish_hz)
    assert provider.pattern_at(0.0).directions.shape == (expected_shape[0] * expected_shape[1], 3)
    assert (provider.height, provider.width) == expected_shape


def test_isaacsim_lidar_mesh_paths_must_not_be_empty() -> None:
    assert IsaacSimLidarConfig().mesh_prim_paths is None
    with pytest.raises(ValueError, match="mesh_prim_paths"):
        IsaacSimLidarConfig(mesh_prim_paths=[])
    explicit = LidarSensorConfig(
        mount=SensorMountConfig(target_kind="world"),
        body_filter=LidarBodyFilterConfig(target_kind="actor", target="shell"),
        isaacsim=IsaacSimLidarConfig(mesh_prim_paths=["/World/ground"]),
    )
    assert explicit.body_filter == LidarBodyFilterConfig(target_kind="actor", target="shell")
    assert explicit.isaacsim is not None
    assert explicit.isaacsim.mesh_prim_paths == ["/World/ground"]


def test_mujoco_lidar_query_filters_are_validated() -> None:
    assert MujocoLidarConfig().geom_groups == [True, True, True, False, False, False]
    with pytest.raises(ValueError, match="six group"):
        MujocoLidarConfig(geom_groups=[True] * 5)


def test_lidar_body_filter_is_backend_neutral_and_validated() -> None:
    mount = SensorMountConfig(target_kind="world")
    assert LidarSensorConfig(mount=mount).body_filter == LidarBodyFilterConfig(target_kind="mount")
    assert LidarBodyFilterConfig(target_kind="none").target == ""
    robot = LidarBodyFilterConfig(target_kind="robot")
    assert LidarSensorConfig(mount=mount, body_filter=robot).body_filter == robot
    explicit = LidarBodyFilterConfig(target_kind="robot_link", target="head_link")
    assert LidarSensorConfig(mount=mount, body_filter=explicit).body_filter == explicit
    with pytest.raises(ValueError, match="must be empty"):
        LidarBodyFilterConfig(target_kind="mount", target="head_link")
    with pytest.raises(ValueError, match="must be empty"):
        LidarBodyFilterConfig(target_kind="robot", target="head_link")
    with pytest.raises(ValueError, match="non-empty name"):
        LidarBodyFilterConfig(target_kind="actor", target=" ")


# The camera-frame sink fields (cameras/modalities/record_video/...) live on CameraVizPluginConfig;
# their validation is covered in config_types/tests/test_plugin_egress_config.py.
