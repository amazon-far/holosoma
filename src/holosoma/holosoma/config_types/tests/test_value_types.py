"""Quaternion boundary contracts, including CLI and saved configuration compatibility."""

from __future__ import annotations

import dataclasses
import io
import json
import math
import sys
from contextlib import redirect_stderr, redirect_stdout
from typing import Any

import pytest
import tyro
from pydantic import TypeAdapter, ValidationError

from holosoma.config_types.light import (
    DefaultLightConfig,
    DistantLightConfig,
    DomeLightConfig,
    RectLightConfig,
    light_ray_direction,
)
from holosoma.config_types.plugin import ROS2OdometryPluginConfig, ROS2PointCloudRoute
from holosoma.config_types.robot import RobotInitState
from holosoma.config_types.scene import RigidObjectConfig, SceneConfig, SceneFileConfig
from holosoma.config_types.sensor import SensorMountConfig
from holosoma.config_types.value_types import (
    QuaternionWXYZ,
    QuaternionXYZW,
    UnitQuaternionWXYZ,
    UnitQuaternionWXYZTuple,
    UnitQuaternionXYZW,
)
from holosoma.utils.tyro_utils import TYRO_CONIFG

pytestmark = pytest.mark.no_sim

_UNIT_TYPES = [UnitQuaternionWXYZ, UnitQuaternionXYZW, UnitQuaternionWXYZTuple]
_ALL_TYPES = [QuaternionWXYZ, QuaternionXYZW, *_UNIT_TYPES]
_CONFIG_FIELDS = [
    pytest.param(SceneFileConfig, "orientation", {}, list, id="scene-file"),
    pytest.param(RigidObjectConfig, "orientation", {}, list, id="rigid-object"),
    pytest.param(SensorMountConfig, "orientation", {"target_kind": "world"}, list, id="sensor-mount"),
    pytest.param(DefaultLightConfig, "orientation", {}, tuple, id="default-light"),
    pytest.param(DomeLightConfig, "orientation", {}, tuple, id="dome-light"),
    pytest.param(DistantLightConfig, "orientation", {}, tuple, id="distant-light"),
    pytest.param(RectLightConfig, "orientation", {}, tuple, id="rect-light"),
    pytest.param(ROS2OdometryPluginConfig, "orientation", {}, list, id="odometry"),
    pytest.param(ROS2PointCloudRoute, "point_rotation", {"lidar": "scan", "topic": "/scan"}, list, id="pointcloud"),
    pytest.param(
        RobotInitState,
        "rot",
        {"pos": [0.0] * 3, "lin_vel": [0.0] * 3, "ang_vel": [0.0] * 3, "default_joint_angles": {}},
        list,
        id="robot-xyzw",
    ),
]


@pytest.mark.parametrize("alias", [QuaternionWXYZ, QuaternionXYZW])
@pytest.mark.parametrize("value", [[0.0] * 4, [-2.0, 3.0, -4.0, 5.0], [sys.float_info.max] * 4])
def test_general_quaternions_preserve_nonunit_values(alias: Any, value: list[float]) -> None:
    assert TypeAdapter(alias).validate_python(value) == value


@pytest.mark.parametrize("alias", _ALL_TYPES)
@pytest.mark.parametrize(
    "value",
    [
        [],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, math.nan, 0.0, 0.0],
        [1.0, 0.0, math.inf, 0.0],
        [1.0, 0.0, 0.0, -math.inf],
        [[1.0, 0.0, 0.0, 0.0]],
        "1,0,0,0",
        None,
    ],
)
def test_quaternions_reject_wrong_shape_and_nonfinite_values(alias: Any, value: Any) -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(alias).validate_python(value)


@pytest.mark.parametrize("alias", _UNIT_TYPES)
@pytest.mark.parametrize("norm", [0.0, 1e-300, 0.5, 2.0, 1e200, sys.float_info.max, 1.0 - 1.01e-4, 1.0 + 1.01e-4])
def test_unit_quaternions_reject_large_norm_errors(alias: Any, norm: float) -> None:
    with pytest.raises(ValidationError, match="quaternion|Quaternion norm"):
        TypeAdapter(alias).validate_python([norm, 0.0, 0.0, 0.0])


@pytest.mark.parametrize("alias", _UNIT_TYPES)
def test_unit_quaternions_reject_overflowing_norm(alias: Any) -> None:
    with pytest.raises(ValidationError, match="Quaternion norm"):
        TypeAdapter(alias).validate_python([sys.float_info.max] * 4)


@pytest.mark.parametrize("alias", _UNIT_TYPES)
@pytest.mark.parametrize("delta", [-1e-4, -1e-6, 0.0, 1e-6, 1e-4])
def test_unit_quaternions_normalize_inclusive_tolerance(alias: Any, delta: float) -> None:
    assert tuple(TypeAdapter(alias).validate_python([-(1.0 + delta), 0.0, 0.0, 0.0])) == (-1.0, 0.0, 0.0, 0.0)


@pytest.mark.parametrize("alias", _UNIT_TYPES)
def test_normalization_preserves_sign_order_and_input(alias: Any) -> None:
    expected = [-1 / math.sqrt(30), 2 / math.sqrt(30), -3 / math.sqrt(30), 4 / math.sqrt(30)]
    value = [component * 1.000001 for component in expected]
    before = value.copy()
    result = TypeAdapter(alias).validate_python(value)
    assert result == pytest.approx(expected)
    assert math.hypot(*result) == pytest.approx(1.0, abs=1e-15)
    assert value == before


@pytest.mark.parametrize(("config_type", "field_name", "kwargs", "container"), _CONFIG_FIELDS)
def test_rotation_configs_normalize_and_round_trip(
    config_type: type[Any], field_name: str, kwargs: dict[str, Any], container: type[Any]
) -> None:
    value = [-0.5000005, 0.5000005, 0.5000005, -0.5000005]
    config = config_type(**kwargs, **{field_name: value})
    result = getattr(config, field_name)
    assert type(result) is container
    assert result == pytest.approx([-0.5, 0.5, 0.5, -0.5])

    restored = config_type(**json.loads(json.dumps(dataclasses.asdict(config))))
    assert type(getattr(restored, field_name)) is container
    assert getattr(restored, field_name) == result

    # Config replacement must validate again.
    with pytest.raises(ValueError, match=field_name):
        dataclasses.replace(config, **{field_name: [2.0, 0.0, 0.0, 0.0]})


@pytest.mark.parametrize(("config_type", "field_name", "kwargs", "container"), _CONFIG_FIELDS)
@pytest.mark.parametrize("value", [[1.0, 0.0, 0.0], [0.0] * 4, [math.nan, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]])
def test_rotation_configs_reject_invalid_values(
    config_type: type[Any], field_name: str, kwargs: dict[str, Any], container: type[Any], value: list[float]
) -> None:
    with pytest.raises(ValueError, match=field_name):
        config_type(**kwargs, **{field_name: value})


@pytest.mark.parametrize(("config_type", "field_name", "kwargs", "container"), _CONFIG_FIELDS)
def test_rotation_cli_preserves_literal_syntax(
    config_type: type[Any], field_name: str, kwargs: dict[str, Any], container: type[Any]
) -> None:
    default = config_type(**kwargs, **{field_name: [1.0, 0.0, 0.0, 0.0]})
    value = container([1.000001, 0.0, 0.0, 0.0])
    args = [f"--{field_name.replace('_', '-')}", repr(value)]
    parsed = tyro.cli(config_type, default=default, args=args, config=TYRO_CONIFG)
    assert type(getattr(parsed, field_name)) is container
    assert tuple(getattr(parsed, field_name)) == (1.0, 0.0, 0.0, 0.0)

    args[-1] = repr(container([2.0, 0.0, 0.0, 0.0]))
    with pytest.raises(SystemExit), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        tyro.cli(config_type, default=default, args=args, config=TYRO_CONIFG)


def test_nested_scene_rejects_invalid_saved_rotation() -> None:
    saved: dict[str, Any] = {"rigid_objects": {"panel": {"orientation": [2.0, 0.0, 0.0, 0.0]}}}
    with pytest.raises(ValidationError, match=r"rigid_objects.panel.orientation"):
        SceneConfig(**saved)


def test_light_direction_uses_normalized_wxyz_orientation() -> None:
    sqrt_half = math.sqrt(0.5)
    light = DistantLightConfig(orientation=(sqrt_half * 1.000001, 0.0, sqrt_half * 1.000001, 0.0))
    assert light_ray_direction(light.orientation) == pytest.approx((-1.0, 0.0, 0.0), abs=1e-15)


def test_robot_reset_keeps_xyzw_list_layout() -> None:
    robot = RobotInitState(
        pos=[1.0, 2.0, 3.0],
        rot=[0.0, 0.0, 0.0, 1.000001],
        lin_vel=[0.0] * 3,
        ang_vel=[0.0] * 3,
        default_joint_angles={},
    )
    assert robot.pos + robot.rot + robot.lin_vel + robot.ang_vel == [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0] + [0.0] * 6


@pytest.mark.parametrize("alias", _ALL_TYPES)
def test_alias_carries_exactly_one_validator(alias: Any) -> None:
    """Every alias must expose exactly one validator, not a chain of them.

    Pydantic 2.11 keeps only the last validator reachable through an ``Annotated`` alias when
    that alias annotates a dataclass field, so an alias carrying several chained validators
    enforces only its final step there while still behaving correctly through ``TypeAdapter``.
    A multi-step contract must therefore be sequenced inside a single validator function, as
    ``value_types._validated_unit_quaternion`` does. This asserts that shape directly, because
    the behavioral tests only expose a violation on an affected pydantic version.
    """
    validators = [
        metadata
        for metadata in getattr(alias, "__metadata__", ())
        if type(metadata).__name__ in ("AfterValidator", "BeforeValidator", "PlainValidator", "WrapValidator")
    ]
    assert len(validators) == 1, (
        f"{alias} carries {len(validators)} validators; use exactly one and sequence the steps "
        f"inside it (see value_types._validated_unit_quaternion)."
    )
