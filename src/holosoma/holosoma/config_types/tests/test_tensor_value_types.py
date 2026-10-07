"""Pydantic tensor quaternion contracts and simulation-startup compatibility."""

from __future__ import annotations

import subprocess
import sys
from typing import Any

import pytest
import torch
from pydantic import BaseModel, TypeAdapter, ValidationError, validate_call
from pydantic.dataclasses import dataclass

from holosoma.config_types.value_types import (
    QuaternionWXYZTensor,
    QuaternionXYZWTensor,
    UnitQuaternionWXYZTensor,
    UnitQuaternionXYZWTensor,
)

pytestmark = pytest.mark.no_sim

_REGULAR_TYPES = [QuaternionWXYZTensor, QuaternionXYZWTensor]
_UNIT_TYPES = [UnitQuaternionWXYZTensor, UnitQuaternionXYZWTensor]
_ALL_TYPES = [*_REGULAR_TYPES, *_UNIT_TYPES]


@pytest.mark.parametrize("alias", _REGULAR_TYPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_regular_tensors_preserve_nonunit_values(alias: Any, dtype: torch.dtype) -> None:
    values = torch.tensor([[0.0] * 4, [-2.0, 3.0, -4.0, 5.0]], dtype=dtype)
    assert TypeAdapter(alias).validate_python(values) is values


@pytest.mark.parametrize("alias", _ALL_TYPES)
@pytest.mark.parametrize(
    "values",
    [
        None,
        [0.0, 0.0, 0.0, 1.0],
        torch.tensor(1.0),
        torch.ones(3),
        torch.ones(2, 5),
        torch.ones(2, 4, dtype=torch.int64),
        torch.ones(2, 4, dtype=torch.complex64),
        torch.tensor([[float("nan"), 0.0, 0.0, 1.0]]),
        torch.tensor([[0.0, float("inf"), 0.0, 1.0]]),
    ],
)
def test_tensor_aliases_reject_invalid_types_shapes_and_components(alias: Any, values: Any) -> None:
    with pytest.raises(ValidationError, match="sample"):
        TypeAdapter(alias).validate_python(values, context={"field_name": "sample"})


@pytest.mark.parametrize("alias", _UNIT_TYPES)
@pytest.mark.parametrize("norm", [0.0, 1e-30, 0.5, 2.0, 1.001, 0.999, 1e30])
def test_unit_tensors_reject_bad_norms(alias: Any, norm: float) -> None:
    values = torch.tensor([[0.0, 0.0, 0.0, norm]])
    with pytest.raises(ValidationError, match="sample.*not unit-norm"):
        TypeAdapter(alias).validate_python(values, context={"field_name": "sample"})


@pytest.mark.parametrize("alias", _UNIT_TYPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("delta", [-1e-4, -1e-6, 0.0, 1e-6, 1e-4])
def test_unit_tensors_normalize_inclusive_tolerance(alias: Any, dtype: torch.dtype, delta: float) -> None:
    values = torch.tensor([[-(1.0 + delta), 0.0, 0.0, 0.0]], dtype=dtype)
    result = TypeAdapter(alias).validate_python(values)
    assert torch.equal(result, torch.tensor([[-1.0, 0.0, 0.0, 0.0]], dtype=dtype))


@pytest.mark.parametrize("alias", _UNIT_TYPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_unit_tensors_preserve_shape_dtype_sign_order_and_input(alias: Any, dtype: torch.dtype) -> None:
    expected = torch.tensor([-1.0, 2.0, -3.0, 4.0], dtype=dtype)
    expected /= torch.linalg.vector_norm(expected)
    # Exercise a noncontiguous view of caller-owned pose storage.
    storage = torch.zeros(2, 3, 7, dtype=dtype)
    storage[..., 3:7] = expected * 1.00001
    values = storage[..., 3:7]
    before = storage.clone()
    result = TypeAdapter(alias).validate_python(values)
    assert result is not values
    assert result.shape == values.shape
    assert result.dtype == values.dtype
    assert result.device == values.device
    assert torch.allclose(result, expected.expand_as(values))
    assert torch.equal(storage, before)


@pytest.mark.parametrize("alias", _ALL_TYPES)
@pytest.mark.parametrize("shape", [(0, 4), (2, 0, 4), (4,)])
def test_tensor_aliases_accept_empty_batches_and_single_quaternions(alias: Any, shape: tuple[int, ...]) -> None:
    values = torch.zeros(shape)
    values[..., 3] = 1.0
    result = TypeAdapter(alias).validate_python(values)
    assert result.shape == shape
    assert torch.equal(result, values)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_norm_cannot_hide_large_deviation(dtype: torch.dtype) -> None:
    values = torch.tensor([[1.0, 0.02, 0.0, 0.0]], dtype=dtype)
    with pytest.raises(ValidationError, match="not unit-norm"):
        TypeAdapter(UnitQuaternionWXYZTensor).validate_python(values)
    unit = torch.tensor([[0.5, -0.5, 0.5, -0.5]], dtype=dtype)
    result = TypeAdapter(UnitQuaternionWXYZTensor).validate_python(unit)
    assert result.dtype == dtype
    assert torch.equal(result, unit)


def test_tensor_validation_reports_bad_row_without_mutating_batch() -> None:
    values = torch.zeros(2, 3, 4)
    values[..., 3] = 1.00001
    values[1, 2, 3] = 2.0
    before = values.clone()
    with pytest.raises(ValidationError, match="flattened index 5"):
        TypeAdapter(UnitQuaternionXYZWTensor).validate_python(values)
    assert torch.equal(values, before)


def test_tensor_aliases_work_in_models_dataclasses_and_validated_functions() -> None:
    class Pose(BaseModel):
        orientation: UnitQuaternionXYZWTensor

    @dataclass
    class Algebra:
        quaternion: QuaternionWXYZTensor

    @validate_call
    def rotation(orientation: UnitQuaternionWXYZTensor) -> torch.Tensor:
        return orientation

    value = torch.tensor([[0.0, 0.0, 0.0, -1.00001]])
    expected = torch.tensor([[0.0, 0.0, 0.0, -1.0]])
    assert torch.equal(Pose(orientation=value).orientation, expected)
    assert torch.equal(rotation(value), expected)
    assert Algebra(quaternion=value).quaternion is value
    with pytest.raises(ValidationError) as exc:
        Pose(orientation=torch.zeros(2, 4))
    assert exc.value.errors()[0]["loc"] == ("orientation",)


def test_unit_tensor_validation_preserves_autograd() -> None:
    values = torch.tensor([[0.0, 0.0, 0.0, 1.00001]], requires_grad=True)
    result = TypeAdapter(UnitQuaternionXYZWTensor).validate_python(values)
    result.sum().backward()
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_unit_tensor_validation_preserves_cuda_device() -> None:
    values = torch.tensor([[0.0, 0.0, 0.0, -1.00001]], device="cuda")
    result = TypeAdapter(UnitQuaternionXYZWTensor).validate_python(values)
    assert result.device == values.device
    assert result.dtype == values.dtype
    assert torch.equal(result, torch.tensor([[0.0, 0.0, 0.0, -1.0]], device=values.device))


def test_value_type_schemas_do_not_import_torch_before_backend_startup() -> None:
    script = """
import sys
from pydantic import TypeAdapter
from holosoma.config_types import value_types
from holosoma.config_types.light import DefaultLightConfig
from holosoma.config_types.robot import RobotInitState
from holosoma.config_types.scene import SceneConfig
from holosoma.config_types.sensor import SensorMountConfig
from holosoma.config_types.plugin import ROS2OdometryPluginConfig

for name in (
    "QuaternionWXYZ", "QuaternionXYZW", "UnitQuaternionWXYZ", "UnitQuaternionXYZW",
    "UnitQuaternionWXYZTuple", "QuaternionWXYZTensor", "QuaternionXYZWTensor",
    "UnitQuaternionWXYZTensor", "UnitQuaternionXYZWTensor",
):
    TypeAdapter(getattr(value_types, name)).json_schema()
DefaultLightConfig()
SceneConfig()
SensorMountConfig(target_kind="world")
ROS2OdometryPluginConfig()
assert "torch" not in sys.modules
assert "holosoma.utils.rotations" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
