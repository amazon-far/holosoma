"""Pydantic value types for scalar configuration and tensor input boundaries.

``QuaternionWXYZ`` and ``QuaternionXYZW`` describe four finite components, including
non-unit and zero quaternions used in algebra. Rotations use the ``UnitQuaternion*``
aliases: norms within 1e-4 of one are normalized; larger deviations are rejected.
Component order and sign are preserved, never inferred or converted.

These are Pydantic annotations, not wrapper objects or validating constructors. Use
them on Pydantic fields, or through ``TypeAdapter`` at a plain Python boundary.
Each alias carries exactly ONE validator and never nests another validator-bearing
alias: pydantic 2.11 keeps only the last validator reachable through an alias when it
annotates a dataclass field. Preserve that shape when adding an alias here.
Validation runs at construction; fields with defaults should set ``validate_default=True``
in their dataclass field metadata. It cannot protect subsequent list mutation. List
aliases preserve existing config/CLI representations; the tuple variant preserves
the light configs' fixed-tuple CLI.

The ``*Tensor`` aliases validate existing floating tensors with shape ``(..., 4)``.
They preserve device, dtype, sign and component order. Regular aliases return the input;
unit aliases normalize into a new tensor without modifying caller-owned data.
Use a cached ``TypeAdapter`` at tensor input boundaries; annotations on ordinary Python
functions alone do not execute validation. Tensor aliases do not accept JSON or convert
lists/arrays to tensors.

Importing this module or building its schemas does not import Torch, so configuration
can be parsed before backend initialization. Torch is loaded only for tensor validation;
numerical quaternion operations and explicit order conversions remain in ``utils.rotations``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, List, Tuple, Union

from pydantic import AfterValidator, Field, PlainValidator, ValidationInfo
from typing_extensions import Annotated, TypeAlias

if TYPE_CHECKING:
    from torch import Tensor as _Tensor
else:
    # PlainValidator enforces the tensor type at runtime without importing Torch during config parsing.
    _Tensor = Any

UNIT_QUATERNION_NORM_TOLERANCE = 1e-4
"""Absolute norm tolerance shared by scalar configuration and tensor input validation."""
_QuaternionValue: TypeAlias = Union[List[float], Tuple[float, ...]]


def _validate_finite_components(value: _QuaternionValue) -> _QuaternionValue:
    if not all(math.isfinite(component) for component in value):
        raise ValueError("Quaternion must contain only finite values.")
    return value


def _validate_unit_norm(value: _QuaternionValue) -> _QuaternionValue:
    """Reject a zero quaternion or a norm further than the tolerance from one."""
    # hypot avoids overflow/underflow from squaring raw components.
    norm = math.hypot(*value)
    if norm == 0.0:
        raise ValueError("A unit quaternion must be a non-zero quaternion.")
    if not 1.0 - UNIT_QUATERNION_NORM_TOLERANCE <= norm <= 1.0 + UNIT_QUATERNION_NORM_TOLERANCE:
        raise ValueError(f"Quaternion norm must be within {UNIT_QUATERNION_NORM_TOLERANCE} of 1, got {norm}.")
    return value


def _normalize_quaternion(value: _QuaternionValue) -> _QuaternionValue:
    """Normalize an already validated near-unit quaternion, preserving its container."""
    norm = math.hypot(*value)
    if isinstance(value, tuple):
        return tuple(component / norm for component in value)
    return [component / norm for component in value]


def _validated_unit_quaternion(value: _QuaternionValue) -> _QuaternionValue:
    """Apply the whole unit-rotation contract: finite, then near-unit, then normalized.

    The three steps are separate functions above and are sequenced here so that each alias
    needs only one ``AfterValidator``. Pydantic 2.11 keeps only the last validator reachable
    through an ``Annotated`` alias when that alias annotates a dataclass field, so a contract
    expressed as several chained validators would run only its final step. Sequencing inside
    one function makes the order independent of how pydantic flattens alias metadata.
    """
    _validate_finite_components(value)
    _validate_unit_norm(value)
    return _normalize_quaternion(value)


# Keep the element type float: per-element Annotated types change Tyro's literal-list CLI.
# Each alias below carries exactly one validator and never nests another validator-bearing
# alias; see :func:`_validated_unit_quaternion` for why that shape is required.
_QUATERNION_WIDTH = Field(min_length=4, max_length=4)

QuaternionWXYZ: TypeAlias = Annotated[
    List[float],
    _QUATERNION_WIDTH,
    Field(description="Quaternion in [w, x, y, z] order."),
    AfterValidator(_validate_finite_components),
]
"""Four finite WXYZ components. Unit length is not required; zero is allowed."""

QuaternionXYZW: TypeAlias = Annotated[
    List[float],
    _QUATERNION_WIDTH,
    Field(description="Quaternion in [x, y, z, w] order."),
    AfterValidator(_validate_finite_components),
]
"""Four finite XYZW components. Unit length is not required; zero is allowed."""

UnitQuaternionWXYZ: TypeAlias = Annotated[
    List[float],
    _QUATERNION_WIDTH,
    Field(description="Unit quaternion in [w, x, y, z] order; norm tolerance 1e-4."),
    AfterValidator(_validated_unit_quaternion),
]
"""WXYZ rotation stored as a list, normalized only after its near-unit norm is validated."""

UnitQuaternionXYZW: TypeAlias = Annotated[
    List[float],
    _QUATERNION_WIDTH,
    Field(description="Unit quaternion in [x, y, z, w] order; norm tolerance 1e-4."),
    AfterValidator(_validated_unit_quaternion),
]
"""XYZW rotation stored as a list, normalized only after its near-unit norm is validated."""

UnitQuaternionWXYZTuple: TypeAlias = Annotated[
    Tuple[float, float, float, float],
    Field(description="Unit quaternion in [w, x, y, z] order; norm tolerance 1e-4."),
    AfterValidator(_validated_unit_quaternion),
]
"""Same WXYZ rotation contract, retaining the light configs' tuple representation."""


def _tensor_field_name(info: ValidationInfo) -> str:
    if isinstance(info.context, dict) and "field_name" in info.context:
        return str(info.context["field_name"])
    return info.field_name or "quaternion"


def _validate_quaternion_tensor(value: Any, info: ValidationInfo) -> _Tensor:
    """Check a tensor's shape, dtype and components without normalizing it."""
    import torch

    field_name = _tensor_field_name(info)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{field_name} must be a floating-point tensor.")
    if value.ndim == 0 or value.shape[-1] != 4:
        raise ValueError(f"{field_name} must have shape (..., 4), got {tuple(value.shape)}.")
    if not value.is_floating_point():
        raise ValueError(f"{field_name} must contain floating-point values, got {value.dtype}.")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{field_name} must contain only finite values.")
    return value


def _validate_tensor_unit_norm(value: _Tensor, info: ValidationInfo) -> _Tensor:
    """Reject any row whose norm is further than the tolerance from one."""
    import torch

    # Compute low-precision norms in float32 so rounding cannot hide a deviation above 1e-4.
    norm_dtype = torch.float64 if value.dtype == torch.float64 else torch.float32
    norms = torch.linalg.vector_norm(value, dim=-1, dtype=norm_dtype)
    valid = (norms >= 1.0 - UNIT_QUATERNION_NORM_TOLERANCE) & (norms <= 1.0 + UNIT_QUATERNION_NORM_TOLERANCE)
    if not bool(valid.all()):
        flat_index = int((~valid).reshape(-1).nonzero()[0, 0])
        norm = float(norms.reshape(-1)[flat_index])
        raise ValueError(
            f"{_tensor_field_name(info)} quaternion at flattened index {flat_index} is not unit-norm: "
            f"norm must be within {UNIT_QUATERNION_NORM_TOLERANCE} of 1, got {norm}."
        )
    return value


def _normalize_quaternion_tensor(value: _Tensor) -> _Tensor:
    """Normalize an already validated tensor using the shared rotation operation."""
    from holosoma.utils.rotations import quat_unit

    return quat_unit(value)


def _validated_unit_quaternion_tensor(value: Any, info: ValidationInfo) -> _Tensor:
    """Tensor counterpart of :func:`_validated_unit_quaternion`: check, then normalize.

    Sequenced in one validator for the same metadata-flattening reason documented there.
    """
    tensor = _validate_quaternion_tensor(value, info)
    _validate_tensor_unit_norm(tensor, info)
    return _normalize_quaternion_tensor(tensor)


QuaternionWXYZTensor: TypeAlias = Annotated[
    _Tensor,
    Field(description="Floating quaternion tensor with shape (..., 4), in [w, x, y, z] order."),
    PlainValidator(_validate_quaternion_tensor),
]
"""Finite WXYZ tensors; non-unit and zero algebraic quaternions are allowed."""

QuaternionXYZWTensor: TypeAlias = Annotated[
    _Tensor,
    Field(description="Floating quaternion tensor with shape (..., 4), in [x, y, z, w] order."),
    PlainValidator(_validate_quaternion_tensor),
]
"""Finite XYZW tensors; non-unit and zero algebraic quaternions are allowed."""

UnitQuaternionWXYZTensor: TypeAlias = Annotated[
    _Tensor,
    Field(description="Unit WXYZ quaternion tensor with shape (..., 4); norm tolerance 1e-4."),
    PlainValidator(_validated_unit_quaternion_tensor),
]
"""WXYZ rotation tensors, normalized only after their near-unit norms are validated."""

UnitQuaternionXYZWTensor: TypeAlias = Annotated[
    _Tensor,
    Field(description="Unit XYZW quaternion tensor with shape (..., 4); norm tolerance 1e-4."),
    PlainValidator(_validated_unit_quaternion_tensor),
]
"""XYZW rotation tensors, normalized only after their near-unit norms are validated."""
