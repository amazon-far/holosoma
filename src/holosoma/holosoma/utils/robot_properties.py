"""Resolve sparse physical-property overrides without changing the authored robot config."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from holosoma.utils.name_tags import expand_name_assignments, expand_name_spec

if TYPE_CHECKING:
    from holosoma.config_types.robot import RobotConfig


@dataclass(frozen=True)
class DofPropertyOverrides:
    position: dict[str, tuple[float, float]]
    velocity: dict[str, float]
    effort: dict[str, float]
    armature: dict[str, float]
    friction: dict[str, float]


def _ordered_dof_names(names: Sequence[Any], *, what: str) -> list[str]:
    if isinstance(names, (str, bytes)):
        raise TypeError(f"{what} names must be a sequence of DOF names, not {names!r}.")
    ordered = list(names)
    invalid = [(index, name) for index, name in enumerate(ordered) if not isinstance(name, str)]
    if invalid:
        raise TypeError(f"{what} names must be strings; invalid entries: {invalid}.")
    duplicates = sorted(name for name, count in Counter(ordered).items() if count > 1)
    if duplicates:
        raise ValueError(f"{what} names contain duplicate DOF(s): {duplicates}.")
    return ordered


def _ordered_values(values: Sequence[Any], *, what: str) -> list[Any]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{what} values must be a sequence, not {values!r}.")
    return list(values)


def _as_float(value: Any, *, what: str, index: int, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{what} value at index {index} for DOF {name!r} must be numeric, got {value!r}.") from exc


def dof_values_from_sequence(
    names: Sequence[str],
    values: Sequence[Any],
    *,
    what: str,
) -> dict[str, float]:
    """Build an exact-name DOF map from parallel ordered sequences.

    This validates authoring alignment and numeric conversion only. Field-specific
    physical constraints remain the responsibility of :class:`RobotConfig`.
    """
    ordered_names = _ordered_dof_names(names, what=what)
    ordered_values = _ordered_values(values, what=what)
    if len(ordered_names) != len(ordered_values):
        raise ValueError(f"{what} has {len(ordered_values)} values for {len(ordered_names)} DOF names.")
    return {
        name: _as_float(value, what=what, index=index, name=name)
        for index, (name, value) in enumerate(zip(ordered_names, ordered_values))
    }


def dof_limits_from_sequences(
    names: Sequence[str],
    lower: Sequence[Any],
    upper: Sequence[Any],
    *,
    what: str = "DOF position limits",
) -> dict[str, tuple[float, float]]:
    """Build exact-name DOF limits from parallel name, lower, and upper sequences.

    This validates authoring alignment and numeric conversion only. Range semantics
    remain the responsibility of :class:`RobotConfig`.
    """
    ordered_names = _ordered_dof_names(names, what=what)
    ordered_lower = _ordered_values(lower, what=f"{what} lower")
    ordered_upper = _ordered_values(upper, what=f"{what} upper")
    if len(ordered_names) != len(ordered_lower) or len(ordered_names) != len(ordered_upper):
        raise ValueError(
            f"{what} has {len(ordered_lower)} lower and {len(ordered_upper)} upper values "
            f"for {len(ordered_names)} DOF names."
        )
    return {
        name: (
            _as_float(lo, what=f"{what} lower", index=index, name=name),
            _as_float(hi, what=f"{what} upper", index=index, name=name),
        )
        for index, (name, lo, hi) in enumerate(zip(ordered_names, ordered_lower, ordered_upper))
    }


def resolve_dof_properties(robot: RobotConfig) -> DofPropertyOverrides:
    """Validate the complete assignment before any backend writes or actuator grouping.

    Each property is an independent assignment layer. Repeated names within a tag are harmless;
    separate keys reaching the same joint are ambiguous, even with equal values. Missing joints
    retain their asset values. Defaults must fall within explicitly configured position limits;
    asset-authored limits are unavailable until the simulator loads the asset. The returned maps
    use canonical names in authored selection order.
    """

    def resolve(spec: Sequence[str]) -> list[str]:
        return expand_name_spec(spec, robot.joint_tags, robot.dof_names, kind="joint")

    positions = expand_name_assignments(
        robot.dof_pos_limits or {}, resolve, what="RobotConfig.dof_pos_limits", kind="DOF"
    )
    for name, (lower, upper) in positions.items():
        if math.isnan(lower) or math.isnan(upper) or lower > upper or lower == math.inf or upper == -math.inf:
            raise ValueError(f"RobotConfig.dof_pos_limits: DOF {name!r} has invalid range [{lower}, {upper}].")

    defaults = robot.default_dof_positions()
    for name, (lower, upper) in positions.items():
        default = defaults[name]
        if not lower <= default <= upper:
            raise ValueError(
                "RobotConfig.init_state.default_joint_positions: "
                f"DOF {name!r} default {default} is outside configured position range [{lower}, {upper}]."
            )

    scalar_maps = []
    for field in ("dof_vel_limits", "dof_effort_limits", "dof_armature", "dof_joint_friction"):
        values = expand_name_assignments(getattr(robot, field) or {}, resolve, what=f"RobotConfig.{field}", kind="DOF")
        for name, value in values.items():
            finite_required = field in ("dof_armature", "dof_joint_friction")
            if math.isnan(value) or value < 0 or (finite_required and not math.isfinite(value)):
                raise ValueError(
                    f"RobotConfig.{field}: DOF {name!r} needs a nonnegative "
                    f"{'finite ' if finite_required else ''}value, got {value}."
                )
        scalar_maps.append(values)
    return DofPropertyOverrides(positions, *scalar_maps)
