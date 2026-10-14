"""Pure tests for ordered DOF-property authoring helpers."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from holosoma.utils.robot_properties import dof_limits_from_sequences, dof_values_from_sequence

pytestmark = pytest.mark.no_sim


def test_dof_values_from_sequence_preserves_name_order_and_converts_numbers() -> None:
    values = dof_values_from_sequence(("knee", "ankle"), (1, "2.5"), what="effort limits")
    assert list(values.items()) == [("knee", 1.0), ("ankle", 2.5)]


def test_dof_limits_from_sequences_pairs_bounds_in_name_order() -> None:
    limits = dof_limits_from_sequences(
        ["knee", "ankle"],
        [-1, "-0.5"],
        [2, "0.75"],
        what="position limits",
    )
    assert list(limits.items()) == [("knee", (-1.0, 2.0)), ("ankle", (-0.5, 0.75))]


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: dof_values_from_sequence(["a", "b"], [1.0], what="values"), "1 values for 2 DOF names"),
        (
            lambda: dof_limits_from_sequences(["a", "b"], [0.0], [1.0, 2.0], what="limits"),
            "1 lower and 2 upper values for 2 DOF names",
        ),
        (lambda: dof_values_from_sequence("ab", [1.0, 2.0], what="values"), "names must be a sequence"),
        (lambda: dof_values_from_sequence(["a"], "1", what="values"), "values must be a sequence"),
        (lambda: dof_values_from_sequence(["a", "a"], [1.0, 2.0], what="values"), "duplicate DOF"),
        (lambda: dof_values_from_sequence(["a"], [object()], what="values"), "must be numeric"),
    ],
)
def test_ordered_property_helpers_reject_ambiguous_or_misaligned_input(call: Callable[[], object], match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        call()
