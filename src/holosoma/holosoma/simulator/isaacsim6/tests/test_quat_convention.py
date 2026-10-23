"""The w-first -> w-last conversion at the Isaac Lab 3.x configuration boundary.

Worth testing despite being four lines: a missed or wrong conversion here is silent. Isaac Lab
3.0 accepts the reordered value without complaint and produces a plausible scene with the wrong
orientation, and the most common configured value -- identity -- reads the same under either
convention, so a default-config smoke run exercises none of it.
"""

from __future__ import annotations

import math

import pytest

from holosoma.simulator.isaacsim6.quat_convention import wxyz_to_xyzw

pytestmark = pytest.mark.no_sim


def test_identity_maps_to_the_3x_identity() -> None:
    """holosoma's identity (1,0,0,0) must become 3.x's identity (0,0,0,1), not pass through."""
    assert wxyz_to_xyzw((1.0, 0.0, 0.0, 0.0)) == (0.0, 0.0, 0.0, 1.0)


def test_components_are_rotated_not_reversed() -> None:
    """w moves to the end; x, y, z keep their order.

    A reversal would also map identity correctly, so this pins the actual permutation with four
    distinguishable components.
    """
    assert wxyz_to_xyzw((1.0, 2.0, 3.0, 4.0)) == (2.0, 3.0, 4.0, 1.0)


def test_a_real_rotation_survives_the_round_trip() -> None:
    """A 90 degree rotation about Z keeps its angle and axis under the reorder."""
    half = math.sqrt(0.5)
    assert wxyz_to_xyzw((half, 0.0, 0.0, half)) == (0.0, 0.0, half, half)


def test_a_list_is_accepted_and_normalised_to_a_tuple() -> None:
    """Config values arrive as lists; the Isaac Lab cfg fields are tuple-typed."""
    result = wxyz_to_xyzw([1.0, 0.0, 0.0, 0.0])
    assert isinstance(result, tuple)
    assert result == (0.0, 0.0, 0.0, 1.0)


@pytest.mark.parametrize("bad", [(), (1.0,), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0, 0.0)])
def test_wrong_component_count_raises(bad: tuple[float, ...]) -> None:
    """A short or long sequence must fail loudly rather than be silently reordered.

    Reordering a 3-vector would yield a syntactically valid quaternion holding the wrong
    rotation, which is precisely the silent failure this module exists to prevent.
    """
    with pytest.raises(ValueError, match="exactly 4 components"):
        wxyz_to_xyzw(bad)
