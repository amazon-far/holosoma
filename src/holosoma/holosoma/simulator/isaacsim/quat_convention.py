"""Quaternion conversion for Isaac Lab configuration boundaries."""

from __future__ import annotations

from collections.abc import Sequence


def wxyz_to_xyzw(quaternion: Sequence[float]) -> tuple[float, float, float, float]:
    if len(quaternion) != 4:
        raise ValueError(f"A quaternion needs exactly 4 components, got {len(quaternion)}: {tuple(quaternion)!r}")
    w, x, y, z = (float(c) for c in quaternion)
    return x, y, z, w
