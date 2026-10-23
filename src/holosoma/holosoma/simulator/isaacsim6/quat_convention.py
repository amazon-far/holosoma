"""Quaternion order conversion at the Isaac Lab 3.x configuration boundary.

Isaac Lab 2.3.2 took configuration quaternions **w-first**: ``AssetBaseCfg.InitialStateCfg.rot``
and the sensor ``OffsetCfg.rot`` fields all defaulted to ``(1, 0, 0, 0)`` and documented
themselves as ``(w, x, y, z)``. Isaac Lab 3.0 flipped both to **w-last**, defaulting to
``(0, 0, 0, 1)``.

holosoma's own configuration surface is w-first (``UnitQuaternionWXYZ`` in
``config_types.value_types``), and that is not changing, so every value handed to one of those
3.x cfg fields has to be reordered here, at the boundary.

Why this is isolated in its own module rather than inlined
----------------------------------------------------------
The flip is a *silent* failure. Passing w-first data into a w-last field raises nothing, produces
a plausible-looking scene, and yields a wrong orientation -- and the most common configured value,
identity, is byte-identical under both conventions (``(1,0,0,0)`` read as xyzw is a 180 degree
rotation about X, but identity is ``(1,0,0,0)`` w-first and ``(0,0,0,1)`` w-last, so a default
config exercises neither error). A default-config smoke test therefore cannot catch a missed
call site. Naming the conversion makes the boundary greppable: every ``OffsetCfg``/
``InitialStateCfg`` built in this package should pass its ``rot`` through this function, and a
reviewer can check that by searching for the two names.

This lives in the 6.1 package, not in ``simulator/isaacsim6``, deliberately: it exists only
because of 3.x's convention, so it should be deleted together with this package rather than
outliving it as a shared helper nobody needs.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Tuple

# `typing.Tuple`, not `tuple`: this alias is evaluated at runtime (an assignment, not an
# annotation, so `from __future__ import annotations` does not defer it) and mypy.ini pins
# `python_version = 3.8`, where the builtin is not subscriptable. Matches the convention in
# simulator/plugins/camera_consumer.py.
Quat = Tuple[float, float, float, float]


def wxyz_to_xyzw(quaternion: Sequence[float]) -> Quat:
    """Reorder a w-first quaternion into the w-last order Isaac Lab 3.x configs expect.

    Parameters
    ----------
    quaternion
        Four components in ``(w, x, y, z)`` order -- holosoma's convention.

    Returns
    -------
    tuple
        The same rotation in ``(x, y, z, w)`` order, for an Isaac Lab 3.x ``rot`` field.

    Raises
    ------
    ValueError
        If four components were not supplied. Raised rather than tolerated because a
        short sequence here would otherwise be reordered into a silently wrong rotation.
    """
    if len(quaternion) != 4:
        raise ValueError(f"A quaternion needs exactly 4 components, got {len(quaternion)}: {tuple(quaternion)!r}")
    w, x, y, z = (float(c) for c in quaternion)
    return (x, y, z, w)
