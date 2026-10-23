"""Resolving which Isaac Sim backend package a simulator or config refers to.

holosoma carries one backend package per Isaac Sim generation -- ``simulator/isaacsim`` for 5.1 /
Isaac Lab 2.3.2 and ``simulator/isaacsim6`` for 6.1 / Isaac Lab 3.0 -- because the two generations
diverge too far for one implementation to serve both (Isaac Lab 3.0 inverted the quaternion
convention globally, renamed or removed most of the accessors holosoma calls, and replaced the
physics engine with a selectable one).

Most code never notices, because it talks to the simulator through ``BaseSimulator``. A handful of
call sites do reach into a backend package directly -- the randomization terms import ``events``
for Isaac-specific property writes, the launcher needs the lane's rendering-flag mapping -- and
those sites are backend-agnostic by design, so they cannot hard-code a package name. They also
cannot distinguish the lanes by ``SimulatorType``, which deliberately has a single ``ISAACSIM``
member covering both generations so the ~31 ``simulator_type ==`` comparisons across holosoma stay
untouched. This module is the counterpart to that choice: where code genuinely needs to know
*which* generation, it asks here.

**This lives at ``simulator/`` level, outside both lanes, and that placement is load-bearing.** It
resolves *between* the lanes, so putting it inside one of them would make the other lane's
resolution depend on the first lane's continued existence -- deleting the retired generation would
break the surviving one. A single module rather than a package because it is one file.

Resolution keys off the live simulator instance where there is one, and off the configured
``_target_`` where there is not. Both read the answer out of something that already exists, so
there is no registry to drift and nothing to update when a generation is added or retired.
"""

from __future__ import annotations

import importlib
import importlib.util
from types import ModuleType
from typing import Any

_LANE_PREFIX = "holosoma.simulator.isaacsim"
"""Shared prefix of every Isaac Sim backend package (``...isaacsim``, ``...isaacsim6``, ...)."""


def isaac_lane_package(simulator: Any) -> str:
    """The dotted name of the Isaac backend package ``simulator`` was built from.

    Parameters
    ----------
    simulator
        A live backend instance, e.g. ``env.simulator``.

    Returns
    -------
    str
        For example ``"holosoma.simulator.isaacsim"`` or ``"holosoma.simulator.isaacsim6"``.

    Raises
    ------
    TypeError
        If ``simulator`` is not an Isaac Sim backend. Raised rather than defaulting, because every
        caller is already inside an Isaac-only branch: arriving here with a MuJoCo
        simulator means the branch condition is wrong, and quietly importing the 5.1 lane would
        hide that behind an unrelated failure.
    """
    package = type(simulator).__module__.rpartition(".")[0]
    if not package.startswith(_LANE_PREFIX):
        raise TypeError(
            f"{type(simulator).__name__} (module {type(simulator).__module__!r}) is not an Isaac "
            f"Sim backend, so it has no Isaac lane package."
        )
    return package


def isaac_lane_submodule(simulator: Any, submodule: str) -> ModuleType:
    """Import ``submodule`` from the Isaac backend package ``simulator`` belongs to.

    Parameters
    ----------
    simulator
        A live backend instance, e.g. ``env.simulator``.
    submodule
        Module name relative to the lane package, e.g. ``"events"``.

    Examples
    --------
    Replaces a hard-coded lane import inside an Isaac-only branch::

        events = isaac_lane_submodule(simulator, "events")
        events.randomize_rigid_body_material(...)
    """
    return importlib.import_module(f"{isaac_lane_package(simulator)}.{submodule}")


def isaac_lane_package_for_target(target: str) -> str | None:
    """The Isaac backend package a config ``_target_`` names, or ``None`` if it is not Isaac.

    For code that runs *before* any simulator exists -- principally the app launcher and config
    validation -- where :func:`isaac_lane_package` cannot help because there is nothing to
    inspect yet. The preset's ``_target_`` is the only lane signal available at that point.

    Parameters
    ----------
    target
        A config ``_target_``, e.g. ``"holosoma.simulator.isaacsim6.isaacsim.IsaacSim"``.

    Returns
    -------
    str or None
        The lane package, or ``None`` when ``target`` names a non-Isaac backend.
    """
    package = target.rpartition(".")[0].rpartition(".")[0]
    return package if package.startswith(_LANE_PREFIX) else None


def isaac_lane_optional_module(target: str, submodule: str) -> ModuleType | None:
    """Import ``submodule`` from the lane named by a config ``_target_``, if that lane has one.

    Lets a lane *opt in* to overriding shared behaviour. A lane that needs no override simply
    omits the module and shared code keeps its existing path -- which is how the rendering-flag
    difference between Isaac Lab 2.3.2 and 3.0 is handled without a version check anywhere in
    shared code. Adding a generation means adding a folder; retiring one means deleting it.

    Returns
    -------
    ModuleType or None
        The module, or ``None`` when ``target`` is not an Isaac lane or that lane does not provide
        ``submodule``.
    """
    package = isaac_lane_package_for_target(target)
    if package is None:
        return None

    # `find_spec` first, so "this lane does not provide the module" is distinguished from "the
    # module exists and failed to import". Catching ModuleNotFoundError around the import would
    # conflate them: a lane launcher that imports an unavailable dependency would be treated as
    # opting out, shared code would fall back to the Isaac Lab 2.3.2 mapping, and the run would
    # come up headless with no error -- the exact silent drop this delegation exists to remove.
    name = f"{package}.{submodule}"
    if importlib.util.find_spec(name) is None:
        return None
    return importlib.import_module(name)
