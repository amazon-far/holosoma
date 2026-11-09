"""Data-only SDK prototype; safe to construct before Kit chooses its USD runtime."""

from __future__ import annotations

from typing import Any

from isaaclab.utils import configclass
from isaaclab_newton.physics import MJWarpSolverCfg


@configclass
class HolosomaMJWarpSolverCfg(MJWarpSolverCfg):
    """Private native config carrying Holosoma's already typed common options."""

    class_type: str = "holosoma.simulator.isaacsim.backends.newton_mjwarp:NewtonMJWarpManagerAdapter"
    jacobian: str | None = None
    ls_tolerance: float | None = None
    # SDK configclass rewrites nested objects in place. Keep a private plain
    # mapping here so it cannot attempt to mutate Holosoma's frozen dataclasses.
    shared_config: dict[str, Any] = {}
