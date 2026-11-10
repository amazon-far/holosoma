"""Translate Holosoma's shared MuJoCo options without importing an SDK."""

from __future__ import annotations

from dataclasses import fields
from typing import Any

from holosoma.config_types.mujoco import MujocoConfig
from holosoma.config_types.simulator import IsaacLab3NewtonConfig


def configured(config: Any) -> dict[str, Any]:
    """Keep explicit zero/False values while leaving SDK defaults untouched."""
    return {f.name: value for f in fields(config) if (value := getattr(config, f.name)) is not None}


def solver_settings(shared: MujocoConfig, newton: IsaacLab3NewtonConfig) -> dict[str, Any]:
    """Map shared solver controls and reject unsupported explicit settings."""
    if shared.solver == "pgs":
        raise ValueError("sim.mujoco.solver='pgs' is unsupported by Newton MJWarp; use cg or newton")
    if shared.integrator == "implicit":
        raise ValueError("sim.mujoco.integrator='implicit' is unsupported by Newton MJWarp; use implicitfast")
    if shared.noslip_iterations not in (None, 0):
        raise ValueError("sim.mujoco.noslip_iterations: Newton MJWarp supports only zero")
    if shared.warp.naconmax is not None:
        raise ValueError(
            "sim.mujoco.warp.naconmax: Newton's solver does not expose total-pool allocation; use nconmax_per_env"
        )
    result = {
        name: getattr(shared, name)
        for name in (
            "integrator",
            "solver",
            "cone",
            "jacobian",
            "impratio",
            "iterations",
            "tolerance",
            "ls_iterations",
            "ls_tolerance",
        )
        if getattr(shared, name) is not None
    }
    # Newton's automatic estimate underallocates contacts in configured G1
    # scenes. Keep this backend default out of the shared user configuration.
    result["nconmax"] = shared.warp.nconmax_per_env if shared.warp.nconmax_per_env is not None else 128
    if shared.warp.njmax_per_env is not None:
        result["njmax"] = shared.warp.njmax_per_env
    result["use_mujoco_contacts"] = newton.collision_source == "mujoco"
    result["disable_sensors"] = newton.deterministic_mode != "not_guaranteed"
    return result
