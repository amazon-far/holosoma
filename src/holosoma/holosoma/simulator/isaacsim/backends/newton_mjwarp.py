"""SDK adapter, imported only after Newton physics has been selected.

The SDK's manager hook applies options before its first step and CUDA capture.
Its public solver config omits jacobian and ls_tolerance even though the pinned
Newton solver accepts them; keep that SDK detail out of user configuration.
"""

from __future__ import annotations

from inspect import signature
from typing import Any

import mujoco
from newton.solvers import SolverMuJoCo
from isaaclab.utils import configclass
from isaaclab_newton.physics import MJWarpSolverCfg
from isaaclab_newton.physics.mjwarp_manager import NewtonMJWarpManager

from holosoma.config_types.mujoco import MujocoConfig
from holosoma.simulator.shared.mujoco_warp_options import (
    apply_warp_model,
    validate_warp_options,
    warp_extra_inputs,
    warp_option_overrides,
)
from holosoma.simulator.shared.mujoco_options import native_value


@configclass
class HolosomaMJWarpSolverCfg(MJWarpSolverCfg):
    """Private native config carrying Holosoma's already typed common options."""

    class_type: str = "holosoma.simulator.isaacsim.backends.newton_mjwarp:NewtonMJWarpManagerAdapter"
    jacobian: str | None = None
    ls_tolerance: float | None = None
    # SDK configclass rewrites nested objects in place. Keep a private plain
    # mapping here so it cannot attempt to mutate Holosoma's frozen dataclasses.
    shared_config: dict[str, Any] = {}


class _ConfiguredSolver(SolverMuJoCo):
    def __init__(self, model: Any, *, shared_config: MujocoConfig, **kwargs: Any):
        self._holosoma_config = shared_config
        super().__init__(model, **kwargs)

    def _convert_to_mjc(self, model: Any, **kwargs: Any) -> None:
        # Flags must reach compilation/upload, rather than merely mutate the
        # CPU model after MJWarp has specialized kernels and allocated buffers.
        cfg = self._holosoma_config
        for name in ("disableflags", "enableflags"):
            value = getattr(cfg, name)
            if value is not None:
                mask = native_value(name, value, f"sim.mujoco.{name}")
                if name == "disableflags" and kwargs[name] & ~mask:
                    raise ValueError("sim.mujoco.disableflags conflicts with Newton's required solver flags")
                if name == "enableflags" and mask & int(mujoco.mjtEnableBit.mjENBL_SLEEP):
                    raise ValueError("sim.mujoco.enableflags: Holosoma's state resets do not support sleeping")
                kwargs[name] = mask
        super()._convert_to_mjc(model, **kwargs)


class NewtonMJWarpManagerAdapter(NewtonMJWarpManager):
    """Keep Newton's required name prefix for the SDK's backend dispatch."""

    @classmethod
    def _create_solver(cls, model: Any, solver_cfg: HolosomaMJWarpSolverCfg) -> Any:
        cfg = MujocoConfig(**solver_cfg.shared_config)
        kwargs = cls._filter_solver_kwargs(SolverMuJoCo, solver_cfg)
        kwargs.pop("ls_parallel", None)
        constructor_names = set(signature(SolverMuJoCo.__init__).parameters)
        for name, value in cfg.extra.items():
            if name not in constructor_names or not isinstance(getattr(mujoco.MjOption, name, None), property):
                raise ValueError(f"sim.mujoco.extra.{name}: not exposed by Newton's solver constructor")
            kwargs[name] = value
        option_inputs, allocations = warp_extra_inputs(cfg.warp)
        if allocations:
            raise ValueError(
                "sim.mujoco.warp.extras: Newton does not expose these allocation inputs: "
                + ", ".join(sorted(allocations))
            )
        options = warp_option_overrides(cfg.warp, inputs=option_inputs)
        solver = _ConfiguredSolver(model, shared_config=cfg, **kwargs)
        overrides = dict(cfg.extra)
        if cfg.noslip_iterations is not None:
            overrides["noslip_iterations"] = cfg.noslip_iterations
        validate_warp_options(solver.mj_model, overrides)
        apply_warp_model(solver.mjw_model, cfg, options)
        return solver
