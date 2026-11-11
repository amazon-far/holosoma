"""SDK adapter, imported after the selected runtime has been launched.

The SDK's manager hook applies options before its first step and CUDA capture.
Its public solver config omits jacobian and ls_tolerance even though the pinned
Newton solver accepts them; keep that SDK detail out of user configuration.
"""

from __future__ import annotations

from typing import Any

import mujoco
from newton.solvers import SolverMuJoCo
from isaaclab_newton.physics.mjwarp_manager import NewtonMJWarpManager

from holosoma.config_types.mujoco import MujocoConfig
from holosoma.simulator.shared.mujoco_warp_options import (
    apply_warp_model,
    validate_warp_options,
    warp_extra_inputs,
    warp_option_overrides,
)
from holosoma.simulator.shared.mujoco_options import native_value
from .newton_mjwarp_cfg import HolosomaMJWarpSolverCfg


class _ConfiguredSolver(SolverMuJoCo):
    """Pass flag masks into native compilation before MJWarp allocates buffers.

    Newton exposes separate contact/sensor toggles, but no complete MuJoCo flag
    mask. Applying the mask after construction leaves compilation and allocations
    stale, so this override supplies it at the SDK's conversion boundary.
    """

    def __init__(self, model: Any, *, shared_config: MujocoConfig, **kwargs: Any):
        self._holosoma_config = shared_config
        if shared_config.disableflags is not None:
            mask = native_value("disableflags", shared_config.disableflags, "sim.mujoco.disableflags")
            sensor_disabled = bool(mask & int(mujoco.mjtDisableBit.mjDSBL_SENSOR))
            if kwargs.get("disable_sensors") and not sensor_disabled:
                raise ValueError("sim.mujoco.disableflags must include sensor when Newton determinism is requested")
            # These flags also control Newton's conversion/allocation policy.
            # Resolve them before the constructor computes its CCD margin and
            # contact workarounds; changing only MjOption would leave those stale.
            kwargs["enable_multiccd"] = not bool(mask & int(mujoco.mjtDisableBit.mjDSBL_MULTICCD))
            kwargs["disable_contacts"] = bool(mask & int(mujoco.mjtDisableBit.mjDSBL_CONTACT))
            kwargs["disable_sensors"] = sensor_disabled
        super().__init__(model, **kwargs)

    def _convert_to_mjc(self, model: Any, state: Any = None, **kwargs: Any) -> Any:
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
        return super()._convert_to_mjc(model, state, **kwargs)


class NewtonMJWarpManagerAdapter(NewtonMJWarpManager):
    """Apply shared MuJoCo controls before the SDK's first step/CUDA capture.

    The SDK selects managers by ``class_type`` and provides ``_create_solver``
    for solver construction. Reusing its argument filtering and normal manager
    lifecycle avoids copying Newton's stepping, allocation and graph logic.
    """

    @classmethod
    def _create_solver(cls, model: Any, solver_cfg: HolosomaMJWarpSolverCfg) -> Any:
        cfg = MujocoConfig(**solver_cfg.shared_config)
        kwargs = cls._filter_solver_kwargs(SolverMuJoCo, solver_cfg)
        kwargs.pop("ls_parallel", None)
        # Public MjOption controls accepted by the pinned Newton constructor.
        # Keep this boundary explicit instead of probing arbitrary signatures.
        constructor_options = {
            "ccd_iterations",
            "ccd_tolerance",
            "sdf_iterations",
            "sdf_initpoints",
            "density",
            "viscosity",
            "wind",
            "magnetic",
            "sleep_tolerance",
        }
        for name, value in cfg.extra.items():
            if name not in constructor_options:
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
