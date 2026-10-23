"""Building Isaac Lab 3.0's ``SimulationCfg.physics`` for the configured engine.

Isaac Lab 2.3.2 had one physics engine, configured through ``SimulationCfg.physx``. Isaac Lab 3.0
replaced that field with ``physics: PhysicsCfg | None`` and resolves a concrete engine from it, so
the engine is now a configuration choice: PhysX (``isaaclab_physx``), Newton
(``isaaclab_newton``) or OmniPhysX (``isaaclab_ov``).

This module is the one place that knows how holosoma's configuration maps onto each engine's own
config object, so the rest of the lane never names an engine.

Why the engines do not share a tuning struct
--------------------------------------------
holosoma's :class:`~holosoma.config_types.simulator.PhysxConfig` is genuinely PhysX-shaped.
Measured against the pinned Isaac Lab 3.0: ``isaaclab_newton.physics.NewtonCfg`` shares **none**
of its fields (no ``solver_type``, no position/velocity iteration counts, no
``bounce_threshold_velocity``; it has substepping and solver sub-configs instead), and
``isaaclab_ov.physics.OvPhysxCfg`` shares only two of six. Mapping one struct onto all three would
mean silently dropping whichever fields the selected engine does not understand, so each engine
reads its own config and supplying the wrong one is an error (see :mod:`.capabilities`).

Import timing
-------------
The engine packages are imported lazily, inside the branch that needs them. Only the selected
engine's package is touched, so a run does not pay for -- or fail on -- an engine it is not using.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from holosoma.config_types.simulator import IsaacPhysicsBackend

if TYPE_CHECKING:
    import torch
    from pxr import Usd

    from holosoma.config_types.simulator import SimulatorInitConfig


def build_physics_cfg(config: SimulatorInitConfig) -> Any:
    """The ``SimulationCfg.physics`` value for ``config``'s selected engine.

    Parameters
    ----------
    config
        The simulator init config. ``isaac_physics_backend`` selects the engine; ``sim.physx`` or
        ``isaac_newton`` supplies that engine's tuning.

    Returns
    -------
    Any
        An ``isaaclab.physics.physics_manager_cfg.PhysicsCfg`` subclass instance, ready to hand to
        ``SimulationCfg(physics=...)``.
    """
    backend = config.isaac_physics_backend
    if backend is IsaacPhysicsBackend.PHYSX:
        return _physx_cfg(config)
    if backend is IsaacPhysicsBackend.NEWTON:
        return _newton_cfg(config)
    if backend is IsaacPhysicsBackend.OVPHYSX:
        return _ovphysx_cfg()
    raise ValueError(f"Unhandled Isaac physics backend {backend!r}.")


def use_newton_actuators_for(config: SimulatorInitConfig) -> bool:
    """Whether to route actuators through Newton's actuator model.

    New in Isaac Lab 3.0 and **defaults to True**, which silently changes actuator behaviour
    relative to Isaac Sim 5.1. Derived from the engine rather than pinned:

    * Under PhysX, ``False`` -- the PhysX actuator path is what the 5.1 lane uses, so this keeps
      the two generations numerically comparable. Leaving 3.0's default would make a run that
      *looks* like a PhysX run quietly use a different actuator model.
    * Under Newton, ``True`` -- it is the only actuator path Newton has.
    """
    return config.isaac_physics_backend is IsaacPhysicsBackend.NEWTON


def replicate_environments(
    config: SimulatorInitConfig,
    *,
    stage: Usd.Stage,
    source: str,
    destination: str,
    env_ids: npt.NDArray[np.integer[Any]],
    env_origins: torch.Tensor,
    global_paths: tuple[str, ...],
) -> None:
    """Replicate ``source`` into every environment, for the configured engine.

    ``cloner.usd_replicate`` copies prim specs on the USD stage. That is the whole job on PhysX,
    which parses its physics model *from* the stage -- but Newton does not. Newton builds its
    model from a ClonePlan, so for it the USD copy is only the visual half and its ``ModelBuilder``
    would stay empty; the first thing to read that empty model (contact-sensor site injection)
    then fails with ``Site 'ft_0' ... matched no builder bodies``, which reads like a sensor
    problem rather than an unpopulated model.

    So Newton gets ``isaaclab_newton.cloner.newton_physics_replicate`` as well -- the Newton
    counterpart of ``usd_replicate``, taking the same source/destination/env_ids plus a boolean
    source-to-environment ``mapping``. Both run: the USD clones are still needed for rendering and
    for the RTX sensors, which read the stage no matter which engine owns the physics.

    Ordering matters. The Newton import reads ``source`` as its prototype, so it must run *before*
    the USD copy exists -- otherwise its stage-wide import sees all N environments and double
    counts them against its own per-world replication.
    """
    # Imported here rather than at module scope, matching this module's lazy-import rule: an
    # eager `isaaclab` import reaches `omni.physics.tensors`, which exists only inside a running
    # Kit runtime, so it makes the module unimportable in every other environment.
    import isaaclab.cloner as cloner

    if config.isaac_physics_backend is IsaacPhysicsBackend.NEWTON:
        from isaaclab_newton.cloner import newton_physics_replicate

        newton_physics_replicate(
            stage,
            [source],
            [destination],
            env_ids,
            # One source routed to every environment. `usd_replicate` defaults this (`mask=None`
            # means "all"); the Newton entry point has no default, so it is spelled out.
            np.ones((1, len(env_ids)), dtype=bool),
            # numpy, not torch: this path reaches `np.unique`/`wp.transform` directly and a CUDA
            # tensor raises "can't convert cuda:0 device type tensor to numpy" deep inside.
            positions=env_origins.cpu().numpy(),
            global_paths=global_paths,
        )

    cloner.usd_replicate(stage, [source], [destination], env_ids, positions=env_origins)


def _physx_cfg(config: SimulatorInitConfig) -> Any:
    """PhysX, tuned from holosoma's ``PhysxConfig``. The verified path."""
    from isaaclab.physics.physics_manager_cfg import PhysxAutoCfg
    from isaaclab_physx.physics import PhysxCfg

    physx = config.sim.physx
    # `PhysxAutoCfg` resolves the concrete PhysX implementation at launch while `isaacsim_physx`
    # carries the tuning. Every keyword below kept its 2.3.2 name through the move to
    # `isaaclab_physx.physics.PhysxCfg`, so the values are unchanged from the 5.1 lane.
    return PhysxAutoCfg(
        isaacsim_physx=PhysxCfg(
            bounce_threshold_velocity=physx.bounce_threshold_velocity,
            solver_type=physx.solver_type,
            max_position_iteration_count=physx.num_position_iterations,
            max_velocity_iteration_count=physx.num_velocity_iterations,
            gpu_max_rigid_patch_count=10 * 2**15,
            # Pinned, not inherited: 2.3.2 defaults this False and 3.0 defaults it True. holosoma
            # applies external pushes (agents/callbacks/push.py, payload.py), so accepting the new
            # default would change push dynamics relative to the 5.1 lane -- a divergence visible
            # only as different numbers.
            enable_external_forces_every_iteration=False,
        )
    )


def _newton_cfg(config: SimulatorInitConfig) -> Any:
    """Newton, tuned from ``isaac_newton``. Fields left ``None`` keep Isaac Lab's default."""
    from isaaclab_newton.physics import NewtonCfg

    newton = config.isaac_newton
    kwargs = {
        "num_substeps": newton.num_substeps,
        "collision_decimation": newton.collision_decimation,
        "use_cuda_graph": newton.use_cuda_graph,
        "deterministic_mode": newton.deterministic_mode,
    }
    # Omit rather than pass None: None is a legitimate value for some Isaac Lab fields, so
    # forwarding it would override a real default with "unset".
    return NewtonCfg(**{k: v for k, v in kwargs.items() if v is not None})


def _ovphysx_cfg() -> Any:
    """OmniPhysX at its defaults.

    No holosoma tuning surface yet: ``OvPhysxCfg`` shares only ``gpu_max_rigid_patch_count`` and
    ``enable_external_forces_every_iteration`` with holosoma's PhysX config, which is too little
    to be worth a config type before anyone runs it. Supplying ``sim.physx`` with this engine is
    rejected in :mod:`.capabilities` rather than silently half-applied.
    """
    from isaaclab_ov.physics import OvPhysxCfg

    return OvPhysxCfg()
