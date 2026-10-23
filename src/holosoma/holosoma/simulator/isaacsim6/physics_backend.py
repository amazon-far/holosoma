"""Build native Isaac Lab 3.0 physics configurations and clone declarations.

PhysX reads Holosoma's existing solver settings; Newton reads isaac_newton.
Engine packages are imported lazily. Scene replication includes the authored
assets, sensors, lights and world-mounted sensor frames in one native ClonePlan."""

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


def replicate_environments(config: SimulatorInitConfig, *, scene: Any, global_paths: tuple[str, ...]) -> None:
    """Publish and replicate one native ClonePlan for the authored source environment."""
    import isaaclab.cloner as cloner
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.cloner.usd import UsdReplicateContext
    from isaaclab.sensors import RayCasterCfg

    asset_cfgs = [asset.cfg for asset in (*scene.articulations.values(), *scene.rigid_objects.values())]
    sensor_cfgs = [sensor.cfg for sensor in scene.sensors.values()]
    cfgs = [*asset_cfgs, *sensor_cfgs]
    cfgs.extend(AssetBaseCfg(prim_path=path, spawn=None) for path in global_paths)
    # Include authored visual fixtures that are not physics assets or sensors.
    env_root = scene.stage.GetPrimAtPath("/World/envs/env_0")
    prefix = scene.env_regex_ns + "/"
    # Sensor views with no spawner do not own clone sources. Their authored world
    # attachment frames still need a USD declaration, just like light fixtures.
    owned_cfgs = [
        *asset_cfgs,
        *(cfg for cfg in sensor_cfgs if not isinstance(cfg, RayCasterCfg) and getattr(cfg, "spawn", None) is not None),
    ]
    declared = {cfg.prim_path[len(prefix) :].split("/", 1)[0] for cfg in owned_cfgs if cfg.prim_path.startswith(prefix)}
    for child in env_root.GetChildren():
        if child.GetName() not in declared:
            # Already-authored fixtures have no spawner to declare their representation.
            # Route their copies explicitly to the native USD clone context.
            cfgs.append(
                AssetBaseCfg(
                    prim_path=f"{scene.env_regex_ns}/{child.GetName()}",
                    spawn=None,
                    cloning_contexts=(UsdReplicateContext,),
                )
            )
    plan = cloner.clone_plan_from_env_0(
        scene.cfg.clone_cfg, cfgs, scene.num_envs, scene.cfg.env_spacing, positions=scene.env_origins.cpu().numpy()
    )
    cloner.replicate(plan, replicate_physics=scene.cfg.replicate_physics)


def _physx_cfg(config: SimulatorInitConfig) -> Any:
    """Isaac Sim PhysX with Holosoma's existing solver parameters."""
    from isaaclab_physx.physics import PhysxCfg

    physx = config.sim.physx
    return PhysxCfg(
        bounce_threshold_velocity=physx.bounce_threshold_velocity,
        solver_type=physx.solver_type,
        max_position_iteration_count=physx.num_position_iterations,
        max_velocity_iteration_count=physx.num_velocity_iterations,
        gpu_max_rigid_patch_count=10 * 2**15,
        enable_external_forces_every_iteration=False,
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
