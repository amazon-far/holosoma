import dataclasses

from holosoma.config_types.simulator import (
    IsaacLabBackend,
    MujocoBackend,
    PhysxConfig,
    SimEngineConfig,
    SimulatorConfig,
    SimulatorInitConfig,
)
from holosoma.utils.config_registry import ConfigRegistry

SIMULATOR_REGISTRY = ConfigRegistry(SimulatorConfig, group="holosoma.config.simulator")

isaaclab2 = SIMULATOR_REGISTRY.add(
    "isaaclab2",
    SimulatorConfig(
        _target_="holosoma.simulator.isaacsim.isaacsim.IsaacSim",
        _recursive_=False,
        config=SimulatorInitConfig(
            name="isaaclab2",
            isaaclab_backend=IsaacLabBackend.ISAACLAB2,
            sim=SimEngineConfig(
                fps=200,
                control_decimation=4,
                substeps=1,
                physx=PhysxConfig(
                    solver_type=1,
                    num_position_iterations=8,
                    num_velocity_iterations=4,
                    bounce_threshold_velocity=0.5,
                ),
                render_mode="human",
            ),
        ),
    ),
)

isaaclab3 = SIMULATOR_REGISTRY.add(
    "isaaclab3",
    dataclasses.replace(
        isaaclab2,
        config=dataclasses.replace(isaaclab2.config, name="isaaclab3", isaaclab_backend=IsaacLabBackend.ISAACLAB3),
    ),
)

# Keep released command lines and external preset imports selecting Lab 2.
isaacsim = SIMULATOR_REGISTRY.add("isaacsim", isaaclab2)


mujoco = SIMULATOR_REGISTRY.add(
    "mujoco",
    SimulatorConfig(
        _target_="holosoma.simulator.mujoco.mujoco.MuJoCo",
        _recursive_=False,
        config=SimulatorInitConfig(
            name="mujoco",
            sim=SimEngineConfig(
                fps=200,
                control_decimation=4,
                substeps=1,
                physx=PhysxConfig(
                    solver_type=1,
                    num_position_iterations=4,
                    num_velocity_iterations=0,
                    bounce_threshold_velocity=0.5,
                ),
                render_mode="fake",
            ),
            mujoco_backend=MujocoBackend.CLASSIC,  # Explicit for clarity
        ),
    ),
)


mjwarp = SIMULATOR_REGISTRY.add(
    "mjwarp",
    SimulatorConfig(
        _target_="holosoma.simulator.mujoco.mujoco.MuJoCo",
        _recursive_=False,
        config=SimulatorInitConfig(
            name="mujoco",
            sim=SimEngineConfig(
                fps=200,
                control_decimation=4,
                substeps=1,
                physx=PhysxConfig(
                    solver_type=1,
                    num_position_iterations=4,
                    num_velocity_iterations=0,
                    bounce_threshold_velocity=0.5,
                ),
                render_mode="fake",
            ),
            mujoco_backend=MujocoBackend.WARP,  # GPU-accelerated backend
        ),
    ),
)

from holosoma.utils.config_registry import (  # noqa: E402
    deprecated_defaults_alias as _deprecated_defaults_alias,
)

__getattr__ = _deprecated_defaults_alias(__name__, SIMULATOR_REGISTRY)
