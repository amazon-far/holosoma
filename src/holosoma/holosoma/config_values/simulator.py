import dataclasses

from holosoma.config_types.isaac import IsaacSimRenderConfig
from holosoma.config_types.simulator import (
    MujocoBackend,
    PhysxConfig,
    SimEngineConfig,
    SimulatorConfig,
    SimulatorInitConfig,
)
from holosoma.utils.config_registry import ConfigRegistry

SIMULATOR_REGISTRY = ConfigRegistry(SimulatorConfig, group="holosoma.config.simulator")

isaacsim = SIMULATOR_REGISTRY.add(
    "isaacsim",
    SimulatorConfig(
        _target_="holosoma.simulator.isaacsim.isaacsim.IsaacSim",
        _recursive_=False,
        config=SimulatorInitConfig(
            name="isaacsim",
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
                render_interval=4,
            ),
        ),
    ),
)


isaacsim6 = SIMULATOR_REGISTRY.add(
    "isaacsim6",
    dataclasses.replace(
        isaacsim,
        config=dataclasses.replace(isaacsim.config, isaacsim=IsaacSimRenderConfig(runtime="6.x")),
    ),
)
isaacsim6_newton = SIMULATOR_REGISTRY.add(
    "isaacsim6_newton",
    dataclasses.replace(
        isaacsim6,
        config=dataclasses.replace(
            isaacsim6.config,
            isaacsim=IsaacSimRenderConfig(runtime="6.x", physics="newton", solver="mjwarp", renderer="newton_renderer"),
        ),
    ),
)
isaacsim6_ovphysx = SIMULATOR_REGISTRY.add(
    "isaacsim6_ovphysx",
    dataclasses.replace(
        isaacsim6,
        config=dataclasses.replace(
            isaacsim6.config,
            isaacsim=IsaacSimRenderConfig(runtime="6.x", physics="ovphysx", renderer="ovrtx"),
        ),
    ),
)


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
                render_interval=1,
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
                render_interval=1,
            ),
            mujoco_backend=MujocoBackend.WARP,  # GPU-accelerated backend
        ),
    ),
)

from holosoma.utils.config_registry import (  # noqa: E402
    deprecated_defaults_alias as _deprecated_defaults_alias,
)

__getattr__ = _deprecated_defaults_alias(__name__, SIMULATOR_REGISTRY)
