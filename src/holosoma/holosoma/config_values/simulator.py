from holosoma.config_types.simulator import (
    MujocoBackend,
    PhysxConfig,
    SimEngineConfig,
    SimulatorConfig,
    SimulatorInitConfig,
)
from holosoma.utils.config_registry import ConfigRegistry

SIMULATOR_REGISTRY = ConfigRegistry(SimulatorConfig, group="holosoma.config.simulator")

isaacgym = SIMULATOR_REGISTRY.add(
    "isaacgym",
    SimulatorConfig(
        _target_="holosoma.simulator.isaacgym.isaacgym.IsaacGym",
        _recursive_=False,
        config=SimulatorInitConfig(
            name="isaacgym",
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
            ),
            contact_sensor_history_length=3,
        ),
    ),
)


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
            contact_sensor_history_length=3,
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
