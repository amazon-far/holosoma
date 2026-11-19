import dataclasses

from holosoma.config_types.simulator import (
    BridgeConfig,
    MujocoBackend,
    PhysxConfig,
    SimEngineConfig,
    SimulatorConfig,
    SimulatorInitConfig,
    VirtualGantryCfg,
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
            ),
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


def _interactive(preset: SimulatorConfig, fps: int) -> SimulatorConfig:
    """``preset`` with the robot bridge and virtual gantry on, stepping physics at ``fps``."""
    return dataclasses.replace(
        preset,
        config=dataclasses.replace(
            preset.config,
            bridge=BridgeConfig(enabled=True),
            virtual_gantry=VirtualGantryCfg(enabled=True),
            sim=dataclasses.replace(preset.config.sim, fps=fps),
        ),
    )


# Interactive simulation (the default `sim` task): an external controller drives the robot over the
# bridge, and the virtual gantry holds it until released.
sim_isaacsim = SIMULATOR_REGISTRY.add("sim-isaacsim", _interactive(isaacsim, 200))
sim_mujoco = SIMULATOR_REGISTRY.add("sim-mujoco", _interactive(mujoco, 2000))
sim_mjwarp = SIMULATOR_REGISTRY.add("sim-mjwarp", _interactive(mjwarp, 400))

from holosoma.utils.config_registry import (  # noqa: E402
    deprecated_defaults_alias as _deprecated_defaults_alias,
)

__getattr__ = _deprecated_defaults_alias(__name__, SIMULATOR_REGISTRY)
