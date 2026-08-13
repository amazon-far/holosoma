"""
Configuration values for run_sim.py - sim2sim-optimized simulator defaults.

These configurations extend the base simulator configs with sim2sim-specific settings:
- Bridge enabled by default
- Virtual gantry enabled
- High FPS for real-time robot control
- DOF force sensors enabled (for IsaacGym)
"""

import dataclasses

import holosoma.config_values.simulator
from holosoma.config_types.simulator import BridgeConfig, SimulatorConfig, VirtualGantryCfg
from holosoma.utils.config_registry import ConfigRegistry

RUN_SIM_REGISTRY = ConfigRegistry(SimulatorConfig, group="holosoma.config.run_sim")

# IsaacGym with sim2sim optimizations
isaacgym = RUN_SIM_REGISTRY.add(
    "isaacgym",
    dataclasses.replace(
        holosoma.config_values.simulator.isaacgym,
        config=dataclasses.replace(
            holosoma.config_values.simulator.isaacgym.config,
            bridge=BridgeConfig(enabled=True),
            virtual_gantry=VirtualGantryCfg(enabled=True),
            sim=dataclasses.replace(
                holosoma.config_values.simulator.isaacgym.config.sim,
                fps=1000,  # Gym on CPU can reach ~900 on 4090 (for sim2sim i.e, num_envs=1)
                physx=dataclasses.replace(
                    holosoma.config_values.simulator.isaacgym.config.sim.physx,
                    enable_dof_force_sensors=True,  # required for force controls
                ),
            ),
        ),
    ),
)

# IsaacSim with sim2sim optimizations
isaacsim = RUN_SIM_REGISTRY.add(
    "isaacsim",
    dataclasses.replace(
        holosoma.config_values.simulator.isaacsim,
        config=dataclasses.replace(
            holosoma.config_values.simulator.isaacsim.config,
            bridge=BridgeConfig(enabled=True),
            virtual_gantry=VirtualGantryCfg(enabled=True),
            sim=dataclasses.replace(
                holosoma.config_values.simulator.isaacsim.config.sim,
                fps=200,
            ),
        ),
    ),
)

# MuJoCo with sim2sim optimizations
mujoco = RUN_SIM_REGISTRY.add(
    "mujoco",
    dataclasses.replace(
        holosoma.config_values.simulator.mujoco,
        config=dataclasses.replace(
            holosoma.config_values.simulator.mujoco.config,
            bridge=BridgeConfig(enabled=True),
            virtual_gantry=VirtualGantryCfg(enabled=True),
            sim=dataclasses.replace(
                holosoma.config_values.simulator.mujoco.config.sim,
                fps=2000,  # mujoco can run faster
            ),
        ),
    ),
)

# MuJoCo Warp with sim2sim optimizations
mjwarp = RUN_SIM_REGISTRY.add(
    "mjwarp",
    dataclasses.replace(
        holosoma.config_values.simulator.mjwarp,
        config=dataclasses.replace(
            holosoma.config_values.simulator.mjwarp.config,
            bridge=BridgeConfig(enabled=True),
            virtual_gantry=VirtualGantryCfg(enabled=True),
            sim=dataclasses.replace(
                holosoma.config_values.simulator.mjwarp.config.sim,
                fps=400,
            ),
        ),
    ),
)

from holosoma.utils.config_registry import (  # noqa: E402
    deprecated_defaults_alias as _deprecated_defaults_alias,
)

__getattr__ = _deprecated_defaults_alias(__name__, RUN_SIM_REGISTRY)
