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
from holosoma.config_types.simulator import (
    BridgeConfig,
    ExternalPusherCfg,
    HardcodedPushEvent,
    VirtualGantryCfg,
)

# IsaacGym with sim2sim optimizations
isaacgym = dataclasses.replace(
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
)

# IsaacSim with sim2sim optimizations
isaacsim = dataclasses.replace(
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
)

# MuJoCo with sim2sim optimizations
#
# Hardcoded EE-push schedule is baked in so it's available at the CLI via
#   --simulator.config.external-pusher.mode hardcoded
# Default remains "randomized" to preserve existing behaviour.
_EE_PUSH_SCHEDULE = [
    HardcodedPushEvent(
        sim_time_s=0.0,
        body_name="left_wrist_yaw_link",
        force_vec=(0.0, 5.0, 0.0),
        duration_s=10.0,
    ),
    HardcodedPushEvent(
        sim_time_s=0.0,
        body_name="right_wrist_yaw_link",
        force_vec=(0.0, 5.0, 0.0),
        duration_s=10.0,
    ),
]

mujoco = dataclasses.replace(
    holosoma.config_values.simulator.mujoco,
    config=dataclasses.replace(
        holosoma.config_values.simulator.mujoco.config,
        bridge=BridgeConfig(enabled=True),
        virtual_gantry=VirtualGantryCfg(enabled=True),
        external_pusher=ExternalPusherCfg(
            enabled=True,
            mode="randomized",
            schedule=_EE_PUSH_SCHEDULE,
        ),
        sim=dataclasses.replace(
            holosoma.config_values.simulator.mujoco.config.sim,
            fps=2000,  # mujoco can run faster
        ),
    ),
)

# MuJoCo Warp with sim2sim optimizations
mjwarp = dataclasses.replace(
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
)

DEFAULTS = {
    "isaacgym": isaacgym,
    "isaacsim": isaacsim,
    "mujoco": mujoco,
    "mjwarp": mjwarp,
}
