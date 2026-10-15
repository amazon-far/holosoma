"""Bounded headless demonstration using the real shared simulator setup, with no plugins."""

from dataclasses import replace

import torch

from holosoma.config_types.run_sim import RunSimConfig
from holosoma.config_types.simulator import BridgeConfig, VirtualGantryCfg
from holosoma.utils.config_registry import parse_config
from holosoma.utils.sim_utils import setup_simulation_environment


def main():
    config = parse_config(RunSimConfig)
    config = replace(
        config,
        plugin={},
        training=replace(config.training, headless=True),
        simulator=replace(
            config.simulator,
            config=replace(
                config.simulator.config,
                bridge=BridgeConfig(enabled=False),
                virtual_gantry=VirtualGantryCfg(enabled=False),
            ),
        ),
    )
    backend = config.simulator.config.mujoco_backend
    device = "cuda:0" if backend == "warp" else "cpu"
    env, _, _ = setup_simulation_environment(config, device=device)
    sim = env.sim
    try:
        sim.set_headless(True)
        sim.setup()
        sim.setup_terrain()
        sim.load_assets()
        init = config.robot.init_state
        base = torch.tensor(init.pos + init.rot + init.lin_vel + init.ang_vel, device=device)
        sim.create_envs(config.training.num_envs, torch.zeros(config.training.num_envs, 3, device=device), base)
        sim.prepare_sim()
        assert sim.plugin_config == sim.installed_plugins == {}
        for _ in range(20):
            sim.backend.step()
        print(f"Native simulation time: {sim.backend.physics_time():.6f}s")
    finally:
        sim.close()


if __name__ == "__main__":
    main()
