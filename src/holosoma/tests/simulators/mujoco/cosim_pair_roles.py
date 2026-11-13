"""Run one side of an MJWarp-to-MJWarp co-simulation and save its last camera frame and state.

usage: cosim_pair_roles.py {source,receiver} --endpoint ipc://... --output DIR [--frames N] [--num-envs N]
The receiver writes ``DIR/ready`` once it accepts connections and exits after the source closes.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

FIELDS = ("robot_root_states", "dof_pos", "dof_vel", "object_root_states")


def config(num_envs: int) -> Any:
    from holosoma.config_types.run_sim import RunSimConfig
    from holosoma.config_values import sensor, simulator
    from holosoma.utils.simulator_config import set_simulator_type

    preset = simulator.mjwarp
    init = preset.config
    init = replace(
        init,
        sim=replace(init.sim, fps=200, control_decimation=1),
        bridge=replace(init.bridge, enabled=False),
        virtual_gantry=replace(init.virtual_gantry, enabled=False),
    )
    run = RunSimConfig(simulator=replace(preset, config=init), device="cuda:0", time_scale=0)
    camera = replace(sensor.overview_camera, width=64, height=48, update_decimation=1)
    run = replace(run, training=replace(run.training, num_envs=num_envs, headless=True), sensor={"overview": camera})
    set_simulator_type(run.simulator)
    return run


def capture(sim: Any, saved: dict[str, Any]) -> None:
    saved["rgb"] = sim.get_camera_data("overview", "rgb").clone().cpu()
    saved["robot_root_states"] = sim.robot_root_states.clone().cpu()
    saved["dof_pos"] = sim.dof_pos.clone().cpu()


def receiver(endpoint: str, output: Path, num_envs: int) -> None:
    import torch

    from holosoma.cosim.bridge import Receiver
    from holosoma.simulator.base_simulator.hooks import Phase

    saved: dict[str, Any] = {}
    with config(num_envs).create_render_task() as task:
        sim = task.simulator
        sim.hooks.add(Phase.FRAME_END, lambda: capture(sim, saved), after=sim.sensor_render_hook)
        receiver_loop = Receiver(task, endpoint=endpoint)
        (output / "ready").write_text("ready\n")
        receiver_loop.run()
        saved["frames"] = task.frames_rendered
    torch.save(saved, output / "receiver.pt")


def source(endpoint: str, output: Path, frames: int, num_envs: int) -> None:
    import torch

    from holosoma.cosim.bridge import Publisher
    from holosoma.cosim.cuda import CacheAllocator
    from holosoma.simulator.base_simulator.hooks import Phase

    saved: dict[str, Any] = {}
    allocator = CacheAllocator(FIELDS)
    with config(num_envs).create_task(buffer_allocator=allocator) as task:
        sim = task.simulator
        publisher = Publisher(sim, allocator, endpoint=endpoint)
        sim.hooks.add(Phase.FRAME_END, lambda: capture(sim, saved), after=publisher.capture_hook)
        task.reset_envs_idx()
        for _ in range(frames):
            task.step()
        saved["frames"] = publisher.sequence
    torch.save(saved, output / "source.pt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=["source", "receiver"])
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--num-envs", type=int, default=2)
    args = parser.parse_args()
    if args.role == "receiver":
        receiver(args.endpoint, args.output, args.num_envs)
    else:
        source(args.endpoint, args.output, args.frames, args.num_envs)


if __name__ == "__main__":
    main()
