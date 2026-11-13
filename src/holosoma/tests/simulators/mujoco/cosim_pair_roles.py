"""Run one side of an MJWarp-to-MJWarp co-simulation and save its last camera frame and state.

usage: cosim_pair_roles.py {source,receiver} --endpoint ipc://... --output DIR [--frames N] [--num-envs N] [--loopback]
The receiver writes ``DIR/ready`` once it accepts connections and exits after the source closes.

With ``--loopback`` both sides add an rgb + depth camera ``loopback`` beside ``overview`` at the same pose; the
source's is remote. After its frames the source resets env 0 and re-renders it.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

FIELDS = ("robot_root_states", "dof_pos", "dof_vel", "object_root_states")


def config(num_envs: int, loopback: bool = False, *, source: bool = False) -> Any:
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
    cameras: dict[str, Any] = {"overview": camera}
    if loopback:
        camera = replace(camera, data_types=["rgb", "depth"])
        cameras = {"overview": camera, "loopback": replace(camera, remote=source)}
    run = replace(run, training=replace(run.training, num_envs=num_envs, headless=True), sensor=cameras)
    set_simulator_type(run.simulator)
    return run


def capture(sim: Any, saved: dict[str, Any]) -> None:
    for name in sim.sensor_manager.camera_names:
        for data_type in sim.sensor_manager.get(name).config.data_types:
            saved[f"{name}/{data_type}"] = sim.get_camera_data(name, data_type).clone().cpu()
    saved["robot_root_states"] = sim.robot_root_states.clone().cpu()
    saved["dof_pos"] = sim.dof_pos.clone().cpu()


def receiver(endpoint: str, output: Path, num_envs: int, loopback: bool) -> None:
    import torch

    from holosoma.cosim.bridge import Receiver
    from holosoma.simulator.base_simulator.hooks import Phase

    captures: list[dict[str, Any]] = []

    def keep() -> None:
        captures[:] = [*captures[-1:], {}]
        capture(sim, captures[-1])

    with config(num_envs, loopback).create_render_task() as task:
        sim = task.simulator
        sim.hooks.add(Phase.FRAME_END, keep, after=sim.sensor_render_hook)
        receiver_loop = Receiver(task, endpoint=endpoint)
        (output / "ready").write_text("ready\n")
        receiver_loop.run()
        frames = task.frames_rendered
    torch.save({**captures[-1], "previous": captures[0], "frames": frames}, output / "receiver.pt")


def source(endpoint: str, output: Path, frames: int, num_envs: int, loopback: bool) -> None:
    import torch

    from holosoma.cosim.bridge import Publisher
    from holosoma.cosim.cuda import CacheAllocator
    from holosoma.simulator.base_simulator.hooks import Phase

    saved: dict[str, Any] = {}
    allocator = CacheAllocator(FIELDS)
    with config(num_envs, loopback, source=True).create_task(buffer_allocator=allocator) as task:
        sim = task.simulator
        publisher = Publisher(sim, allocator, endpoint=endpoint)
        after = sim.sensor_render_hook if loopback else publisher.capture_hook
        sim.hooks.add(Phase.FRAME_END, lambda: capture(sim, saved), after=after)
        task.reset_envs_idx()
        for _ in range(frames):
            task.step()
        saved["frames"] = publisher.sequence
        if loopback:
            env_ids = torch.tensor([0], device=sim.sim_device)
            task.reset_envs_idx(env_ids)
            saved["stale"] = sim.get_camera_data("loopback", "rgb").clone().cpu()
            sim.rerender_sensors(env_ids)
            sim.backend.render_cameras([sim.sensor_manager.get("overview")])
            capture(sim, saved.setdefault("reset", {}))
    torch.save(saved, output / "source.pt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=["source", "receiver"])
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--loopback", action="store_true")
    args = parser.parse_args()
    if args.role == "receiver":
        receiver(args.endpoint, args.output, args.num_envs, args.loopback)
    else:
        source(args.endpoint, args.output, args.frames, args.num_envs, args.loopback)


if __name__ == "__main__":
    main()
