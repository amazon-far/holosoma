"""Run one side of an MJWarp-to-MJWarp co-simulation and save its last camera frame and state.

usage: cosim_pair_roles.py {source,receiver} --endpoint ipc://... --output DIR [--frames N] [--num-envs N] [--loopback]
The receiver writes ``DIR/ready`` once it accepts connections and exits after the source closes.

With ``--loopback`` both sides add an rgb + depth camera ``loopback`` beside ``overview`` at the same pose; the
source's is remote.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any


def config(num_envs: int, loopback: bool = False, *, endpoint: str | None = None) -> Any:
    """The scene both sides run; with ``endpoint``, the source's, publishing there through the plugin."""
    from holosoma.config_types.plugin import CosimRenderPluginConfig, PluginConfig
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
        cameras = {"overview": camera, "loopback": replace(camera, remote=None if endpoint is None else "cosim")}
    plugins: dict[str, PluginConfig] = {} if endpoint is None else {"cosim": CosimRenderPluginConfig(endpoint=endpoint)}
    run = replace(run, training=replace(run.training, num_envs=num_envs, headless=True), sensor=cameras, plugin=plugins)
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
    from holosoma.cosim.render.renderer import RenderHandler, create_render_task
    from holosoma.simulator.base_simulator.hooks import Phase

    saved: dict[str, Any] = {}

    with create_render_task(config(num_envs, loopback)) as task:
        sim = task.simulator
        sim.hooks.add(Phase.FRAME_END, lambda: capture(sim, saved), after=sim.sensor_render_hook)
        receiver_loop = Receiver(RenderHandler(task), endpoint=endpoint)
        (output / "ready").write_text("ready\n")
        receiver_loop.run()
        frames = task.frames_rendered
    torch.save({**saved, "frames": frames}, output / "receiver.pt")


def source(endpoint: str, output: Path, frames: int, num_envs: int, loopback: bool) -> None:
    import torch

    from holosoma.simulator.base_simulator.hooks import Phase

    saved: dict[str, Any] = {}
    with config(num_envs, loopback, endpoint=endpoint).create_task() as task:
        sim = task.simulator
        cosim = sim.installed_plugins["cosim"]
        sim.hooks.add(Phase.FRAME_END, lambda: capture(sim, saved), after=sim.sensor_render_hook)
        task.reset_envs_idx()
        for _ in range(frames):
            task.step()
        saved["frames"] = cosim.publisher.sequence
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
