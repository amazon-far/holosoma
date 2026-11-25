"""Run one side of a co-simulation with an MJWarp head and save its last camera frame and state.

usage: cosim_pair_roles.py {head,peer} --endpoint ipc://... --output DIR [--frames N] [--num-envs N]
       [--loopback] [--simulator {mjwarp,isaacsim}]
The peer writes ``DIR/ready`` once it accepts connections and exits after the head closes. ``--simulator``
selects the peer's backend; the head is always MJWarp.

The head resets every environment, then the last one a quarter of the way through. With ``--loopback`` both
sides add rgb + depth cameras ``loopback`` and ``still`` at ``overview``'s pose; the head's ``loopback`` is
remote, and halfway through the head moves ``overview`` and ``loopback`` the same way.
:func:`run_pair` runs both roles for tests.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[4]
_TIMEOUT_S = 900


def config(num_envs: int, loopback: bool = False, *, endpoint: str | None = None, backend: str = "mjwarp") -> Any:
    """The scene both sides run on ``backend``; with ``endpoint``, the head's, publishing there."""
    from holosoma.config_types.plugin import CosimRenderPluginConfig, PluginConfig
    from holosoma.config_types.run_sim import RunSimConfig
    from holosoma.config_values import sensor, simulator
    from holosoma.utils.simulator_config import set_simulator_type

    preset = getattr(simulator, backend)
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
        remote = replace(camera, remote=None if endpoint is None else "cosim")
        cameras = {"overview": camera, "loopback": remote, "still": camera}
    plugins: dict[str, PluginConfig] = {} if endpoint is None else {"cosim": CosimRenderPluginConfig(endpoint=endpoint)}
    run = replace(run, training=replace(run.training, num_envs=num_envs, headless=True), sensor=cameras, plugin=plugins)
    set_simulator_type(run.simulator)
    return run


def capture(sim: Any, saved: dict[str, Any]) -> None:
    """Save the latest frame's cameras and state into ``saved``, and the first frame's into ``saved["first"]``."""
    latest: dict[str, Any] = {}
    for name in sim.sensor_manager.camera_names:
        for data_type in sim.sensor_manager.get(name).config.data_types:
            latest[f"{name}/{data_type}"] = sim.get_camera_data(name, data_type).clone().cpu()
    latest["robot_root_states"] = sim.robot_root_states.clone().cpu()
    latest["dof_pos"] = sim.dof_pos.clone().cpu()
    latest["body_pos"] = sim.rigid_body_pos_w.clone().cpu()
    latest["body_rot"] = sim.rigid_body_quat_w.clone().cpu()
    latest["body_names"] = list(sim.body_names)
    saved.update(latest)
    saved.setdefault("first", latest)


def record_episodes(sim: Any, saved: dict[str, Any]) -> None:
    """Append each episode emission on ``sim`` to ``saved["episodes"]`` as ``(phase, env_ids)``."""
    from holosoma.hooks import Phase

    episodes = saved.setdefault("episodes", [])
    for phase in (Phase.EPISODE_END, Phase.EPISODE_START):
        sim.hooks.add(phase, lambda env_ids, phase=phase: episodes.append((phase.value, env_ids.tolist())))


def peer(endpoint: str, output: Path, num_envs: int, loopback: bool, backend: str) -> None:
    import torch

    from holosoma.cosim.bridge import PeerBridge
    from holosoma.cosim.render.renderer import RenderHandler, create_render_task
    from holosoma.hooks import Phase

    saved: dict[str, Any] = {}

    with create_render_task(config(num_envs, loopback, backend=backend)) as task:
        sim = task.simulator
        sim.hooks.add(Phase.FRAME_END, lambda: capture(sim, saved), after=sim.sensor_render_hook)
        record_episodes(sim, saved)

        class LinkCheckingHandler(RenderHandler):
            def step(self, snapshot: Any, state: Any) -> None:
                # The renderer's state is the head's memory, not a copy of it.
                saved["linked"] = all(state[name] is sim.state.get(name) for name in state)
                # Its cameras render straight into the exports.
                saved["exports_in_place"] = all(
                    sim.state.get(name).data_ptr() == allocation.ptr for name, allocation in self.allocations.items()
                )
                super().step(snapshot, state)

        bridge = PeerBridge(LinkCheckingHandler(task), endpoint=endpoint)
        (output / "ready").write_text("ready\n")
        bridge.run()
        # Inside the session: closing Isaac Sim's app ends the process.
        torch.save({**saved, "frames": task.frames_rendered}, output / "peer.pt")


def head(endpoint: str, output: Path, frames: int, num_envs: int, loopback: bool) -> None:
    import torch

    from holosoma.hooks import Phase

    saved: dict[str, Any] = {}
    with config(num_envs, loopback, endpoint=endpoint).create_task() as task:
        sim = task.simulator
        cosim = sim.installed_plugins["cosim"]
        sim.hooks.add(Phase.FRAME_END, lambda: capture(sim, saved), after=sim.sensor_render_hook)
        record_episodes(sim, saved)
        task.reset_envs_idx()
        for frame in range(frames):
            if frame == frames // 4:
                task.reset_envs_idx(torch.tensor([num_envs - 1], device=task.device))
            if loopback and frame == frames // 2:
                # Move both cameras the same way; the remote one moves through the renderer.
                manager = sim.sensor_manager
                moved = manager.get_mount_pose("overview")
                moved[:, 0] += 0.2
                for name in ("overview", "loopback"):
                    manager.set_mount_pose(name, moved)
            task.step()
        saved["episodes"] = list(saved["episodes"])  # without the episode ends that close emits
        # Remote camera outputs are the renderer's exports, not copies.
        saved["imports_in_place"] = all(
            sim.state.get(name).data_ptr() == tensor.data_ptr() for name, tensor in cosim.bridge.imports.items()
        )
    saved["exchanges"] = cosim.bridge.sequence
    torch.save(saved, output / "head.pt")


def _role(role: str, output: Path, args: list[str], renderer: str) -> subprocess.Popen[bytes]:
    command = [str(Path(__file__)), role, "--output", str(output), *args]
    log = (output / f"{role}.log").open("wb")
    if role == "peer" and renderer != "mjwarp":
        # Another simulator's environment, running this tree.
        setup = _REPO / "scripts" / f"source_{renderer}_setup.sh"
        script = f"source {shlex.quote(str(setup))} && PYTHONPATH={shlex.quote(str(_REPO / 'src/holosoma'))} "
        script += f"exec python {shlex.join([*command, '--simulator', renderer])}"
        return subprocess.Popen(["/bin/bash", "-c", script], stdout=log, stderr=subprocess.STDOUT)
    return subprocess.Popen([sys.executable, *command], stdout=log, stderr=subprocess.STDOUT)


def run_pair(num_envs: int, *extra: str, renderer: str = "mjwarp") -> tuple[dict, dict]:
    """Run an MJWarp head against a ``renderer`` peer to completion; return what each saved.

    A ``renderer`` other than ``mjwarp`` runs under ``scripts/source_<renderer>_setup.sh``.
    """
    import torch

    # Unix socket paths are limited to 107 characters; keep the endpoint short.
    with tempfile.TemporaryDirectory(prefix="cosim-") as directory:
        output = Path(directory)
        args = ["--endpoint", f"ipc://{output}/control.sock", "--num-envs", str(num_envs), *extra]
        peer_process = _role("peer", output, args, renderer)
        try:
            deadline = time.monotonic() + _TIMEOUT_S
            while not (output / "ready").exists():
                assert peer_process.poll() is None, (output / "peer.log").read_text()
                assert time.monotonic() < deadline, "peer did not start"
                time.sleep(0.2)
            head_process = _role("head", output, args, renderer)
            assert head_process.wait(_TIMEOUT_S) == 0, (output / "head.log").read_text()
            assert peer_process.wait(_TIMEOUT_S) == 0, (output / "peer.log").read_text()
        finally:
            peer_process.kill()
        return torch.load(output / "head.pt"), torch.load(output / "peer.pt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=["head", "peer"])
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--loopback", action="store_true")
    parser.add_argument("--simulator", choices=["mjwarp", "isaacsim"], default="mjwarp")
    args = parser.parse_args()
    if args.role == "peer":
        peer(args.endpoint, args.output, args.num_envs, args.loopback, args.simulator)
    else:
        head(args.endpoint, args.output, args.frames, args.num_envs, args.loopback)


if __name__ == "__main__":
    main()
