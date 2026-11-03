"""Co-simulation follower: render camera images for a leader's robot and object poses."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from holosoma.hooks import Phase
from holosoma.simulator.plugins.cosim.protocol import Array, Message, Replier, env_facts
from holosoma.utils.safe_torch_import import torch

if TYPE_CHECKING:
    from holosoma.config_types.plugin import CoSimFollowerPluginConfig
    from holosoma.envs.base_task.base_task import BaseTask


class CoSimFollowerPlugin:
    """Turn each leader request into one frame: write its state, render, reply with the images.

    FRAME_BEGIN blocks for the next request, PRE_STEP writes the poses (kinematic playback then
    propagates them), and FRAME_END replies after the simulator has rendered its sensors.
    """

    def __init__(self, cfg: CoSimFollowerPluginConfig, task: BaseTask) -> None:
        self.cfg = cfg
        self.task = task
        self.simulator = sim = task.simulator
        sim_cfg = sim.simulator_config.sim
        needs = [
            flag
            for flag, missing in (
                ("--simulator.config.sim.kinematic_playback True", not sim.kinematic_playback),
                ("--simulator.config.sim.control_decimation 1", sim_cfg.control_decimation_steps != 1),
                ("--simulator.config.sim.render_interval 1", sim_cfg.render_interval_steps != 1),
                ("--time-scale 0", task.config.time_scale > 0),
                ("no --simulator.config.sim.remote_sensors", sim.remote_sensors),
                ("a camera sensor.* config", sim.sensor_manager is None or not sim.sensor_manager.cameras),
            )
            if missing
        ]
        if needs:
            raise ValueError(f"cosim-follower needs {', '.join(needs)}.")
        assert sim.sensor_manager is not None
        self.sensors = sim.sensor_manager
        for record in self.sensors.cameras:
            record.effective_decimation = 1  # the leader decides when a camera is due
        self._env_ids = torch.arange(sim.num_envs, device=sim.sim_device)
        self._objects: list[str] = []
        self._pending: Message | None = None

        self.socket = Replier(cfg.endpoint)
        logger.info(f"cosim follower bound {cfg.endpoint}")
        task.hooks.add(Phase.FRAME_BEGIN, self._on_frame_begin, name="cosim.follower.recv")
        task.hooks.add(Phase.PRE_STEP, self._write_state, name="cosim.follower.write")
        task.hooks.add(Phase.FRAME_END, self._reply, name="cosim.follower.reply")
        task.hooks.add(Phase.CLOSE, self.socket.close, name="cosim.follower.close")

    def _on_frame_begin(self) -> None:
        while True:
            msg = self.socket.recv(self.cfg.idle_timeout_s)
            if msg.type == "hello":
                self._objects = msg.header["objects"]
                self.socket.send("hello", env_facts(self.simulator))
            elif msg.type == "bye":
                self.socket.send("ok")
                self.task.request_shutdown("cosim leader closed")
                return
            elif msg.type == "step":
                self._pending = msg
                return
            else:
                self.socket.send("error", {"msg": f"unexpected message type {msg.type!r}"})

    def _write_state(self) -> None:
        if self._pending is None:
            return
        try:
            sim = self.simulator
            arrays = self._pending.arrays
            origins = sim.scene.env_origins
            root = torch.zeros(sim.num_envs, 13, device=sim.sim_device)
            root[:, :7] = _tensor(arrays["root"], sim.sim_device)
            root[:, :3] += origins
            dof = torch.zeros(sim.num_envs, sim.num_dof, 2, device=sim.sim_device)
            dof[..., 0] = _tensor(arrays["dof_pos"], sim.sim_device)
            sim.set_robot_state(self._env_ids, root_states=root, dof_states=dof)
            if self._objects:
                poses = _tensor(arrays["objects"], sim.sim_device)
                poses[..., :3] += origins
                states = torch.zeros(poses.shape[0] * sim.num_envs, 13, device=sim.sim_device)
                states[:, :7] = poses.reshape(-1, 7)
                sim.set_actor_states(self._objects, self._env_ids, states)
        except Exception as exc:
            self._fail(exc)
            raise

    def _reply(self) -> None:
        msg, self._pending = self._pending, None
        if msg is None:
            return
        try:
            rows = msg.header["rows"]
            images: dict[str, Array] = {}
            for cam in msg.header["cams"]:
                for data_type in self.sensors.get(cam).config.data_types:
                    image = self.simulator.get_camera_data(cam, data_type, device="cpu")
                    images[f"{cam}/{data_type}"] = (image if rows is None else image[rows]).numpy()
            self.socket.send("frames", arrays=images)
        except Exception as exc:
            self._fail(exc)
            raise

    def _fail(self, exc: Exception) -> None:
        """Answer the pending request with the failure, so the leader raises instead of timing out."""
        self._pending = None
        self.socket.send("error", {"msg": f"{type(exc).__name__}: {exc}"})


def _tensor(array: Array, device: str) -> torch.Tensor:
    return torch.from_numpy(array).to(device=device, dtype=torch.float32)
