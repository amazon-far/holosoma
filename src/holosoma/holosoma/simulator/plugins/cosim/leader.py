"""Co-simulation leader: this task's physics, another process's camera images."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

from holosoma.hooks import Phase
from holosoma.simulator.plugins.cosim.protocol import CoSimError, Requester, env_facts
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.simulator_config import SimulatorType

if TYPE_CHECKING:
    from holosoma.config_types.plugin import CoSimLeaderPluginConfig
    from holosoma.envs.base_task.base_task import BaseTask
    from holosoma.simulator.shared.sensor_manager import SensorManager


class CoSimLeaderPlugin:
    """Fill the task's camera buffers with images rendered by a ``cosim-follower``.

    At FRAME_END, when any camera is due, sends the current robot and object poses and stores the
    returned images. Reset envs are re-rendered through ``SensorManager.rerender`` before
    observations read them.
    """

    def __init__(self, cfg: CoSimLeaderPluginConfig, task: BaseTask) -> None:
        self.cfg = cfg
        self.simulator = sim = task.simulator
        if not sim.remote_sensors:
            raise ValueError("cosim-leader needs --simulator.config.sim.remote_sensors True.")
        if sim.sensor_manager is None or not sim.sensor_manager.cameras:
            raise ValueError("cosim-leader needs at least one camera sensor.* config.")
        if task.installed_plugins:
            raise ValueError(
                f"cosim-leader must be the first plugin, so later plugins read the images it returns; "
                f"it follows {sorted(task.installed_plugins)}."
            )
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            raise ValueError("cosim-leader supports a single process; multi-GPU runs are not supported.")
        self.sensors: SensorManager = sim.sensor_manager
        self._env_ids = torch.arange(sim.num_envs, device=sim.sim_device)

        self.socket = Requester(cfg.endpoint)
        self._awaiting_reply = False
        facts = env_facts(sim)
        self.objects: list[str] = facts["objects"]
        logger.info(f"cosim leader: connecting to {cfg.endpoint}")
        self.socket.send("hello", facts)
        follower = self.socket.recv(cfg.connect_timeout_s).header
        mismatch = [key for key in ("num_envs", "dof_names", "cameras") if follower.get(key) != facts[key]]
        if set(follower.get("objects", [])) != set(self.objects):
            mismatch.append("objects")
        if mismatch:
            self._send_bye()
            details = "; ".join(f"{key}: leader {facts[key]} vs follower {follower.get(key)}" for key in mismatch)
            raise CoSimError(f"cosim follower does not match this task ({details}).")
        self.sensors.simulator_type = SimulatorType(follower["simulator"])
        logger.info(f"cosim leader: connected to a {follower['simulator']} follower")

        self.sensors.cameras[0].rerender = self._rerender  # one request refreshes every camera
        task.hooks.add(Phase.FRAME_END, self._on_frame_end, name="cosim.leader.render")
        task.hooks.add(Phase.CLOSE, self._on_close, name="cosim.leader.close")

    def _state(self) -> dict[str, np.ndarray]:
        """Robot and object poses, positions relative to each env's origin."""
        sim = self.simulator
        origins = sim.scene.env_origins
        root = sim.robot_root_states[:, :7].clone()
        root[:, :3] -= origins
        if self.objects:
            poses = sim.get_actor_states(self.objects, self._env_ids)[:, :7].reshape(len(self.objects), -1, 7)
            objects = torch.cat([poses[..., :3] - origins, poses[..., 3:]], dim=-1)
        else:
            objects = root.new_zeros((0, sim.num_envs, 7))
        return {name: _host(t) for name, t in (("root", root), ("dof_pos", sim.dof_pos[:]), ("objects", objects))}

    def _request(self, cams: list[str], rows: list[int] | None) -> dict[str, np.ndarray]:
        self.socket.send("step", {"cams": cams, "rows": rows}, self._state())
        self._awaiting_reply = True
        reply = self.socket.recv(self.cfg.step_timeout_s)
        self._awaiting_reply = False
        return reply.arrays

    def _on_frame_end(self) -> None:
        due = self.sensors.collect_due()
        if not due:
            return
        images = self._request([record.name for record in due], None)
        for record in due:
            for data_type, buffer in record.buffers.items():
                record.set_buffer(data_type, _device(images[f"{record.name}/{data_type}"], buffer))

    def _rerender(self, env_ids: torch.Tensor) -> None:
        rows = env_ids.to(self._env_ids.device, torch.long)
        images = self._request(self.sensors.camera_names, rows.tolist())
        for record in self.sensors.cameras:
            for data_type, buffer in record.buffers.items():
                record.set_buffer_rows(data_type, rows, _device(images[f"{record.name}/{data_type}"], buffer))

    def _send_bye(self) -> None:
        try:
            if self._awaiting_reply:  # interrupted mid-request: REQ must take the reply before sending
                self.socket.recv(timeout_s=5.0)
            self.socket.send("bye")
            self.socket.recv(timeout_s=5.0)
        except Exception as exc:  # teardown: the socket may be mid-request after a failed step
            logger.warning(f"cosim leader: follower did not acknowledge close: {exc}")
        self.socket.close()

    def _on_close(self) -> None:
        self._send_bye()


def _host(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().to("cpu", torch.float32).numpy()


def _device(array: Any, like: torch.Tensor) -> torch.Tensor:
    return torch.from_numpy(array).to(device=like.device, dtype=like.dtype)
