"""Co-simulation rendering, head side: the ``cosim-render`` plugin and the contract a renderer serves.

A camera with ``remote`` set to the plugin's slot takes its images from the renderer. Any process can serve the
plugin by subclassing :class:`holosoma.cosim.base.PeerHandler`; :mod:`.renderer` is the Holosoma one.

- ``handshake`` (:func:`scene_layout`): ``dof_names``, ``body_names`` and ``object_names`` give the column
  order of the joint, robot-body and object fields; ``env_origins`` ``[E, 3]`` is where the head simulated
  each environment, and the renderer places its environments there; ``num_rerenders_on_reset`` is the
  head's ``sim.num_rerenders_on_reset``.
- State: every non-empty SimState field. Roots are world-frame ``[pos, quat xyzw, lin vel, ang vel]``;
  ``rigid_body_pos_w`` and ``rigid_body_quat_w`` are each body's world position and xyzw orientation;
  ``mount_pose/<sensor>`` ``[E, 7]`` is a sensor's current offset from its mount, XYZ + XYZW. A renderer
  must apply it for remote cameras to follow ``set_mount_pose()``; the head warns once if one moves unread.
- Writes: the remote cameras' SimState fields, ``camera/<name>/rgb``
  ``uint8 [E, H, W, 3]`` and ``camera/<name>/depth`` ``float32 [E, H, W, 1]``, metric distance to the image
  plane with the camera's ``depth_clipping_behavior`` applied. Each camera's settings (field of view, near/far,
  clipping, mount target) are its config's on both sides; the renderer writes the frames the head's cameras
  would, from the state it receives.
- Snapshots: one per frame a camera is due, and one per head ``episode_start`` with its ``phase`` and
  ``env_ids``, after the reset and the head's own re-render: the renderer ends those environments' open
  episodes, then starts them, as a reset does. With ``num_rerenders_on_reset`` above 0 it re-renders those
  environments' rows of the exports, as the head re-renders its local cameras; otherwise episode snapshots
  must not write the exports. Episodes still open end when the head closes.

The remote cameras' outputs on the head are linked onto the exports, read in place until the next exchange;
exports are final, since both sides' settings (clipping included) match. Head consumers that read them on
another CUDA stream synchronise before the next exchange.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import torch
from loguru import logger

from holosoma.cosim.base import HeadPlugin
from holosoma.cosim.bridge import as_received
from holosoma.hooks import Phase

if TYPE_CHECKING:
    from holosoma.config_types.plugin import CosimRenderPluginConfig
    from holosoma.envs.base_task.base_task import BaseTask
    from holosoma.hooks import HookHandle, HookRegistry
    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator


def scene_layout(sim: BaseSimulator) -> dict[str, Any]:
    """The ``handshake``: joint, body and object order of the state fields, environment origins and re-render count."""
    layout: dict[str, Any] = as_received(
        {
            "dof_names": list(sim.dof_names),
            "body_names": list(sim.body_names),
            "object_names": list(sim.object_state_names),
            "env_origins": sim.scene.env_origins.cpu().tolist(),
            "num_rerenders_on_reset": sim.simulator_config.sim.num_rerenders_on_reset,
        }
    )
    return layout


class CosimRenderPlugin(HeadPlugin):
    """Share the task's simulator state with a renderer and publish a frame whenever a camera is due.

    Remote cameras take the renderer's images: each frame they are filled after local sensors render
    and before consumers read them. Without remote cameras, state is published after local cameras render.
    Episode starts are published too, after the simulator's re-render, so the renderer ends and starts the
    same episodes and, with ``sim.num_rerenders_on_reset`` above 0, re-renders them as local cameras are.
    """

    def __init__(self, cfg: CosimRenderPluginConfig, task: BaseTask) -> None:
        simulator = task.simulator
        manager = simulator.sensor_manager
        if not manager.cameras:
            raise ValueError("The head publishes when its cameras are due; configure at least one camera")
        slot = simulator.hooks.installing_slot
        self.remote = [record for record in manager.cameras if slot is not None and record.config.remote == slot]
        if len({record.effective_decimation for record in self.remote}) > 1:
            rates = {record.name: record.effective_decimation for record in self.remote}
            raise ValueError(f"Remote cameras of one co-simulation plugin must share a decimation, got {rates}")
        super().__init__(cfg, task)
        read = set(self.bridge.reads)
        self._unread_mounts = [
            (record, record.mount_poses.clone())
            for record in self.remote
            if record.mount_poses is not None and f"mount_pose/{record.name}" not in read
        ]

    def writes(self) -> list[str]:
        return [record.field_name(data_type) for record in self.remote for data_type in record.config.data_types]

    def handshake(self) -> dict[str, Any]:
        return scene_layout(self.simulator)

    def install(self, hooks: HookRegistry) -> Iterable[HookHandle]:
        yield hooks.add(
            Phase.FRAME_END,
            self._frame,
            name=f"{self.name}.frame",
            after=self.simulator.sensor_capture_hook,
            before=self.simulator.sensor_render_hook,
        )
        yield hooks.add(
            Phase.EPISODE_START,
            self._episode_start,
            name=f"{self.name}.episode_start",
            after=self.simulator.sensor_rerender_hook,
        )

    def _episode_start(self, env_ids: torch.Tensor) -> None:
        """Publish the start of ``env_ids``' episodes; the renderer re-renders their rows when configured to."""
        self.bridge.publish(Phase.EPISODE_START, env_ids)
        if self.simulator.simulator_config.sim.num_rerenders_on_reset > 0:
            for record in self.remote:
                for data_type in record.config.data_types:
                    record.mark_written(data_type)

    def _frame(self) -> None:
        """Publish when a camera is due, and fill the due remote cameras from the renderer's images."""
        due = self.simulator.sensor_manager.last_due
        if self.remote:
            cameras = [record for record in self.remote if record.name in due]
            if not cameras:
                return
            self._warn_if_unread_mount_moved()
            self.bridge.publish()
            for record in cameras:  # their outputs are linked onto the renderer's exports
                for data_type in record.config.data_types:
                    record.mark_written(data_type)
        elif due:
            self.bridge.publish()

    def _warn_if_unread_mount_moved(self) -> None:
        """Warn once if a remote camera moves although its renderer does not read its mount pose."""
        if moved := [record.name for record, pose in self._unread_mounts if not torch.equal(pose, record.mount_poses)]:
            logger.warning(f"Remote cameras {moved} moved, but their renderer does not read mount_pose/<camera>")
            self._unread_mounts = []
