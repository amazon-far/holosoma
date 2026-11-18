"""Co-simulation rendering, head side: the ``cosim-render`` plugin and the contract a renderer serves.

A camera with ``remote`` set to the plugin's slot takes its images from the renderer. Any process can serve the
plugin by subclassing :class:`holosoma.cosim.base.PeerHandler`; :mod:`.renderer` is the Holosoma one.

- Handshake ``meta`` (:func:`scene_layout`): ``dof_names``, ``body_names`` and ``object_names`` give the column
  order of the joint, robot-body and object fields; ``env_origins`` ``[E, 3]`` is where the head simulated
  each environment, and the renderer places its environments there.
- State: every non-empty SimState field. Roots are world-frame ``[pos, quat xyzw, lin vel, ang vel]``;
  ``_rigid_body_pos`` and ``_rigid_body_rot`` are each body's world position and xyzw orientation;
  ``mount_pose/<sensor>`` ``[E, 7]`` is a sensor's current offset from its mount, XYZ + XYZW. A renderer
  must apply it for remote cameras to follow ``set_mount_pose()``; the head warns once if one moves unread.
- Requests: one per remote-camera output, named :func:`image_name` and decoded by :func:`camera_output`. rgb is
  ``uint8 [E, H, W, 3]``; depth is ``float32 [E, H, W, 1]``, metric distance to the image plane, ``+inf`` for no
  return. The head applies the camera's ``depth_clipping_behavior`` when it stores the image.
- Request ``meta``: ``vertical_fov`` in degrees, ``near`` and ``far`` in meters, ``depth_clipping_behavior``,
  and ``mount`` (a ``SensorMountConfig``): ``target_kind`` ``robot_link``, ``actor`` or ``world``; ``target``,
  a ``body_names`` or ``object_names`` entry; and ``position`` and ``orientation`` ``[w, x, y, z]`` in the
  target's frame, or the environment's for ``world``. Identity looks along ``-Z`` with ``+Y`` up.

Exports are filled after each step and read in place by the head, which copies due cameras' images
before its next exchange. :class:`~holosoma.cosim.base.PeerHandler` reports which it wrote.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import torch
from loguru import logger

from holosoma.cosim.base import HeadPlugin
from holosoma.cosim.bridge import BufferRequest, as_received
from holosoma.simulator.base_simulator.hooks import Phase

if TYPE_CHECKING:
    from holosoma.config_types.plugin import CosimRenderPluginConfig
    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator
    from holosoma.simulator.base_simulator.hooks import HookHandle, HookRegistry
    from holosoma.simulator.shared.sensor_manager import CameraRecord


def scene_layout(sim: BaseSimulator) -> dict[str, Any]:
    """The handshake ``meta``: joint, body and object order of the state fields, and environment origins."""
    layout: dict[str, Any] = as_received(
        {
            "dof_names": list(sim.dof_names),
            "body_names": list(sim.body_names),
            "object_names": list(sim.object_state_names),
            "env_origins": sim.scene.env_origins.cpu().tolist(),
        }
    )
    return layout


def image_name(camera: str, data_type: str) -> str:
    """The request name of ``camera``'s ``data_type`` output."""
    return f"{camera}/{data_type}"


def camera_output(name: str) -> tuple[str, str]:
    """The camera and data type a request ``name`` from :func:`image_name` refers to."""
    camera, _, data_type = name.rpartition("/")
    return camera, data_type


def camera_requests(record: CameraRecord, num_envs: int) -> list[BufferRequest]:
    """One request per output of ``record``: rgb ``uint8 [E, H, W, 3]``, depth ``float32 [E, H, W, 1]``."""
    config = record.config
    meta = as_received(
        {
            "mount": asdict(config.mount),
            "vertical_fov": config.vertical_fov,
            "near": config.near,
            "far": config.far,
            "depth_clipping_behavior": config.depth_clipping_behavior,
        }
    )
    return [
        BufferRequest(
            image_name(record.name, data_type),
            (num_envs, config.height, config.width, 3 if data_type == "rgb" else 1),
            "uint8" if data_type == "rgb" else "float32",
            meta,
        )
        for data_type in config.data_types
    ]


class CosimRenderPlugin(HeadPlugin):
    """Share ``simulator``'s state with a renderer and publish a snapshot whenever a camera is due.

    Remote cameras take the renderer's images: each frame they are filled after local sensors render
    and before consumers read them. Without remote cameras, state is published after local cameras render.
    """

    def __init__(self, cfg: CosimRenderPluginConfig, simulator: BaseSimulator) -> None:
        manager = simulator.sensor_manager
        if not manager.cameras:
            raise ValueError("The head publishes when its cameras are due; configure at least one camera")
        slot = simulator.hooks.installing_slot
        self.remote = [record for record in manager.cameras if slot is not None and record.config.remote == slot]
        super().__init__(cfg, simulator)
        read = set(self.bridge.reads)
        self._unread_mounts = [
            (record, record.mount_poses.clone())
            for record in self.remote
            if record.mount_poses is not None and f"mount_pose/{record.name}" not in read
        ]

    def requests(self) -> list[BufferRequest]:
        num_envs = self.simulator.sensor_manager.num_envs
        return [request for record in self.remote for request in camera_requests(record, num_envs)]

    def meta(self) -> dict[str, Any]:
        return scene_layout(self.simulator)

    def install(self, hooks: HookRegistry) -> Iterable[HookHandle]:
        yield hooks.add(
            Phase.FRAME_END,
            self._frame,
            name="cosim.frame",
            after=self.simulator.sensor_capture_hook,
            before=self.simulator.sensor_render_hook,
        )

    def _frame(self) -> None:
        """Publish when a camera is due, and fill the due remote cameras from the renderer's images."""
        due = self.simulator.sensor_manager.last_due
        if self.remote:
            cameras = [record for record in self.remote if record.name in due]
            if not cameras:
                return
            self._warn_if_unread_mount_moved()
            self.bridge.publish()
            images = self.bridge.imports
            for record in cameras:
                for data_type in record.config.data_types:
                    record.set_buffer(data_type, images[image_name(record.name, data_type)].clone())
        elif due:
            self.bridge.publish()

    def _warn_if_unread_mount_moved(self) -> None:
        """Warn once if a remote camera moves although its renderer does not read its mount pose."""
        if moved := [record.name for record, pose in self._unread_mounts if not torch.equal(pose, record.mount_poses)]:
            logger.warning(f"Remote cameras {moved} moved, but their renderer does not read mount_pose/<camera>")
            self._unread_mounts = []
