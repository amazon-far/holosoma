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
- ``meta["sensors"]`` (:func:`sensor_settings`): per remote camera, ``vertical_fov`` in degrees, ``near`` and
  ``far`` in meters, ``depth_clipping_behavior``, and ``mount`` (a ``SensorMountConfig``): ``target_kind``
  ``robot_link``, ``actor`` or ``world``; ``target``, a ``body_names`` or ``object_names`` entry; and
  ``position`` and ``orientation`` ``[w, x, y, z]`` in the target's frame, or the environment's for ``world``. Identity looks along ``-Z`` with ``+Y`` up. The renderer replies with the
  settings it applies, which must match.
- Requests (:func:`output_fields`): the remote cameras' SimState output fields, ``camera/<name>/rgb``
  ``uint8 [E, H, W, 3]`` and ``camera/<name>/depth`` ``float32 [E, H, W, 1]``, metric distance to the image
  plane with the camera's ``depth_clipping_behavior`` applied.

The remote cameras' outputs on the head are linked onto the exports, read in place until the next exchange;
exports are final, since both sides' settings (clipping included) match. Head consumers that read them on
another CUDA stream synchronise before the next exchange.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import torch
from loguru import logger

from holosoma.cosim.base import HeadPlugin
from holosoma.cosim.bridge import as_received
from holosoma.cosim.cuda import FieldSpec
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


def sensor_settings(record: CameraRecord) -> dict[str, Any]:
    """The settings of camera ``record`` a renderer must match beyond its output fields: optics, clipping, mount."""
    config = record.config
    settings: dict[str, Any] = as_received(
        {
            "vertical_fov": config.vertical_fov,
            "near": config.near,
            "far": config.far,
            "depth_clipping_behavior": config.depth_clipping_behavior,
            "mount": asdict(config.mount),
        }
    )
    return settings


def output_fields(record: CameraRecord) -> list[FieldSpec]:
    """The SimState output fields of ``record``: rgb ``uint8 [E, H, W, 3]``, depth ``float32 [E, H, W, 1]``."""
    return [
        FieldSpec(
            record.field_name(data_type),
            tuple(record.buffers[data_type].shape),
            str(record.buffers[data_type].dtype).removeprefix("torch."),
        )
        for data_type in record.config.data_types
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
        if len({record.effective_decimation for record in self.remote}) > 1:
            rates = {record.name: record.effective_decimation for record in self.remote}
            raise ValueError(f"Remote cameras of one co-simulation plugin must share a decimation, got {rates}")
        super().__init__(cfg, simulator)
        read = set(self.bridge.reads)
        self._unread_mounts = [
            (record, record.mount_poses.clone())
            for record in self.remote
            if record.mount_poses is not None and f"mount_pose/{record.name}" not in read
        ]

    def requests(self) -> list[FieldSpec]:
        return [spec for record in self.remote for spec in output_fields(record)]

    def meta(self) -> dict[str, Any]:
        return {**scene_layout(self.simulator), "sensors": self._sensors()}

    def accept(self, meta: Mapping[str, Any]) -> None:
        if meta.get("sensors") != (sent := self._sensors()):
            raise ValueError(f"The renderer applies sensor settings {meta.get('sensors')}, the head sent {sent}")

    def _sensors(self) -> dict[str, Any]:
        return {record.name: sensor_settings(record) for record in self.remote}

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
