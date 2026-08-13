"""Backend-agnostic, ROS-free substrate for camera-frame egress plugins.

Defines :class:`CameraConsumerPlugin`, the base for a plugin that consumes rendered camera frames
(ROS2 publish, live window, mp4 recording), plus :class:`FramePacket` and :class:`CameraIntrinsics`.

A consumer is a plugin (``cls(cfg, simulator)``, no base class required): it registers its per-step
:meth:`~CameraConsumerPlugin.publish` on ``FRAME_END`` and its :meth:`~CameraConsumerPlugin.stop`
on ``CLOSE`` in ``__init__``. The cameras' ``render_sensors`` is registered on the SAME phase first
(in ``BaseSimulator.__init__``, before the plugins), so by registration order the buffers are
fresh when a consumer's callback runs. Each consumer declares the streams it needs via
:meth:`~CameraConsumerPlugin.wanted_streams`; the base validates them against the configured cameras
at construction (fail-loud) and, each step, snapshots exactly those to host once — sharing the single
``get_camera_data(device="cpu")`` copy across every consumer of the same (camera, modality) — and
hands the consumer a per-step batch of just its streams.

Stream identity is a :data:`StreamKey` = ``(camera, modality, env_id)``. ROS2 wants env 0 only; a
tiling recorder may want several envs — env selection is a per-consumer lever, not a global bake-in.
Concrete consumers live in their own modules and import heavy deps (rclpy, cv2) at module top; those
modules are reached only via a config's ``get_cls()``, so this base and the config layer stay
importable without those deps.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

import numpy as np
from loguru import logger

from holosoma.simulator.base_simulator.hooks import Phase

if TYPE_CHECKING:
    from holosoma.config_types.plugin import PluginConfig
    from holosoma.config_types.sensor import CameraSensorConfig
    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator

# (camera_name, modality, env_id). The unit of stream identity across the egress system.
# ``typing.Tuple`` (not the builtin ``tuple[...]``) so this runtime alias is valid on the
# mypy target (python_version = 3.8), where builtin generics aren't subscriptable at runtime.
StreamKey = Tuple[str, str, int]


@dataclass(frozen=True)
class CameraIntrinsics:
    """Static intrinsics of one camera, carried on every :class:`FramePacket`.

    Backend-agnostic core fields (the same set on every backend), sufficient to derive a
    pinhole projection matrix ``K`` for a ``CameraInfo`` message.
    """

    width: int
    height: int
    vertical_fov: float
    """Vertical field of view, degrees."""
    near: float
    far: float


@dataclass
class FramePacket:
    """One rendered frame, already copied to host, ready for a consumer to use.

    Snapshotted by :class:`CameraConsumerPlugin` (via the shared cached ``get_camera_data`` host copy)
    and handed to :meth:`CameraConsumerPlugin.publish` for each ``(camera, modality, env_id)`` the
    consumer wanted this step. Holds no GPU tensor, so it is safe to pass to a worker thread.
    """

    camera: str
    modality: str
    """``"rgb"`` or ``"depth"``."""
    env_id: int
    array: np.ndarray
    """Host-side copy: ``uint8 [H, W, 3]`` R,G,B for rgb; ``float32 [H, W, 1]`` meters for depth."""
    sim_time: float
    """Simulation time of this frame (``simulator.time()``), for the message timestamp."""
    intrinsics: CameraIntrinsics

    @property
    def key(self) -> StreamKey:
        return (self.camera, self.modality, self.env_id)


class CameraConsumerPlugin:
    """Base for a plugin that consumes rendered camera frames each control step.

    Subclass and, in ``__init__``, set any transport fields then call
    ``super().__init__(config, simulator)`` (which validates the streams and registers the callbacks;
    ``wanted_streams`` runs there, so fields it reads must be set first). Open the live transport lazily
    in :meth:`start` — the base defers ``start`` to the first :meth:`publish`. Implement
    :meth:`wanted_streams`, :meth:`start`, :meth:`publish`, and :meth:`stop`.

    The base handles: fail-loud validation of the wanted streams against the configured cameras;
    the per-step snapshot (one shared ``get_camera_data(device="cpu")`` host copy per (camera,
    modality), sliced per wanted env) restricted to cameras that actually rendered this step; and
    failure isolation (an exception in ``publish``/``stop`` is logged, never propagated to the sim).
    """

    def __init__(self, config: PluginConfig, simulator: BaseSimulator) -> None:
        # A plugin is duck-typed (cls(cfg, simulator) registering on simulator.hooks); no base to call.
        # ``config`` is typed per subclass; ``cfg`` mirrors the plugin convention (ClockPublishPlugin etc).
        self.cfg = config
        self.config = config
        self.simulator = simulator
        self._wanted = self.wanted_streams()
        self._validate_streams()
        self._started = False
        simulator.hooks.add(Phase.FRAME_END, self._on_frame_end, name=f"{self._label()}.publish")
        simulator.hooks.add(Phase.CLOSE, self._on_close, name=f"{self._label()}.stop")

    def _label(self) -> str:
        return type(self).__name__

    @property
    def sensors_config(self) -> dict[str, CameraSensorConfig]:
        """The active mounted cameras, keyed by sensor name."""
        return self.simulator.sensor_config

    @property
    def control_hz(self) -> float:
        """Control-step rate (sim fps / control_decimation), the base for resolving frequency strings."""
        sim = self.simulator.simulator_config.sim
        return sim.fps / sim.control_decimation_steps

    # ----- subclass contract -----

    @abstractmethod
    def wanted_streams(self) -> set[StreamKey]:
        """The ``(camera, modality, env_id)`` triples this consumer needs.

        The base validates these against the configured cameras and snapshots only these each step.
        ROS2 typically returns ``{(cam, mod, 0)}``; a tiling recorder returns one triple per
        (camera, modality, env) it watches. Called once in ``__init__``.
        """

    @abstractmethod
    def start(self) -> None:
        """Open the transport (node/sockets/threads/window) and publish any static/latched data.

        Called by the base exactly once, lazily, just before the first :meth:`publish`."""

    @abstractmethod
    def publish(self, frames: dict[StreamKey, FramePacket]) -> None:
        """Handle one control step's fresh frames for THIS consumer (only its wanted streams present).

        ``frames`` holds an entry per wanted stream that rendered this step (slower/decimated cameras
        are absent on steps they did not render). Called only when ``frames`` is non-empty.
        """

    @abstractmethod
    def stop(self) -> None:
        """Drain/stop workers, finalize output (encode video), and close the transport/window."""

    # ----- base machinery -----

    def _validate_streams(self) -> None:
        """Fail loud if a wanted stream names a camera/modality/env the sim does not provide."""
        cams = dict(self.simulator.sensor_config)
        # training_config.num_envs is set at __init__ on every backend; self.num_envs is not yet
        # populated when IsaacSim builds hooks during scene setup (it lands in create_envs).
        num_envs = self.simulator.training_config.num_envs
        for cam, mod, env in sorted(self._wanted):
            if cam not in cams:
                raise ValueError(
                    f"{self._label()} references camera '{cam}', which is not among the configured cameras "
                    f"(cameras: {sorted(cams)}). Consumer streams must name a configured camera."
                )
            if mod not in cams[cam].data_types:
                raise ValueError(
                    f"{self._label()} wants '{mod}' from camera '{cam}', but that camera renders only "
                    f"{list(cams[cam].data_types)}. Add '{mod}' to its data_types or fix the consumer."
                )
            if not 0 <= env < num_envs:
                raise ValueError(
                    f"{self._label()} wants env {env} of camera '{cam}', but the sim has {num_envs} env(s) "
                    f"[0, {num_envs})."
                )

    def _intrinsics_of(self, cam: str) -> CameraIntrinsics:
        c = self.simulator.sensor_config_by_name(cam)
        return CameraIntrinsics(width=c.width, height=c.height, vertical_fov=c.vertical_fov, near=c.near, far=c.far)

    def _on_frame_end(self) -> None:
        """FRAME_END callback: snapshot this consumer's fresh wanted streams, then publish.

        Wrapped so a failure in one consumer neither propagates to the sim nor stops sibling hooks."""
        try:
            if not self._started:
                self.start()
                self._started = True
            batch = self._snapshot()
            if batch:
                self.publish(batch)
        except Exception as exc:  # isolation: a consumer must never break the sim loop
            logger.error(f"Camera consumer {self._label()} publish failed: {exc}")

    def _snapshot(self) -> dict[StreamKey, FramePacket]:
        """Snapshot this consumer's wanted streams that rendered this step, into host FramePackets.

        One ``get_camera_data(device="cpu")`` read per (camera, modality) — the cache shares that
        single device->host copy with every other consumer reading the same buffer this step."""
        manager = self.simulator.sensor_manager
        if manager is None:
            return {}
        fresh = manager.last_due  # set[str] of camera names rendered this step
        sim_time = self.simulator.time()

        by_cam_mod: dict[tuple[str, str], list[int]] = {}
        for cam, mod, env in self._wanted:
            if cam in fresh:
                by_cam_mod.setdefault((cam, mod), []).append(env)

        packets: dict[StreamKey, FramePacket] = {}
        for (cam, mod), envs in by_cam_mod.items():
            buf = self.simulator.get_camera_data(cam, mod, device="cpu")  # [N, H, W, C] host, cached+shared
            intr = self._intrinsics_of(cam)
            host = buf.detach().numpy()
            for env in envs:
                packets[(cam, mod, env)] = FramePacket(
                    camera=cam, modality=mod, env_id=env, array=host[env], sim_time=sim_time, intrinsics=intr
                )
        return packets

    def _on_close(self) -> None:
        """CLOSE callback: tear down, isolating any failure so other close hooks still run."""
        try:
            self.stop()
        except Exception as exc:
            logger.error(f"Camera consumer {self._label()} stop failed: {exc}")
