"""Frame-writer plugin: dump rendered camera frames to disk as image sequences or per-stream mp4s.

A :class:`CameraConsumerPlugin` for dataset capture: every frame of each watched
``(camera, modality, env)`` stream is written inline on the sim thread (lossless, no drops —
the sim waits on the disk write, unlike the async ROS2 egress). RGB goes out as PNG, JPEG, or —
with ``rgb_format="mp4"`` — one H.264 video per stream (``<camera>_<modality>_env<k>.mp4``,
encoded at stop); depth as 16-bit-millimeter PNG or raw ``.npy`` float32 meters. An
``index.jsonl`` per run records one line per frame (path, camera, modality, env, frame index,
sim time) plus a first header line with the camera intrinsics, so a downstream loader needs no
sidecar logic.

Scope: a simple cross-backend writer. For synthetic-data generation at scale, backend-native
writers (IsaacSim's Replicator handles modalities and async IO natively) would outperform this;
that would be a separate backend-specific plugin, not an extension of this one.

cv2 is imported at module top — this module is reached only via ``FrameWriterPluginConfig.get_cls``,
so the config layer stays cv2-free.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, TextIO, cast

import cv2
import numpy as np
import numpy.typing as npt
from loguru import logger

from holosoma.config_types.sensor import CameraSensorConfig
from holosoma.simulator.plugins.camera_consumer import CameraConsumerPlugin
from holosoma.utils.video_utils import create_video

if TYPE_CHECKING:
    from holosoma.config_types.plugin import FrameWriterPluginConfig
    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator
    from holosoma.simulator.plugins.camera_consumer import FramePacket, StreamKey


class FrameWriterPlugin(CameraConsumerPlugin):
    """Write every watched stream's frames to ``<output_dir>/<camera>_<modality>_env<k>/``."""

    config: FrameWriterPluginConfig

    def __init__(self, config: FrameWriterPluginConfig, simulator: BaseSimulator) -> None:
        # Streams to watch: configured cameras/modalities intersected with what the sim renders,
        # mirroring CameraVizPlugin. Must be set before super().__init__ (it calls wanted_streams).
        cams_by_name = {
            name: sensor for name, sensor in simulator.sensor_config.items() if isinstance(sensor, CameraSensorConfig)
        }
        cam_names = config.cameras if config.cameras is not None else list(cams_by_name)
        # training_config.num_envs, not simulator.num_envs: the latter is not yet populated when
        # IsaacSim constructs plugins during scene setup (same reason as the base validation).
        env_ids = config.env_ids if config.env_ids is not None else list(range(simulator.training_config.num_envs))
        self._streams: set[StreamKey] = set()
        for name in cam_names:
            cam_mods = list(cams_by_name[name].data_types)
            mods = cam_mods if config.modalities is None else [m for m in config.modalities if m in cam_mods]
            self._streams.update((name, m, env) for m in mods for env in env_ids)
        if not self._streams:
            raise ValueError(
                f"FrameWriterPlugin: no streams to write (cameras={config.cameras}, "
                f"modalities={config.modalities} match nothing among {sorted(cams_by_name)})."
            )
        self._counts: dict[StreamKey, int] = {}
        self._index: TextIO | None = None
        self._out = Path(config.output_dir)
        # mp4 mode: rgb frames buffer here per stream and encode at stop.
        self._video_frames: dict[StreamKey, list[npt.NDArray[np.uint8]]] = {}
        self._video_times: dict[StreamKey, list[float]] = {}
        super().__init__(config, simulator)

    def wanted_streams(self) -> set[StreamKey]:
        return self._streams

    def start(self) -> None:
        self._out.mkdir(parents=True, exist_ok=True)
        intrinsics = {}
        for cam in sorted({cam for cam, _, _ in self._streams}):
            camera_intrinsics = self._intrinsics_of(cam)
            serialized = asdict(camera_intrinsics)
            if camera_intrinsics.simulator_type is not None:
                serialized["simulator_type"] = camera_intrinsics.simulator_type.value
            intrinsics[cam] = serialized
        header = {
            "intrinsics": intrinsics,
            "rgb_format": self.config.rgb_format,
            "depth_format": self.config.depth_format,
        }
        self._index = (self._out / "index.jsonl").open("w")
        self._index.write(json.dumps(header) + "\n")
        logger.info(f"FrameWriterPlugin writing {len(self._streams)} stream(s) to {self._out}")

    def publish(self, frames: dict[StreamKey, FramePacket]) -> None:
        assert self._index is not None
        for key in sorted(frames):
            packet = frames[key]
            idx = self._counts.get(key, 0)
            self._counts[key] = idx + 1
            cam, mod, env = key
            stream_name = f"{cam}_{mod}_env{env}"
            if mod == "rgb" and self.config.rgb_format == "mp4":
                self._video_frames.setdefault(key, []).append(cast("npt.NDArray[np.uint8]", packet.array))
                self._video_times.setdefault(key, []).append(packet.sim_time)
                path = self._out / f"{stream_name}.mp4"
            else:
                stream_dir = self._out / stream_name
                if idx == 0:
                    stream_dir.mkdir(parents=True, exist_ok=True)
                path = stream_dir / f"{idx:06d}.{self.extension(mod)}"
                self.write_frame(path, mod, packet.array)
            self._index.write(
                json.dumps(
                    {
                        "path": str(path.relative_to(self._out)),
                        "camera": cam,
                        "modality": mod,
                        "env_id": env,
                        "frame": idx,
                        "sim_time": packet.sim_time,
                    }
                )
                + "\n"
            )

    def stop(self) -> None:
        for key, buffered in self._video_frames.items():
            self.encode_video(key, buffered, self._video_times[key])
        self._video_frames = {}
        self._video_times = {}
        if self._index is not None:
            self._index.close()
            self._index = None
        total = sum(self._counts.values())
        logger.info(f"FrameWriterPlugin wrote {total} frame(s) across {len(self._counts)} stream(s).")

    def encode_video(self, key: StreamKey, buffered: list[npt.NDArray[np.uint8]], times: list[float]) -> None:
        """Encode one stream's buffered rgb frames to ``<camera>_<modality>_env<k>.mp4`` (H.264)."""
        cam, mod, env = key
        if len(buffered) < 2:
            logger.warning(f"FrameWriterPlugin: stream {cam}_{mod}_env{env} has {len(buffered)} frame(s); no mp4.")
            return
        # Playback rate from the frames' own sim-time stamps, so the video plays at sim speed.
        fps = (len(times) - 1) / (times[-1] - times[0])
        written = create_video(
            np.array(buffered, dtype=np.uint8),
            fps=fps,  # type: ignore[arg-type]
            save_dir=self._out,
            output_format="h264",
            wandb_logging=False,
        )
        if written is not None:
            final = self._out / f"{cam}_{mod}_env{env}.mp4"
            written.replace(final)
            logger.info(f"FrameWriterPlugin wrote video: {final} ({len(buffered)} frames @ {fps:.4g} fps)")

    def extension(self, modality: str) -> str:
        """File extension for a modality's frames, per the configured format."""
        if modality == "depth":
            return "npy" if self.config.depth_format == "npy" else "png"
        return self.config.rgb_format

    def write_frame(self, path: Path, modality: str, array: npt.NDArray[np.uint8 | np.float32]) -> None:
        """Encode one frame to ``path`` (rgb as png/jpeg; depth as png16 millimeters or npy meters)."""
        if modality == "depth":
            depth = array[..., 0]  # [H, W] float32 meters, +inf = no hit
            if self.config.depth_format == "npy":
                np.save(path, depth)
            else:  # 16-bit millimeters; no-hit and >65.5m saturate to uint16 max
                mm = np.nan_to_num(depth * 1000.0, posinf=np.iinfo(np.uint16).max)
                cv2.imwrite(str(path), np.clip(mm, 0, np.iinfo(np.uint16).max).astype(np.uint16))
        else:  # rgb uint8 [H, W, 3] R,G,B; cv2 writes BGR
            bgr = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
            if self.config.rgb_format == "jpeg":
                cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality])
            else:
                cv2.imwrite(str(path), bgr)
