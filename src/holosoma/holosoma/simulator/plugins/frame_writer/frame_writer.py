"""Frame-writer plugin: dump rendered camera frames to disk as image sequences or per-stream mp4s.

A :class:`CameraConsumerPlugin` for dataset capture: every frame of each watched
``(camera, modality, env)`` stream is written inline on the sim thread (lossless, no drops —
the sim waits on the disk write, unlike the async ROS2 egress). RGB goes out as PNG, JPEG, or —
with ``rgb_format="mp4"`` — one H.264 video per stream (``<camera>_<modality>_env<k>.mp4``),
streamed into an ffmpeg process per stream as frames arrive and finalized at stop; depth as
16-bit-millimeter PNG or raw ``.npy`` float32 meters. An
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
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, TextIO, cast

import cv2
import numpy as np
import numpy.typing as npt
from loguru import logger

from holosoma.config_types.frequency import resolve_decimation
from holosoma.config_types.sensor import CameraSensorConfig
from holosoma.simulator.plugins.camera_consumer import CameraConsumerPlugin
from holosoma.utils.video_utils import H264StreamEncoder, retime_mp4

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
        # mp4 mode: one open encoder per rgb stream, the sim-time span of the frames it has taken,
        # and the error of any stream whose encoder died (its later frames are dropped).
        self._encoders: dict[StreamKey, H264StreamEncoder] = {}
        self._video_span: dict[StreamKey, tuple[float, float]] = {}
        self._video_failed: dict[StreamKey, str] = {}
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
                path = self._out / f"{stream_name}.mp4"
                self._encode_frame(key, path, packet)
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
        """Finalize every open mp4 and close the index.

        Called during exception teardown (an interrupt, a crash, SIGTERM's ``SystemExit``), each mp4
        keeps its ``.partial`` name, so a truncated episode is never mistaken for a complete one. A
        ``SystemExit`` with a success status (code ``0`` or ``None``) is a clean exit and publishes.
        Raises ``RuntimeError`` after closing everything if any stream's encode failed.
        """
        active = sys.exc_info()[1]
        clean_exit = isinstance(active, SystemExit) and active.code in (0, None)
        interrupted = active is not None and not clean_exit
        errors = [f"{self._stream_name(key)}: {err}" for key, err in sorted(self._video_failed.items())]
        for encoder in self._encoders.values():
            encoder.end_input()  # every stream flushes at once; the loop below waits on each
        errors += [
            err for key, encoder in sorted(self._encoders.items()) if (err := self._close(key, encoder, interrupted))
        ]
        self._encoders = {}
        self._video_span = {}
        self._video_failed = {}
        if self._index is not None:
            self._index.close()
            self._index = None
        total = sum(self._counts.values())
        logger.info(f"FrameWriterPlugin wrote {total} frame(s) across {len(self._counts)} stream(s).")
        if errors:
            raise RuntimeError("FrameWriterPlugin: mp4 encoding failed for " + "; ".join(errors))

    @staticmethod
    def _stream_name(key: StreamKey) -> str:
        cam, mod, env = key
        return f"{cam}_{mod}_env{env}"

    def _close(self, key: StreamKey, encoder: H264StreamEncoder, interrupted: bool) -> str | None:
        """Finalize one stream's video; on failure stop its encoder and return the error line."""
        try:
            self._finish_video(key, encoder, interrupted)
        except Exception as exc:
            encoder.abort()
            return f"{self._stream_name(key)}: {exc}"
        return None

    def _encode_frame(self, key: StreamKey, path: Path, packet: FramePacket) -> None:
        """Feed one rgb frame to its stream's encoder, opening the encoder on the stream's first frame.

        A failure disables only that stream: it is logged once here and reported again by ``stop``.
        """
        if key in self._video_failed:
            return
        encoder = self._encoders.get(key)
        try:
            if encoder is None:
                encoder = self._open_encoder(key, path, packet)
                self._encoders[key] = encoder
            encoder.write(cast("npt.NDArray[np.uint8]", packet.array))
        except Exception as exc:
            if encoder is not None:
                encoder.abort()
            self._encoders.pop(key, None)
            self._video_failed[key] = str(exc)
            logger.error(f"FrameWriterPlugin: {self._stream_name(key)} mp4 encoding failed; dropping its frames: {exc}")
            return
        first = self._video_span.get(key, (packet.sim_time, packet.sim_time))[0]
        self._video_span[key] = (first, packet.sim_time)

    def _open_encoder(self, key: StreamKey, path: Path, packet: FramePacket) -> H264StreamEncoder:
        """Start a stream's encoder at its camera's render rate (control rate / update decimation)."""
        cam = key[0]
        decimation = resolve_decimation(
            self.sensors_config[cam].update_decimation, self.control_hz, field=f"camera '{cam}' update_decimation"
        )
        height, width = packet.array.shape[:2]
        return H264StreamEncoder(
            path,
            width,
            height,
            self.control_hz / decimation,
            codec=self.config.video_codec,
            crf=self.config.video_crf,
            preset=self.config.video_preset,
            gop=self.config.video_gop,
            nvenc_max_sessions=self.config.video_nvenc_max_sessions,
        )

    def _finish_video(self, key: StreamKey, encoder: H264StreamEncoder, interrupted: bool) -> None:
        """Close one stream's encoder: drop a <2-frame stream, keep ``.partial`` if interrupted, else publish.

        The playback rate is checked against the frames' own sim-time stamps, ``(n - 1) / (t_last -
        t_first)``; if the render-rate guess the encoder opened with is off, the timestamps are rescaled
        losslessly so the video plays at sim speed.
        """
        name = self._stream_name(key)
        n = encoder.frames
        if n < 2:
            encoder.discard()
            logger.warning(f"FrameWriterPlugin: stream {name} has {n} frame(s); no mp4.")
            return
        partial = encoder.finish(publish=False)
        if interrupted:
            logger.warning(f"FrameWriterPlugin: run interrupted; left {partial} ({n} frames) incomplete.")
            return
        first, last = self._video_span[key]
        if last > first:
            fps = (n - 1) / (last - first)
            if abs(fps / encoder.fps - 1.0) > 1e-3:
                logger.warning(
                    f"FrameWriterPlugin: {name} frames span {fps:.6g} fps of sim time, not the "
                    f"{encoder.fps:.6g} fps render rate; retiming the mp4 to match."
                )
                retime_mp4(partial, encoder.fps / fps)
        else:
            fps = encoder.fps
            logger.warning(f"FrameWriterPlugin: {name} frames share one sim time; kept the {fps:.6g} fps render rate.")
        partial.replace(encoder.path)
        logger.info(f"FrameWriterPlugin wrote video: {encoder.path} ({n} frames @ {fps:.4g} fps, {encoder.codec})")

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
