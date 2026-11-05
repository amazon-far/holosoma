"""Optional local visualization for fresh LiDAR point-cloud scans.

The plugin is intentionally a :class:`LidarConsumerPlugin`: it visualizes host snapshots after
the normal sensor lifecycle has produced them, so enabling it cannot alter backend ray casts or
scan cadence. Matplotlib renders through its off-screen canvas; live windows use the same OpenCV
HighGUI path as camera visualization.
"""

from __future__ import annotations

import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import numpy.typing as npt
from loguru import logger

from holosoma.config_types.sensor import LidarSensorConfig
from holosoma.simulator.plugins.lidar_consumer import LidarConsumerPlugin, LidarStreamKey, PointCloudPacket
from holosoma.utils.video_utils import create_video

if TYPE_CHECKING:
    from holosoma.config_types.plugin import LidarVizPluginConfig
    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator


class LidarVizPlugin(LidarConsumerPlugin):
    """Display and/or save range-colored optical-frame LiDAR point clouds for debugging."""

    config: LidarVizPluginConfig

    def __init__(self, config: LidarVizPluginConfig, simulator: BaseSimulator) -> None:
        lidars = {
            name: sensor for name, sensor in simulator.sensor_config.items() if isinstance(sensor, LidarSensorConfig)
        }
        self._lidar_names = config.lidars if config.lidars is not None else list(lidars)
        self._env_ids = list(config.env_ids)
        self._show_live = False

        self._figure_cls: Any | None = None
        self._canvas_cls: Any | None = None
        self._open_windows: dict[LidarStreamKey, str] = {}
        self._closed_windows: set[LidarStreamKey] = set()
        self._scan_index: defaultdict[LidarStreamKey, int] = defaultdict(int)
        self._video_frames: defaultdict[LidarStreamKey, list[npt.NDArray[np.uint8]]] = defaultdict(list)
        self._video_times: defaultdict[LidarStreamKey, list[float]] = defaultdict(list)

        super().__init__(config, simulator)

    def wanted_streams(self) -> set[LidarStreamKey]:
        return {(lidar, env_id) for lidar in self._lidar_names for env_id in self._env_ids}

    def start(self) -> None:
        self._show_live = self.config.live_window and not self.simulator.headless and bool(os.environ.get("DISPLAY"))
        if self.config.live_window and not self._show_live:
            logger.warning(
                "LidarVizPlugin: live_window requested but simulator is headless or no display is available."
            )
        if not (self._show_live or self.config.save_scans or self.config.record_video):
            logger.warning(
                "LidarVizPlugin selected with no usable output; enable live_window, save_scans, or record_video."
            )
            return

        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        self._figure_cls = Figure
        self._canvas_cls = FigureCanvasAgg
        logger.info(
            f"LidarVizPlugin active: lidars={self._lidar_names} envs={self._env_ids} "
            f"live_window={self._show_live} save_scans={self.config.save_scans} record_video={self.config.record_video}"
        )

    def publish(self, clouds: dict[LidarStreamKey, PointCloudPacket]) -> None:
        if self._figure_cls is None:
            return

        for key, packet in clouds.items():
            points = self._finite_points(packet.points)
            if not len(points):
                logger.warning(
                    f"LidarVizPlugin: LiDAR '{packet.lidar}' env {packet.env_id} scan has no finite points; skipped."
                )
                continue

            index = self._scan_index[key]
            self._scan_index[key] += 1
            title = f"{packet.lidar} env {packet.env_id} scan {index} t={packet.sim_time:.3f}s"

            frame = self._render(points, title)
            if self._show_live and key not in self._closed_windows:
                self._show_frame(key, packet, frame.rgb)
            if self.config.save_scans:
                self._save_frame(frame.figure, packet, index)
            if self.config.record_video:
                self._video_frames[key].append(frame.rgb)
                self._video_times[key].append(packet.sim_time)

    def _finite_points(self, points: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        xyz = np.asarray(points, dtype=np.float32)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"LidarVizPlugin expected a point cloud shaped [N, 3], got {xyz.shape}.")
        finite = np.ascontiguousarray(xyz[np.isfinite(xyz).all(axis=1)])
        max_points = self.config.max_points
        if max_points is not None and len(finite) > max_points:
            stride = math.ceil(len(finite) / max_points)
            finite = finite[::stride]
        return finite

    def _render(self, points: npt.NDArray[np.float32], title: str) -> _RenderedScan:
        assert self._figure_cls is not None and self._canvas_cls is not None
        figure = self._figure_cls(figsize=(7, 6))
        canvas = self._canvas_cls(figure)
        axes = figure.add_subplot(111, projection="3d")
        self._draw(axes, points, title)
        figure.tight_layout()
        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba())
        return _RenderedScan(figure=figure, rgb=np.ascontiguousarray(rgba[..., :3]))

    def _show_frame(
        self,
        key: LidarStreamKey,
        packet: PointCloudPacket,
        rgb: npt.NDArray[np.uint8],
    ) -> None:
        window = f"holosoma LiDAR: {packet.lidar} env {packet.env_id}"
        if key in self._open_windows and cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            self._open_windows.pop(key, None)
            self._closed_windows.add(key)
            return
        cv2.imshow(window, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.pollKey()
        self._open_windows[key] = window

    def _save_frame(self, figure: Any, packet: PointCloudPacket, index: int) -> None:
        path = self._save_dir() / self._path_component(packet.lidar) / f"env_{packet.env_id:03d}" / f"{index:06d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=160)

    def _draw(self, axes: Any, points: npt.NDArray[np.float32], title: str) -> None:
        axes.clear()
        ranges = np.linalg.norm(points, axis=1)
        axes.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=ranges,
            cmap="viridis",
            s=self.config.point_size,
            linewidths=0,
        )
        center = (points.min(axis=0) + points.max(axis=0)) / 2.0
        half_extent = max(float(np.ptp(points, axis=0).max()) / 2.0, 0.1)
        axes.set(
            xlim=(center[0] - half_extent, center[0] + half_extent),
            ylim=(center[1] - half_extent, center[1] + half_extent),
            zlim=(center[2] - half_extent, center[2] + half_extent),
            xlabel="sensor X, right (m)",
            ylabel="sensor Y, up (m)",
            zlabel="sensor Z, back (m)",
            title=title,
        )
        axes.set_box_aspect((1, 1, 1))
        axes.view_init(elev=self.config.view_elevation, azim=self.config.view_azimuth)
        axes.grid(alpha=0.25)

    def stop(self) -> None:
        if self._figure_cls is None:
            return
        if self.config.record_video:
            for key, frames in self._video_frames.items():
                if not frames:
                    continue
                lidar, env_id = key
                create_video(
                    np.stack(frames),
                    fps=self._video_fps(self._video_times[key]),  # type: ignore[arg-type]
                    save_dir=self._stream_dir(lidar, env_id),
                    output_format="h264",
                    wandb_logging=False,
                )
        self._video_frames.clear()
        self._video_times.clear()
        for window in self._open_windows.values():
            cv2.destroyWindow(window)
        self._open_windows.clear()
        self._closed_windows.clear()

    def _save_dir(self) -> Path:
        if self.config.save_dir is not None:
            return Path(self.config.save_dir)
        video_dir = self.simulator.video_config.save_dir
        return Path(video_dir) if video_dir else Path("logs/lidar_sensors")

    def _stream_dir(self, lidar: str, env_id: int) -> Path:
        return self._save_dir() / self._path_component(lidar) / f"env_{env_id:03d}"

    def _video_fps(self, sim_times: list[float]) -> float:
        if len(sim_times) < 2:
            return self.simulator.sim_dt**-1 * self.config.playback_rate
        deltas = np.diff(np.asarray(sim_times, dtype=np.float64))
        valid = deltas[np.isfinite(deltas) & (deltas > 0.0)]
        base_fps = self.simulator.sim_dt**-1 if not len(valid) else 1.0 / float(np.median(valid))
        return base_fps * self.config.playback_rate

    @staticmethod
    def _path_component(value: str) -> str:
        """Return a stable filesystem-safe representation of a user-provided sensor name."""
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


class _RenderedScan:
    """One off-screen Matplotlib render shared by optional PNG and MP4 outputs."""

    def __init__(self, figure: Any, rgb: npt.NDArray[np.uint8]) -> None:
        self.figure = figure
        self.rgb = rgb
