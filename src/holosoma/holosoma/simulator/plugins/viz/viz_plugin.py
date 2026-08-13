"""Local-visualization plugin: tile mounted-camera views into a live cv2 window and/or mp4.

A :class:`CameraConsumerPlugin`: the base snapshots its wanted ``(camera, modality, env)`` streams
once per control step (the single, shared GPU→host copy via the cached ``get_camera_data``) and hands
them to :meth:`publish` as a batch; this class colorizes depth, tiles the panels into one grid
(rows = envs, cols = (camera, modality)), and shows it live and/or buffers it for an H.264 file at
:meth:`stop`. cv2 + video utils are imported at module top — this module is reached only via
``CameraVizPluginConfig.get_cls``, so importing the plugins package stays cv2-free.

Inherently inline (composes + shows/buffers on the calling thread) — no async worker path: a live
cv2 window must be driven where the loop runs, and the per-step grid is cheap.
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from loguru import logger

from holosoma.config_types.frequency import is_frequency_string, resolve_decimation
from holosoma.simulator.plugins.camera_consumer import CameraConsumerPlugin
from holosoma.simulator.plugins.viz.image_grid import colorize_depth, tile_images
from holosoma.utils.video_utils import create_video

if TYPE_CHECKING:
    from holosoma.config_types.plugin import CameraVizPluginConfig
    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator
    from holosoma.simulator.plugins.camera_consumer import FramePacket, StreamKey

_WINDOW = "holosoma camera sensors"


class CameraVizPlugin(CameraConsumerPlugin):
    """Tiling cv2-window / mp4 sink over mounted-camera frames (a camera-consumer plugin)."""

    config: CameraVizPluginConfig

    def __init__(self, config: CameraVizPluginConfig, simulator: BaseSimulator) -> None:
        self._frames_video: list[np.ndarray] = []  # buffered grids for the mp4
        self._step = -1  # batches seen (proxy for the fastest watched camera's render count)
        self._last_captured = -1

        # Cameras to watch: configured selection or all cameras in the active camera dict. Read the
        # sim's camera dict directly (the sensors_config property needs self.simulator, set by super
        # below — but wanted_streams runs inside super().__init__, so the panels must exist first).
        cams_by_name = dict(simulator.sensor_config)
        all_cams = list(cams_by_name)
        self._cam_names = config.cameras if config.cameras is not None else all_cams

        # A PANEL is one (camera, modality): its own grid column. For each watched camera take the
        # modalities it actually produces, intersected with the config's modality selection.
        self._panels: list[tuple[str, str]] = []
        for name in self._cam_names:
            cam_mods = list(cams_by_name[name].data_types)
            mods = cam_mods if config.modalities is None else [m for m in config.modalities if m in cam_mods]
            self._panels.extend((name, m) for m in mods)
        if not self._panels:
            logger.warning(f"CameraVizPlugin: selected modalities {config.modalities} match no camera output.")

        # Panel label disambiguation: append ":modality" only for cameras shown with >1 modality.
        per_cam = Counter(name for name, _ in self._panels)
        self._multi_modality = {name for name, n in per_cam.items() if n > 1}

        self._env_ids = list(config.env_ids)
        self._depth_range = (config.depth_range[0], config.depth_range[1]) if config.depth_range else (0.01, 5.0)

        # Panels + env_ids are set, so wanted_streams() works: validate streams and register the
        # publish/close callbacks. Fields below use self.config / self.control_hz (set by super).
        super().__init__(config, simulator)

        self._frame_decimation = self._resolve_frame_decimation()

        # Live window viable only with a non-headless sim AND a usable display.
        self._show_live = config.live_window and not simulator.headless and bool(os.environ.get("DISPLAY"))
        if config.live_window and not self._show_live:
            logger.warning("CameraVizPlugin: live_window requested but no display; window disabled.")
        self._window_open = False
        self._cell_wh = {name: (c.width, c.height) for name, c in cams_by_name.items()}

    def _resolve_frame_decimation(self) -> int:
        # update_decimation is in units of the fastest watched camera's RENDERED frames. An int is
        # already that; a frequency string is a target against the control rate, converted to frames
        # via the fastest watched camera's render decimation (d_min control steps between its frames).
        cfg = self.config
        if not is_frequency_string(cfg.update_decimation):
            return int(cfg.update_decimation)
        # We do not see SensorManager here; approximate d_min as 1 (publish() already only fires on
        # steps a panel rendered, so the batch cadence is the fastest camera's frame cadence).
        n_viz = resolve_decimation(cfg.update_decimation, self.control_hz, field="recorder update_decimation")
        return max(1, n_viz)

    def wanted_streams(self) -> set[StreamKey]:
        return {(cam, mod, env) for (cam, mod) in self._panels for env in self._env_ids}

    def start(self) -> None:
        logger.info(
            f"CameraVizPlugin active: panels={[f'{n}:{m}' for n, m in self._panels]} "
            f"envs={self._env_ids} live_window={self._show_live} record_video={self.config.record_video}"
        )

    def publish(self, frames: dict[StreamKey, FramePacket]) -> None:
        if not self._panels or not (self._show_live or self.config.record_video):
            return
        # Batch arrival == this step's fastest watched camera rendered. Gate by frame-decimation.
        self._step += 1
        if self._step - self._last_captured < self._frame_decimation:
            return
        self._last_captured = self._step

        views: list[np.ndarray] = []
        labels: list[str] = []
        for env in self._env_ids:
            for name, modality in self._panels:
                packet = frames.get((name, modality, env))
                if packet is None:
                    w, h = self._cell_wh.get(name, (128, 128))
                    views.append(self._missing_tile(w, h))
                else:
                    img = packet.array
                    if modality == "depth":
                        img = colorize_depth(img, self._depth_range, self.config.depth_colormap)
                    views.append(img)
                label = f"env{env}/{name}"
                labels.append(f"{label}:{modality}" if name in self._multi_modality else label)
        grid = tile_images(views, layout=(len(self._env_ids), len(self._panels)), labels=labels)  # RGB

        if self._show_live:
            self._show(grid)
        if self.config.record_video:
            self._frames_video.append(grid)

    def _show(self, grid: np.ndarray) -> None:
        cv2.imshow(_WINDOW, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        cv2.pollKey()  # non-blocking HighGUI pump (unlike waitKey(1))
        if self._window_open and cv2.getWindowProperty(_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            self._show_live = False  # user closed it -> stop drawing live for the rest of the run
            self._window_open = False
        else:
            self._window_open = True

    @staticmethod
    def _missing_tile(width: int, height: int, cell: int = 16) -> np.ndarray:
        """Source-style magenta/black 'missing texture' for a panel with no frame yet (RGB)."""
        ys = (np.arange(height) // cell)[:, None]
        xs = (np.arange(width) // cell)[None, :]
        tile = np.zeros((height, width, 3), dtype=np.uint8)
        tile[(ys + xs) % 2 == 1] = (255, 0, 255)
        return tile

    def stop(self) -> None:
        if self.config.record_video and self._frames_video:
            # Capture cadence = the recorder's frame-decimation (batches already track the fastest
            # camera's render rate). Encode fps so the video plays at true wall-clock speed.
            fps = self.control_hz / self._frame_decimation * self.config.playback_rate
            create_video(
                np.array(self._frames_video, dtype=np.uint8),
                fps=fps,
                save_dir=str(self._save_dir()),
                output_format="h264",
                wandb_logging=False,
            )
            self._frames_video = []
        if self._window_open:
            cv2.destroyWindow(_WINDOW)
            self._window_open = False

    def _save_dir(self) -> Path:
        if self.config.save_dir is not None:
            return Path(self.config.save_dir)
        spectator_dir = self.simulator.video_config.save_dir
        return Path(spectator_dir) if spectator_dir else Path("logs/camera_sensors")
