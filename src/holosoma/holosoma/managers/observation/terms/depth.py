"""Latency-delayed, preprocessed depth images from any mounted depth camera.

:class:`DelayedDepthImageObsTerm` reads a camera's metric depth through ``get_camera_data``,
preprocesses it with :mod:`holosoma.utils.depth_processing` (optionally adding sensor
artifacts), and serves each env a frame from a short history to model camera latency. Envs
that are reset (including teleports, which reset observations) have their history replaced by
the current frame, so they never see frames from before the reset.

Example term config (in a ``concatenate=False`` group)::

    ObsTermCfg(
        func="holosoma.managers.observation.terms.depth:DelayedDepthImageObsTerm",
        global_clip=False,  # an image term; the global clip only applies to 2-D observations
        params={
            "sensor": "head_depth",
            "latency_frames": [3, 4],
            "buffer_len": 6,
            "processing": {"near": 0.3, "far": 3.0, "crop_top": 2, "resize": [58, 87]},
            "augmentation": {"gaussian_noise": {"enabled": True}},
        },
    )
"""

from __future__ import annotations

from typing import Any

from holosoma.config_types.depth import DepthAugmentationConfig, DepthProcessingConfig
from holosoma.managers.observation.base import ObservationTermBase
from holosoma.utils.depth_processing import DepthAugmenter, normalize_depth, processed_shape, resize_depth
from holosoma.utils.safe_torch_import import torch


class DelayedDepthImageObsTerm(ObservationTermBase):
    """Preprocessed depth ``[N, H, W]`` from a mounted camera, delayed by a per-env latency.

    Parameters (via ``cfg.params``)
    -------------------------------
    sensor : str
        Name of a mounted camera that produces ``"depth"``.
    latency_frames : int | list[int]
        Fixed delay in control steps, or an inclusive ``[min, max]`` range sampled per env and
        step. Must be smaller than ``buffer_len``.
    buffer_len : int
        Number of past frames kept per env.
    processing : dict
        :class:`DepthProcessingConfig` fields.
    augmentation : dict, optional
        :class:`DepthAugmentationConfig` fields; every effect is off unless enabled.
    """

    def __init__(self, cfg: Any, env: Any) -> None:
        super().__init__(cfg, env)
        params = cfg.params
        self.num_envs = env.num_envs
        self.device = env.device
        self.sensor: str = params["sensor"]
        self.processing = DepthProcessingConfig(**params["processing"])
        augmentation = DepthAugmentationConfig(**params.get("augmentation", {}))

        latency = params["latency_frames"]
        self.latency_range = (int(latency[0]), int(latency[1])) if isinstance(latency, (list, tuple)) else None
        self.buffer_len = int(params["buffer_len"])
        max_latency = self.latency_range[1] if self.latency_range is not None else int(latency)
        min_latency = self.latency_range[0] if self.latency_range is not None else int(latency)
        if not 0 <= min_latency <= max_latency < self.buffer_len:
            raise ValueError(f"latency_frames {latency} must lie in [0, buffer_len={self.buffer_len}) with min <= max.")

        camera = env.simulator.sensor_config[self.sensor]
        self.frame_shape = processed_shape(camera.height, camera.width, self.processing)
        self.buffer = torch.zeros(self.num_envs, self.buffer_len, *self.frame_shape, device=self.device)
        # Latency of the latest full read, also used by row-subset reads.
        self.latency = torch.full((self.num_envs,), min_latency, dtype=torch.long, device=self.device)
        # Envs whose history must be replaced by the next frame; all of them before the first read.
        self.pending_refill = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.augmenter = DepthAugmenter(augmentation, self.num_envs, self.processing.far, self.device)

    # ----------------------------------------------------------- lifecycle

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.pending_refill[:] = True
        else:
            self.pending_refill[env_ids] = True

    # --------------------------------------------------------------- reads

    def _metric_frame(self, env: Any, rows: torch.Tensor | None) -> torch.Tensor:
        depth = env.simulator.get_camera_data(self.sensor, "depth")[..., 0].to(self.device)
        return resize_depth(depth if rows is None else depth[rows], self.processing)

    def _history_with(self, buffer: torch.Tensor, frame: torch.Tensor, refill: torch.Tensor) -> torch.Tensor:
        """``buffer`` after appending ``frame``, with the ``refill`` rows filled by it."""
        history = torch.cat([buffer[:, 1:], frame.unsqueeze(1)], dim=1)
        if bool(refill.any()):
            history[refill] = frame[refill].unsqueeze(1).expand(-1, self.buffer_len, -1, -1)
        return history

    def _select(self, history: torch.Tensor, latency: torch.Tensor) -> torch.Tensor:
        rows = torch.arange(history.shape[0], device=self.device)
        return history[rows, self.buffer_len - 1 - latency]

    def read(self, env: Any, env_ids: torch.Tensor | None = None, **kwargs: Any) -> torch.Tensor:
        """Capture the current frame for ``env_ids``, advance their history, and serve the delayed frame.

        A read of every env (``env_ids=None``, once per frame) augments the frame and resamples
        the latency. A read of a row subset, such as the terminal observation of envs about to
        reset, advances only those rows and keeps their latency. It leaves the frame unaugmented,
        because the Perlin hole sequence is shared by every env and advancing it for some envs
        would change the others. With a latency of at least one frame the served frame comes
        from the buffer, already augmented, so it matches a full read exactly.
        """
        if env_ids is None:
            frame = normalize_depth(self.augmenter(self._metric_frame(env, None)), self.processing)
            self.buffer = self._history_with(self.buffer, frame, self.pending_refill)
            self.pending_refill[:] = False
            if self.latency_range is not None:
                self.latency = torch.randint(
                    self.latency_range[0], self.latency_range[1] + 1, (self.num_envs,), device=self.device
                )
            return self._select(self.buffer, self.latency)
        if env_ids.numel() == 0:
            # How ObservationTermBase.shape sizes the term; reads no camera data.
            return self.buffer.new_zeros(0, *self.frame_shape)
        rows = env_ids.to(device=self.device, dtype=torch.long)
        frame = normalize_depth(self._metric_frame(env, rows), self.processing)
        history = self._history_with(self.buffer[rows], frame, self.pending_refill[rows])
        self.buffer[rows] = history
        self.pending_refill[rows] = False
        return self._select(history, self.latency[rows])
