"""The single implementation of depth-frame preprocessing and training-time depth augmentation.

Training observation terms and deployment-side depth producers must apply the same transform,
or the policy sees inputs it was never trained on. Keep every producer on these functions.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from holosoma.config_types.depth import DepthAugmentationConfig, DepthProcessingConfig
from holosoma.utils.perlin_noise import InfiniteFractalPerlin3D

# Perlin octave layout used for depth holes: one octave on a 2x2-cell lattice per 64x96 frame.
_HOLE_RESOLUTIONS = [(2, 2), (4, 4), (8, 8), (16, 16), (32, 32)]
_HOLE_PERIODS = [32, 16, 8, 4, 2]
_HOLE_FACTORS = [1.0]

_SOBEL_X = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
_SOBEL_Y = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]


def processed_shape(height: int, width: int, cfg: DepthProcessingConfig) -> tuple[int, int]:
    """Output ``(height, width)`` of :func:`resize_depth` for a ``height x width`` frame."""
    if cfg.resize is not None:
        return int(cfg.resize[0]), int(cfg.resize[1])
    out = (height - cfg.crop_top - cfg.crop_bottom, width - cfg.crop_left - cfg.crop_right)
    if min(out) <= 0:
        raise ValueError(f"Depth crop removes the whole {height}x{width} frame: {cfg}.")
    return out


def resize_depth(depth: torch.Tensor, cfg: DepthProcessingConfig) -> torch.Tensor:
    """Metric ``[N, H, W]`` depth to the cropped, resized, range-limited metric frame.

    Everything in :class:`DepthProcessingConfig` except the final normalization.
    """
    near, far = cfg.near, cfg.far
    depth = torch.nan_to_num(depth, nan=far, posinf=far, neginf=near).clamp(near, far)
    height, width = depth.shape[-2:]
    processed_shape(height, width, cfg)  # validates the crop
    depth = depth[
        :,
        cfg.crop_top : height - cfg.crop_bottom,
        cfg.crop_left : width - cfg.crop_right,
    ]
    if cfg.resize is not None:
        depth = F.interpolate(
            depth.unsqueeze(1),
            size=(int(cfg.resize[0]), int(cfg.resize[1])),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).squeeze(1)
    depth = depth.clamp(max=far)
    return torch.where(depth < cfg.empty_threshold, torch.full_like(depth, far), depth)


def normalize_depth(depth: torch.Tensor, cfg: DepthProcessingConfig) -> torch.Tensor:
    """Map metric depth to the network input range, ``near -> -0.5`` and ``far -> 0.5``."""
    return (depth - cfg.near) / (cfg.far - cfg.near) - 0.5


def preprocess_depth(depth: torch.Tensor, cfg: DepthProcessingConfig) -> torch.Tensor:
    """Metric ``[N, H, W]`` depth to the normalized network input, without augmentation."""
    return normalize_depth(resize_depth(depth, cfg), cfg)


class DepthAugmenter:
    """Applies :class:`DepthAugmentationConfig` artifacts to resized metric depth frames.

    Holds the state some effects need across frames: the per-env offset (drawn at creation from
    the global RNG) and the Perlin hole sequence. Every call advances that state.
    """

    def __init__(
        self,
        cfg: DepthAugmentationConfig,
        num_envs: int,
        far: float,
        device: torch.device | str,
    ) -> None:
        self.cfg = cfg
        self.far = far
        self.device = torch.device(device)
        self._holes: InfiniteFractalPerlin3D | None = None
        if cfg.holes.enabled:
            self._holes = InfiniteFractalPerlin3D(
                (int(cfg.holes.noise_shape[0]), int(cfg.holes.noise_shape[1])),
                _HOLE_RESOLUTIONS,
                _HOLE_PERIODS,
                _HOLE_FACTORS,
                generator=torch.Generator(device=self.device),
                batch_size=num_envs,
                seed=cfg.holes.seed,
            )
        self._offset = (
            torch.randn(num_envs, device=self.device) * cfg.per_env_offset.std if cfg.per_env_offset.enabled else None
        )

    def __call__(self, depth: torch.Tensor) -> torch.Tensor:
        """Augment resized metric depth ``[N, H, W]`` (one frame per env)."""
        depth = depth.clone()
        if self.cfg.edge_noise.enabled:
            depth = self._edge_noise(depth)
        if self._holes is not None:
            depth = self._apply_holes(depth)
        if self.cfg.gaussian_noise.enabled:
            depth += torch.randn_like(depth) * self.cfg.gaussian_noise.std
        if self._offset is not None:
            depth += self._offset[:, None, None]
        return depth

    def _edge_noise(self, depth: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg.edge_noise
        n, height, width = depth.shape
        kernel = {"device": depth.device, "dtype": depth.dtype}
        sobel_x = torch.tensor(_SOBEL_X, **kernel).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor(_SOBEL_Y, **kernel).unsqueeze(0).unsqueeze(0)
        image = depth.unsqueeze(1)
        gx = F.conv2d(image, sobel_x, padding=1)
        gy = F.conv2d(image, sobel_y, padding=1)
        gradient = torch.sqrt(gx * gx + gy * gy).squeeze(1)

        def interior(mask: torch.Tensor) -> torch.Tensor:
            b = cfg.border
            mask[:, :, :b] = False
            mask[:, :, -b:] = False
            mask[:, :b, :] = False
            mask[:, -b:, :] = False
            return mask

        # One uniform draw per pixel decides both the shuffle and the drop.
        draw = torch.rand_like(gradient)
        shuffle = interior(gradient > cfg.shuffle_gradient_threshold) & (draw < cfg.shuffle_probability)
        shuffle = F.max_pool2d(shuffle.unsqueeze(1).float(), kernel_size=3, stride=1, padding=1).squeeze(1).bool()
        padded = F.pad(depth.unsqueeze(1), (1, 1, 1, 1), mode="circular")
        neighbors = F.unfold(padded, kernel_size=3, padding=0, stride=1).view(n, 9, height, width)
        choice = torch.randint(0, 9, (n, 1, height, width), device=depth.device)
        depth = torch.where(shuffle, torch.gather(neighbors, 1, choice).squeeze(1), depth)

        local_max = F.max_pool2d(depth.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
        drop = interior((gradient > cfg.drop_gradient_threshold) & (local_max > cfg.drop_min_depth))
        depth[drop & (draw < cfg.drop_probability)] = self.far
        return depth

    def _apply_holes(self, depth: torch.Tensor) -> torch.Tensor:
        assert self._holes is not None
        cfg = self.cfg.holes
        noise = self._holes.generate_frame()
        noise = F.interpolate(noise.unsqueeze(1), size=depth.shape[-2:], mode="bilinear", align_corners=False)
        noise = F.max_pool2d(noise, kernel_size=3, stride=1, padding=1)
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
        holes = (noise.squeeze(1) < cfg.threshold) & (depth < cfg.max_depth) & (depth > cfg.min_depth)
        depth[holes] = self.far
        return depth
