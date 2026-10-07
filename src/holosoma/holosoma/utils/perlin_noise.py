from __future__ import annotations

import math
from collections.abc import Sequence
from typing import cast

import torch


class InfiniteFractalPerlin3D:
    """Generate a temporally continuous batch of multi-octave Perlin noise."""

    def __init__(
        self,
        shape: tuple[int, int],
        resolutions: Sequence[tuple[int, int]],
        periods: Sequence[float],
        factors: Sequence[float],
        batch_size: int = 1,
        generator: torch.Generator = torch.random.default_generator,
    ) -> None:
        if len(shape) != 2 or min(shape) <= 0:  # type: ignore[redundant-expr]
            raise ValueError(f"shape must contain two positive dimensions, got {shape}.")
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        if not factors:
            raise ValueError("At least one Perlin octave factor is required.")
        if len(resolutions) < len(factors) or len(periods) < len(factors):
            raise ValueError("Each Perlin octave factor requires a resolution and period.")

        octave_count = len(factors)
        self.shape = shape
        self.batch_size = batch_size
        self.resolutions = tuple(resolutions[:octave_count])
        self.periods = tuple(periods[:octave_count])
        self.factors = tuple(factors)
        if any(min(resolution) <= 0 for resolution in self.resolutions):
            raise ValueError(f"Perlin resolutions must be positive, got {self.resolutions}.")
        if any(period <= 0 for period in self.periods):
            raise ValueError(f"Perlin periods must be positive, got {self.periods}.")

        self.device = generator.device
        self.grid_shapes = [(shape[0] // resolution[0], shape[1] // resolution[1]) for resolution in self.resolutions]
        if any(min(grid_shape) <= 0 for grid_shape in self.grid_shapes):
            raise ValueError(f"Perlin resolutions cannot exceed shape {shape}: {self.resolutions}.")

        self.linys = [torch.linspace(0, 1, grid[0], device=self.device) for grid in self.grid_shapes]
        self.linxs = [torch.linspace(0, 1, grid[1], device=self.device) for grid in self.grid_shapes]
        self.masks: list[dict[tuple[int, int], torch.Tensor]] = []
        for linear_y, linear_x in zip(self.linys, self.linxs):
            fade_y = self.fade(linear_y)
            fade_x = self.fade(linear_x)
            octave_masks = {}
            for y_corner in range(2):
                for x_corner in range(2):
                    weight_y = fade_y if y_corner == 1 else torch.flip(fade_y, [0])
                    weight_x = fade_x if x_corner == 1 else torch.flip(fade_x, [0])
                    octave_masks[(y_corner, x_corner)] = weight_y[:, None] * weight_x[None, :]
            self.masks.append(octave_masks)

        self.gradient_cache: list[dict[int, torch.Tensor]] = [{} for _ in self.resolutions]
        self.frame_idx = 0

    @staticmethod
    def fade(value: torch.Tensor) -> torch.Tensor:
        return 6 * value**5 - 15 * value**4 + 10 * value**3

    def get_gradients(self, octave: int, z_index: int) -> torch.Tensor:
        cache = self.gradient_cache[octave]
        if z_index in cache:
            return cache[z_index]

        for cached_index in list(cache):
            if cached_index < z_index - 1:
                del cache[cached_index]

        seed = hash((octave, z_index)) % (2**31 - 1)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        resolution = self.resolutions[octave]
        gradients = torch.randn(
            (self.batch_size, resolution[0] + 2, resolution[1] + 2, 3),
            device=self.device,
            generator=generator,
        )
        gradients /= torch.norm(gradients, dim=-1, keepdim=True) + 1e-8
        cache[z_index] = gradients
        return gradients

    def generate_frame(self) -> torch.Tensor:
        frame_index = self.frame_idx
        self.frame_idx += 1
        noise = torch.zeros((self.batch_size, *self.shape), device=self.device)

        for octave, factor in enumerate(self.factors):
            z_value = frame_index / self.periods[octave]
            z_index = math.floor(z_value)
            z_fraction = z_value - z_index
            current_gradients = self.get_gradients(octave, z_index)
            next_gradients = self.get_gradients(octave, z_index + 1)
            linear_y = self.linys[octave]
            linear_x = self.linxs[octave]
            next_weight = self.fade(torch.as_tensor(z_fraction, device=self.device))
            current_weight = 1.0 - next_weight
            octave_noise = cast("torch.Tensor", 0.0)

            for temporal_corner in range(2):
                gradients = next_gradients if temporal_corner == 1 else current_gradients
                temporal_weight = next_weight if temporal_corner == 1 else current_weight
                temporal_offset = z_fraction - temporal_corner
                gradient_z = gradients[..., 0]
                gradient_y = gradients[..., 1]
                gradient_x = gradients[..., 2]
                x_positive = gradient_x[..., None, None] * linear_x
                y_positive = gradient_y[..., None, None] * linear_y[:, None]
                x_negative = -torch.flip(x_positive, dims=[-1])
                y_negative = -torch.flip(y_positive, dims=[-2])
                offset = (gradient_z * temporal_offset)[..., None, None]

                for y_corner in range(2):
                    for x_corner in range(2):
                        term_x = x_positive if x_corner == 0 else x_negative
                        term_y = y_positive if y_corner == 0 else y_negative
                        y_slice = slice(None, -1) if y_corner == 0 else slice(1, None)
                        x_slice = slice(None, -1) if x_corner == 0 else slice(1, None)
                        contribution = (offset + term_y + term_x)[:, y_slice, x_slice]
                        octave_noise += temporal_weight * self.masks[octave][(y_corner, x_corner)] * contribution

            resolution_y, resolution_x = self.resolutions[octave]
            grid_y, grid_x = self.grid_shapes[octave]
            octave_noise = octave_noise.permute(0, 1, 3, 2, 4).reshape(
                self.batch_size,
                (resolution_y + 1) * grid_y,
                (resolution_x + 1) * grid_x,
            )
            noise += factor * octave_noise[:, : self.shape[0], : self.shape[1]]

        return noise
