"""Depth-image encoders for distillation students.

Each backbone maps one depth frame ``[batch, height, width]`` (or ``[batch, 1, height, width]``)
to a latent of ``scandots_output_dim``, followed by an ELU. They are built from
``StudentTeacherModuleConfig.depth_backbone`` as ``cls(depth_output_dim)``.
"""

from __future__ import annotations

import torch
from torch import nn

# from https://github.com/CAI23sbP/Isaaclab_Parkour


def _prepare_images(images: torch.Tensor) -> torch.Tensor:
    if images.ndim == 3:
        images = images.unsqueeze(1)
    if images.ndim != 4 or images.shape[1] != 1:
        raise ValueError(
            f"Expected depth images with shape [batch, 1, height, width] or [batch, height, width], "
            f"got {tuple(images.shape)}."
        )
    return images


class DepthOnlyFCBackbone58x87(nn.Module):
    """Conv encoder with a fully connected head sized for exactly 58x87 frames."""

    def __init__(self, scandots_output_dim: int) -> None:
        super().__init__()

        activation = nn.ELU()
        self.image_compression = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            activation,
            nn.Flatten(),
            nn.Linear(64 * 25 * 39, 128),
            activation,
            nn.Linear(128, scandots_output_dim),
        )
        self.output_activation = activation

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.output_activation(self.image_compression(_prepare_images(images)))


class DepthOnlyFCBackbone58x87Small(nn.Module):
    """Strided conv encoder with global average pooling.

    Designed for 58x87 frames (the shapes noted below), but the pooling makes it accept any
    frame that survives three stride-2 convolutions. The name is kept for existing configs.
    """

    def __init__(self, scandots_output_dim: int) -> None:
        super().__init__()

        activation = nn.ELU()
        self.image_compression = nn.Sequential(
            # Input: (B, 1, 58, 87)
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=2, padding=2),
            # (58, 87) -> (29, 44)
            activation,
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            # (29, 44) -> (15, 22)
            activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            # (15, 22) -> (8, 11)
            activation,
            # Global average pooling over spatial dims -> (B, 64, 1, 1)
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),  # (B, 64)
            nn.Linear(64, scandots_output_dim),
        )
        self.output_activation = activation

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.output_activation(self.image_compression(_prepare_images(images)))
