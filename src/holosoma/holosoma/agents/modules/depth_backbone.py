from __future__ import annotations

from typing import Any

import torch
from torch import nn

# from https://github.com/CAI23sbP/Isaaclab_Parkour


def _prepare_images(images: torch.Tensor, num_frames: int) -> torch.Tensor:
    if images.ndim == 3:
        images = images.unsqueeze(1)
    if images.ndim != 4 or images.shape[1] != num_frames:
        raise ValueError(
            f"Expected depth images with shape [batch, {num_frames}, height, width] "
            f"or [batch, height, width] for one frame, got {tuple(images.shape)}."
        )
    return images


class DepthOnlyFCBackbone58x87(nn.Module):
    def __init__(self, scandots_output_dim: int, output_activation: str | None = None, num_frames: int = 1) -> None:
        super().__init__()

        self.num_frames = num_frames
        activation = nn.ELU()
        self.image_compression = nn.Sequential(
            nn.Conv2d(in_channels=self.num_frames, out_channels=32, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            activation,
            nn.Flatten(),
            nn.Linear(64 * 25 * 39, 128),
            activation,
            nn.Linear(128, scandots_output_dim),
        )

        if output_activation == "tanh":
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = activation

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images_compressed = self.image_compression(_prepare_images(images, self.num_frames))
        return self.output_activation(images_compressed)


class DepthOnlyFCBackbone58x87Small(nn.Module):
    def __init__(self, scandots_output_dim: int, output_activation: str | None = None, num_frames: int = 1) -> None:
        super().__init__()

        self.num_frames = num_frames
        activation = nn.ELU()
        self.image_compression = nn.Sequential(
            # Input: (B, num_frames, 58, 87)
            nn.Conv2d(in_channels=self.num_frames, out_channels=16, kernel_size=5, stride=2, padding=2),
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

        if output_activation == "tanh":
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = activation

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images_compressed = self.image_compression(_prepare_images(images, self.num_frames))
        return self.output_activation(images_compressed)


class RecurrentDepthBackbone(nn.Module):
    def __init__(self, base_backbone: nn.Module, depth_cfg: dict[str, Any]) -> None:
        super().__init__()
        activation = nn.ELU()
        last_activation = nn.Tanh()
        self.base_backbone = base_backbone
        num_prop = depth_cfg["num_prop"]
        if num_prop is None:
            num_prop = 53
        self.combination_mlp = nn.Sequential(
            nn.Linear(32 + num_prop, 128),
            activation,
            nn.Linear(128, 32),
        )
        self.recurrent_size = 512

        self.rnn = nn.GRU(input_size=32, hidden_size=512, batch_first=True)
        self.output_mlp = nn.Sequential(
            nn.Linear(512, 32 + 2),
            last_activation,
        )
        self.hidden_states = torch.zeros(1, 0, 512)
        self.rnn.flatten_parameters()

    def forward(self, depth_image: torch.Tensor, proprioception: torch.Tensor) -> torch.Tensor:
        if self.hidden_states.shape[1] != depth_image.shape[0] or self.hidden_states.device != depth_image.device:
            self.hidden_states = depth_image.new_zeros(1, depth_image.shape[0], self.recurrent_size)

        depth_image = self.base_backbone(depth_image)
        depth_latent = self.combination_mlp(torch.cat((depth_image, proprioception), dim=-1))
        depth_latent, self.hidden_states = self.rnn(depth_latent[:, None, :], self.hidden_states)
        return self.output_mlp(depth_latent.squeeze(1))

    def detach_hidden_states(self) -> None:
        self.hidden_states = self.hidden_states.detach().clone()
