"""Encoder modules for processing sequential data like motion targets and height maps."""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class ConvEncoderConfig:
    """Configuration for Conv1D encoder."""
    
    input_dim: int
    """Input dimension per timestep."""
    
    hidden_dim: int = 60
    """Hidden dimension after initial linear layer."""
    
    output_dim: int = 128
    """Output dimension of the encoder."""
    
    num_timesteps: int = 20
    """Number of timesteps in the input sequence."""
    
    activation: str = "SiLU"
    """Activation function name."""


class Conv1DEncoder(nn.Module):
    """Conv1D encoder for processing sequential data (e.g., future motion targets).
    
    Architecture:
    1. Linear layer to project each timestep to hidden_dim
    2. Conv1D layers to process the sequence
    3. Flatten and linear layer to output_dim
    
    Based on PBHC's ConvEncoder but adapted for holosoma.
    """
    
    def __init__(self, config: ConvEncoderConfig):
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim
        self.num_timesteps = config.num_timesteps
        
        activation = getattr(nn, config.activation)()
        
        # Initial linear projection per timestep
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
        )
        
        # Build conv layers based on num_timesteps
        conv_config = self._get_conv_config(self.num_timesteps)
        
        layers = []
        in_ch = self.hidden_dim
        for i in range(len(conv_config["out_channels"])):
            layers.append(
                nn.Conv1d(
                    in_ch,
                    conv_config["out_channels"][i],
                    conv_config["kernel_sizes"][i],
                    conv_config["strides"][i],
                )
            )
            layers.append(activation)
            in_ch = conv_config["out_channels"][i]
        
        self.conv_module = nn.Sequential(*layers)
        
        # Calculate output size of conv layers
        conv_output_size = self._calculate_conv_output_size(conv_config)
        
        # Final output layer
        self.output_layer = nn.Linear(conv_output_size, self.output_dim)
    
    def _get_conv_config(self, tsteps: int) -> dict:
        """Get conv layer configuration based on number of timesteps."""
        if tsteps == 5:
            return {
                "out_channels": [20, 10],
                "kernel_sizes": [2, 2],
                "strides": [1, 1],
            }
        elif tsteps == 10:
            return {
                "out_channels": [20, 10],
                "kernel_sizes": [4, 2],
                "strides": [2, 1],
            }
        elif tsteps == 20:
            return {
                "out_channels": [40, 20],
                "kernel_sizes": [6, 4],
                "strides": [2, 2],
            }
        else:
            # Default configuration for other timesteps
            # Use a more generic configuration
            return {
                "out_channels": [32, 16],
                "kernel_sizes": [3, 3],
                "strides": [2, 2],
            }
    
    def _calculate_conv_output_size(self, conv_config: dict) -> int:
        """Calculate the output size after conv layers."""
        length = self.num_timesteps
        for i in range(len(conv_config["out_channels"])):
            kernel = conv_config["kernel_sizes"][i]
            stride = conv_config["strides"][i]
            length = (length - kernel) // stride + 1
        return length * conv_config["out_channels"][-1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, num_timesteps * input_dim]
               (flattened sequence)
        
        Returns:
            Output tensor of shape [batch_size, output_dim]
        """
        batch_size = x.shape[0]
        
        # Reshape to [batch_size * num_timesteps, input_dim]
        x = x.view(batch_size * self.num_timesteps, self.input_dim)
        
        # Apply initial linear layer
        x = self.encoder(x)  # [batch_size * num_timesteps, hidden_dim]
        
        # Reshape to [batch_size, num_timesteps, hidden_dim]
        x = x.view(batch_size, self.num_timesteps, self.hidden_dim)
        
        # Permute for Conv1D: [batch_size, hidden_dim, num_timesteps]
        x = x.permute(0, 2, 1)
        
        # Apply conv layers
        x = self.conv_module(x)  # [batch_size, out_channels, reduced_length]
        
        # Flatten
        x = x.flatten(start_dim=1)  # [batch_size, out_channels * reduced_length]
        
        # Final output layer
        return self.output_layer(x)  # [batch_size, output_dim]


class HeightMapEncoder(nn.Module):
    """Height map encoder using CNN + Cross-Attention.

    Extracts local spatial features from a height map via Conv2D, then uses
    multi-head cross-attention where proprioception is the query and height map
    features are key/value.

    Based on KraftonLab's RayEncoder architecture.

    Architecture:
        1. Conv2D extracts spatial features from height channel → (B, latent_dim-C, H, W)
        2. Concatenate with original channels → (B, latent_dim, H, W)
        3. Flatten to sequence → (B, H*W, latent_dim) = Key, Value
        4. Project proprioception → (B, 1, latent_dim) = Query
        5. MultiheadAttention(Q, K, V) → (B, latent_dim)
    """

    def __init__(
        self,
        proprio_dim: int,
        latent_dim: int = 64,
        num_heads: int = 16,
        map_height: int = 11,
        map_width: int = 17,
        num_channels: int = 3,
        conv_hidden_channels: int = 16,
    ):
        super().__init__()

        if latent_dim % num_heads != 0:
            raise ValueError(f"latent_dim {latent_dim} must be divisible by num_heads {num_heads}")
        if latent_dim <= num_channels:
            raise ValueError(f"latent_dim {latent_dim} must be > num_channels {num_channels}")

        self.latent_dim = latent_dim
        self.map_height = map_height
        self.map_width = map_width
        self.num_channels = num_channels

        # Conv2D feature extractor on height channel
        self.conv = nn.Sequential(
            nn.Conv2d(1, conv_hidden_channels, kernel_size=5, stride=1, padding=2),
            nn.ELU(),
            nn.Conv2d(conv_hidden_channels, latent_dim - num_channels, kernel_size=5, stride=1, padding=2),
            nn.ELU(),
        )

        # Proprioception → query projection
        self.linear = nn.Linear(proprio_dim, latent_dim)

        # Multi-head cross-attention
        self.mha = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.ln_q = nn.LayerNorm(latent_dim)
        self.ln_kv = nn.LayerNorm(latent_dim)

    def forward(self, height_map: torch.Tensor, proprioception: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            height_map: Flattened height map (B, H*W*C) or (B, H, W, C).
            proprioception: Proprioceptive observation (B, proprio_dim).
                This should NOT include future_motion_targets.

        Returns:
            Latent encoding (B, latent_dim).
        """
        B = height_map.shape[0]

        # Reshape to (B, H, W, C) if flattened
        if height_map.dim() == 2:
            height_map = height_map.view(B, self.map_height, self.map_width, self.num_channels)

        # (B, H, W, C) → (B, C, H, W)
        height_map = height_map.permute(0, 3, 1, 2)

        # Extract height channel and compute CNN features
        height = height_map[:, -1:, :, :]  # (B, 1, H, W) - last channel is height
        height_feature = self.conv(height)  # (B, latent_dim - C, H, W)

        # Concatenate original channels with CNN features
        local_features = torch.cat([height_map, height_feature], dim=1)  # (B, latent_dim, H, W)

        # Flatten spatial dims to sequence
        local_features_seq = local_features.flatten(start_dim=2).permute(0, 2, 1)  # (B, H*W, latent_dim)

        # Cross-attention: proprio (query) attends to height map features (key/value)
        query = self.linear(proprioception).unsqueeze(1)  # (B, 1, latent_dim)
        encoding, _ = self.mha(
            self.ln_q(query),
            self.ln_kv(local_features_seq),
            self.ln_kv(local_features_seq),
        )  # (B, 1, latent_dim)

        return encoding.squeeze(1)  # (B, latent_dim)
