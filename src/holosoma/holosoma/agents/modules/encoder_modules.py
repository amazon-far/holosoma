"""Encoder modules for processing sequential data like motion targets and history."""

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
