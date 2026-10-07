from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Sequence, cast

import torch
from torch import nn

from holosoma.config_types.algo import LayerConfig, ModuleConfig

if TYPE_CHECKING:
    from types import FrameType


class ImgChLayerNorm(nn.Module):
    """Image channel-wise layer normalization."""

    def __init__(self, num_channels: int, eps: float = 1e-5) -> None:
        """Initialize ImgChLayerNorm module.

        Parameters
        ----------
        num_channels: int
            Number of channels in the input tensor
        eps: float, optional
            Small value to prevent division by zero, by default 1e-5
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for image channel-wise layer normalization.

        Normalizes each channel of the input tensor independently.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape [B, C, H, W]

        Returns
        -------
        torch.Tensor
            Output tensor of shape [B, C, H, W]
        """
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class CNNWrapper(nn.Module):
    """Wrapper module that handles reshaping for CNN layers when working with flattened inputs."""

    def __init__(
        self,
        cnn_layers: nn.Module,
        input_channels: int,
        input_height: int,
        input_width: int,
        flatten_output: bool = True,
    ) -> None:
        """Initialize CNNWrapper module.

        Wraps CNN layers to handle reshaping for CNN layers when working with flattened inputs.
        For instance, this is useful when giving the flattened output of a CNN layer to an MLP layer.

        Parameters
        ----------
        cnn_layers: nn.Module
            CNN layers to wrap
        input_channels: int
            Number of input channels
        input_height: int
            Height of input feature maps
        input_width: int
            Width of input feature maps
        flatten_output: bool, optional
            Whether to flatten the output, by default True
        """
        super().__init__()
        self.cnn_layers = cnn_layers
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.expected_input_size = input_channels * input_height * input_width
        self.flatten_output = flatten_output

    @property
    def output_size(self) -> int:
        """Computes the output size of the CNN layers by doing a forward pass with dummy data."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.input_channels * self.input_height * self.input_width)
            dummy_output = self.forward(dummy_input)
            return int(dummy_output.shape[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for CNNWrapper module.

        Reshapes the input tensor to (batch_size, channels, height, width) and applies the CNN layers.
        If flatten_output is True, flattens the output back to (batch_size, -1).

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape [B, C, H, W]

        Returns
        -------
        torch.Tensor
            Output tensor of shape [B, -1] if flatten_output is True, otherwise [B, C, H, W]
        """
        # Validate input size
        batch_size = x.shape[0]
        if x.shape[1] != self.expected_input_size:
            raise ValueError(
                f"Input size mismatch: expected {self.expected_input_size} "
                f"(channels={self.input_channels}, height={self.input_height}, width={self.input_width}), "
                f"but got {x.shape[1]}"
            )

        # Reshape from flattened input to (batch_size, channels, height, width)
        x = x.view(batch_size, self.input_channels, self.input_height, self.input_width)

        # Apply CNN layers
        x = self.cnn_layers(x)

        if self.flatten_output:
            # Flatten back to (batch_size, -1)
            x = x.view(batch_size, -1)
        else:
            # x is currently [batch_size, channels, height, width]
            # Reshape to [batch_size, height * width, channels]
            x = x.view(batch_size, x.shape[1], -1)
            x = x.permute(0, 2, 1)

        return x


def build_mlp_layer(
    input_dim: int,
    hidden_dims: Sequence[int] | None,
    output_dim: int,
    layer_config: LayerConfig,
) -> nn.Sequential | None:
    """Builds a multi-layer perceptron (MLP) layer.

    Parameters
    ----------
    input_dim: int
        Number of input dimensions
    hidden_dims: tuple[int, ...]
        Tuple of hidden dimensions
    output_dim: int
        Number of output dimensions
    layer_config: dict
        Dictionary containing:
        - activation: Activation function name (e.g., "ReLU")
        - dropout_prob: Dropout probability (default: 0)

    Returns
    -------
    nn.Sequential
        The constructed MLP layer
    """
    if hidden_dims is None:
        return None

    layers = []
    activation = getattr(nn, layer_config.activation)()
    dropout = layer_config.dropout_prob

    if len(hidden_dims) == 0:
        # No hidden layer, just one linear layer
        layers.append(nn.Linear(input_dim, output_dim))
    else:
        # First hidden layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(activation)
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))

        # Additional hidden layers
        for layer_idx in range(len(hidden_dims)):
            if layer_idx == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[layer_idx], output_dim))
            else:
                layers.append(nn.Linear(hidden_dims[layer_idx], hidden_dims[layer_idx + 1]))
                layers.append(activation)
                if dropout > 0:
                    layers.append(nn.Dropout(p=dropout))

    return nn.Sequential(*layers)


def build_cnn_layer(
    input_channels: int,
    input_height: int,
    input_width: int,
    hidden_channels: tuple[int, ...] | None,
    kernel_size: int | tuple[int, ...],
    stride: int | tuple[int, ...],
    padding: str | int | tuple[str | int, ...],
    layer_config: LayerConfig,
    flatten_output: bool = True,
) -> CNNWrapper | None:
    """Builds a convolutional neural network layer that works with flattened inputs.

    Parameters
    ----------
        input_channels: int
            Number of input channels
        input_height: int
            Height of input feature maps
        input_width: int
            Width of input feature maps
        hidden_channels: tuple[int, ...]
            Tuple of channel dimensions (last value becomes output channels)
        kernel_size: int or tuple[int, ...]
            Kernel size for convolutions (int or tuple for per-layer values)
        stride: int or tuple[int, ...]
            Stride for convolutions (int or tuple for per-layer values)
        padding: str | int | tuple[str | int, ...]
            Padding mode (int, "same", "valid", or tuple for per-layer values)
        layer_config: dict
            Dictionary containing:
            - activation: Activation function name (e.g., "ReLU")
            - dropout_prob: Dropout probability (default: 0)
            - use_layer_norm: Whether to use layer normalization (default: False)

    Returns
    -------
        CNNWrapper
            The constructed CNN layer wrapped to handle flattened inputs/outputs
    """
    if hidden_channels is None:
        return None
    assert len(hidden_channels) > 0, "hidden_channels must be a non-empty tuple"

    layers: list[nn.Module] = []
    activation = getattr(nn, layer_config.encoder_activation)()
    dropout = layer_config.dropout_prob
    use_layer_norm = layer_config.use_layer_norm

    num_layers = len(hidden_channels)
    # Convert single values to tuples if needed
    if isinstance(kernel_size, int):
        kernel_sizes = (kernel_size,) * num_layers
    else:
        kernel_sizes = kernel_size
        if len(kernel_sizes) != num_layers:
            raise ValueError(
                f"kernel_size tuple length ({len(kernel_sizes)}) must match number of layers ({num_layers})"
            )

    if isinstance(stride, int):
        strides = (stride,) * num_layers
    else:
        strides = stride
        if len(strides) != num_layers:
            raise ValueError(f"stride tuple length ({len(strides)}) must match number of layers ({num_layers})")

    if isinstance(padding, (str, int)):
        paddings = (padding,) * num_layers
    else:
        paddings = padding
        if len(paddings) != num_layers:
            raise ValueError(f"padding tuple length ({len(paddings)}) must match number of layers ({num_layers})")

    # Helper function to get padding value
    def get_padding_value(padding_spec: str | int, kernel_size_val: int) -> str | int:
        if padding_spec == "same":
            return kernel_size_val // 2
        if padding_spec == "valid":
            return 0
        return padding_spec

    # Build layers
    current_in_channels = input_channels
    for layer_idx in range(num_layers):
        current_out_channels = hidden_channels[layer_idx]
        current_kernel_size = kernel_sizes[layer_idx]
        current_stride = strides[layer_idx]
        current_padding = get_padding_value(paddings[layer_idx], current_kernel_size)

        # Add convolution layer
        layers.append(
            nn.Conv2d(
                current_in_channels,
                current_out_channels,
                kernel_size=current_kernel_size,
                stride=current_stride,
                padding=current_padding,
            )
        )

        # Add layer norm, activation and dropout for all layers except the last one
        if layer_idx < num_layers - 1:
            if use_layer_norm:
                layers.append(ImgChLayerNorm(current_out_channels))
            layers.append(activation)
            if dropout > 0:
                layers.append(nn.Dropout2d(p=dropout))

        current_in_channels = current_out_channels

    cnn_sequential = nn.Sequential(*layers)

    # Wrap with CNNWrapper to handle flattened inputs/outputs
    return CNNWrapper(cnn_sequential, input_channels, input_height, input_width, flatten_output)


class BaseModule(nn.Module):
    def __init__(
        self,
        obs_dim_dict: dict[str, int],
        module_config_dict: ModuleConfig,
        history_length: dict[str, int],
    ) -> None:
        super().__init__()
        self.obs_dim_dict = obs_dim_dict
        self.module_config_dict = module_config_dict
        self.history_length = history_length
        self._calculate_input_dim()
        self._calculate_output_dim()
        self._build_network_layer(self.module_config_dict)

    def _calculate_input_dim(self) -> None:
        # calculate input dimension and input slices
        self.input_dim = 0
        self.input_dim_dict: dict[str | int | float, int] = {}
        self.input_indices_dict: dict[str | int | float, slice] = {}

        current_index = 0
        for each_input in cast("list[str | int | float]", self.module_config_dict.input_dim):
            if each_input in self.obs_dim_dict:
                # atomic observation type
                # Note: obs_dim_dict already includes history, so we don't multiply by history_length
                input_dim = self.obs_dim_dict[cast("str", each_input)]
                self.input_dim += input_dim
                self.input_dim_dict[each_input] = input_dim
                self.input_indices_dict[each_input] = slice(current_index, current_index + input_dim)
                current_index += input_dim

            elif isinstance(each_input, (int, float)):
                # direct numeric input
                input_dim = int(each_input)
                self.input_dim += input_dim
                self.input_dim_dict[each_input] = input_dim
                self.input_indices_dict[each_input] = slice(current_index, current_index + input_dim)
                current_index += input_dim

            else:
                current_function_name = cast("FrameType", inspect.currentframe()).f_code.co_name
                raise ValueError(f"{current_function_name} - Unknown input type: {each_input}")

    def _calculate_output_dim(self) -> None:
        # calculate output dimension based on the output specifications
        self.output_dim = 0
        for each_output in self.module_config_dict.output_dim:
            if isinstance(each_output, (int, float)):
                self.output_dim += each_output
            else:
                current_function_name = cast("FrameType", inspect.currentframe()).f_code.co_name
                raise ValueError(f"{current_function_name} - Unknown output type: {each_output}")

    def _build_network_layer(self, module_config: ModuleConfig) -> None:
        layer_type = module_config.type
        layer_config = module_config.layer_config
        if layer_type == "MLP":
            self.module = build_mlp_layer(
                self.input_dim,
                layer_config.hidden_dims,
                self.output_dim,
                layer_config,
            )
        elif layer_type == "CNNEncoder":
            self.encoder = build_cnn_layer(
                layer_config.input_channels,
                layer_config.input_height,
                layer_config.input_width,
                layer_config.hidden_channels,
                layer_config.kernel_size,
                layer_config.stride,
                layer_config.padding,
                layer_config,
                flatten_output=True,
            )
            if self.encoder is None:
                raise ValueError("CNNEncoder requires layer_config.hidden_channels to be set")
            encoder_output_dim = self.encoder.output_size
            mlp_input_dim = sum(self.input_dim_dict[each_input] for each_input in layer_config.module_input_name)
            self.module = build_mlp_layer(
                mlp_input_dim + encoder_output_dim,
                layer_config.hidden_dims,
                self.output_dim,
                layer_config,
            )
        elif layer_type == "MLPEncoder":
            if layer_config.encoder_hidden_dims is not None:
                # Guaranteed non-None by LayerConfig._validate_encoder_output_dim.
                assert layer_config.encoder_output_dim is not None
                encoder_output_dim = layer_config.encoder_output_dim
            else:
                encoder_output_dim = self.input_dim_dict[layer_config.encoder_input_name]
            self.encoder = build_mlp_layer(
                self.input_dim_dict[layer_config.encoder_input_name],
                layer_config.encoder_hidden_dims,
                encoder_output_dim,
                layer_config,
            )
            mlp_input_dim = sum(self.input_dim_dict[each_input] for each_input in layer_config.module_input_name)
            self.module = build_mlp_layer(
                mlp_input_dim + encoder_output_dim,
                layer_config.hidden_dims,
                self.output_dim,
                layer_config,
            )
        else:
            raise NotImplementedError(f"Unsupported layer type: {layer_type}")

    def forward(self, policy_input: torch.Tensor) -> torch.Tensor:
        # Only forward the MLP layer
        return cast("nn.Sequential", self.module)(policy_input)
