"""Shared image-transform helper for camera observation terms.

Turns a ``get_camera_data`` frame (``[N, H, W, C]``, rgb uint8 [0,255] or depth float32 meters
with a ``+inf`` no-hit) into the layout/dtype a visual policy expects, applied in a fixed order:
resize, scale, layout, flatten. Each step is opt-in via :class:`ImageTransformConfig`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from holosoma.utils.safe_torch_import import torch

if TYPE_CHECKING:
    from holosoma.config_types.sensor import ImageTransformConfig


def apply_image_transform(image: torch.Tensor, config: ImageTransformConfig, modality: str) -> torch.Tensor:
    """Apply ``config`` to a ``[N, H, W, C]`` camera frame.

    Parameters
    ----------
    image : torch.Tensor
        ``[N, H, W, C]``, rgb uint8 [0,255] or depth float32 meters (``+inf`` no-hit).
    config : ImageTransformConfig
        The transform spec (resize / scale / layout / flatten / depth_range).
    modality : str
        ``"rgb"`` or ``"depth"``; selects bilinear vs nearest resize and the scaling rule.

    Returns
    -------
    torch.Tensor
        ``[N, H', W', C]`` (HWC), ``[N, C, H', W']`` (CHW), or ``[N, C*H'*W']`` (CHW + flatten),
        dtype per ``config.scale``.
    """
    if config.resize is not None:
        image = _resize(image, config.resize, modality)
    image = _scale(image, config, modality)  # before layout: _scale assumes HWC channel-last
    if config.layout == "CHW":
        image = image.permute(0, 3, 1, 2).contiguous()
    if config.flatten:
        image = image.reshape(image.shape[0], -1)
    return image


def _resize(image: torch.Tensor, size: list[int], modality: str) -> torch.Tensor:
    """Resize ``[N, H, W, C]`` to ``[N, size[0], size[1], C]`` (bilinear rgb, nearest depth).

    Depth uses nearest to keep the +inf no-hit sentinel; bilinear would blend it into finite
    distances.
    """
    # interpolate wants NCHW; round-trip the channel axis.
    nchw = image.permute(0, 3, 1, 2).float()
    mode = "nearest" if modality == "depth" else "bilinear"
    kwargs = {"align_corners": False} if mode == "bilinear" else {}
    resized = torch.nn.functional.interpolate(nchw, size=(size[0], size[1]), mode=mode, **kwargs)
    out = resized.permute(0, 2, 3, 1)
    return out.round().to(image.dtype) if modality == "rgb" else out


def _scale(image: torch.Tensor, config: ImageTransformConfig, modality: str) -> torch.Tensor:
    """Map a frame to ``config.scale`` (uint8 passthrough, float [0,1], or float [-1,1])."""
    if config.scale == "native":
        return image
    if modality == "rgb":
        unit = image.float() / 255.0  # [0,1]
    else:
        if config.depth_range is None:
            raise ValueError("ImageTransformConfig.depth_range is required to scale depth to a float range.")
        lo, hi = config.depth_range
        # Clamp to range and normalize to [0,1]; +inf no-hit clamps to the far end (1.0).
        unit = (image.float().clamp(lo, hi) - lo) / (hi - lo)
    return unit if config.scale == "float01" else unit * 2.0 - 1.0
