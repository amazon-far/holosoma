"""Camera-sensor observation terms.

Read a mounted camera's rendered image via the simulator's ``get_camera_data`` accessor.

Image terms return a 4-D ``[N, H, W, C]`` tensor and belong in a ``concatenate=False`` observation
group, not the flat policy vector. An optional ``transform`` (an :class:`ImageTransformConfig`
dict) reshapes a term's frame via ``apply_image_transform``; when omitted, the frame is returned
unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from holosoma.config_types.sensor import ImageTransformConfig
from holosoma.simulator.shared.image_transform import apply_image_transform

if TYPE_CHECKING:
    import torch

    from holosoma.envs.base_task.base_task import BaseTask


def _camera_obs(env: BaseTask, sensor: str, modality: str, transform: dict | None) -> torch.Tensor:
    """Cached read of one camera modality, optionally transformed for a visual policy."""
    image = env.simulator.get_camera_data(sensor, modality)
    if transform is None:
        return image
    return apply_image_transform(image, ImageTransformConfig(**transform), modality)


def camera_rgb(env: BaseTask, sensor: str, transform: dict | None = None) -> torch.Tensor:
    """RGB frames for one mounted camera: ``[num_envs, H, W, 3]`` uint8 (R,G,B), all envs.

    Cached read of the frame rendered this step by ``render_sensors``.

    Parameters
    ----------
    env : BaseTask
        The task (provides ``env.simulator``).
    sensor : str
        Camera name (the ``CameraSensorConfig.name`` key).
    transform : dict | None
        Optional :class:`ImageTransformConfig` fields (resize, layout, scale) for a visual policy;
        ``None`` returns the unmodified frame.
    """
    return _camera_obs(env, sensor, "rgb", transform)


def camera_depth(env: BaseTask, sensor: str, transform: dict | None = None) -> torch.Tensor:
    """Depth frames for one mounted camera: ``[num_envs, H, W, 1]`` float32 meters
    (image-plane, ``+inf`` no-hit), all envs. Cached read like :func:`camera_rgb`; ``transform``
    reshapes or scales it (depth float scaling needs ``depth_range``)."""
    return _camera_obs(env, sensor, "depth", transform)
