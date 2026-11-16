"""Camera outputs as co-simulation buffers: the layout a source requests and a renderer exports."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

from holosoma.cosim.bridge import BufferRequest, as_received

if TYPE_CHECKING:
    from holosoma.simulator.shared.sensor_manager import CameraRecord


def image_name(camera: str, data_type: str) -> str:
    return f"{camera}/{data_type}"


def camera_requests(record: CameraRecord, num_envs: int) -> list[BufferRequest]:
    """One request per output of ``record``: rgb ``uint8 [E, H, W, 3]``, depth ``float32 [E, H, W, 1]``."""
    config = record.config
    meta = as_received(
        {
            "mount": asdict(config.mount),
            "vertical_fov": config.vertical_fov,
            "near": config.near,
            "far": config.far,
            "depth_clipping_behavior": config.depth_clipping_behavior,
        }
    )
    return [
        BufferRequest(
            image_name(record.name, data_type),
            (num_envs, config.height, config.width, 3 if data_type == "rgb" else 1),
            "uint8" if data_type == "rgb" else "float32",
            meta,
        )
        for data_type in config.data_types
    ]
