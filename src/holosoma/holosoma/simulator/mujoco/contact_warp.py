# Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""Fixed-shape GPU contact queries for named MuJoCo Warp sensors."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import warp as wp

from holosoma.utils.safe_torch_import import torch

if TYPE_CHECKING:
    from holosoma.simulator.mujoco.backends.warp_backend import WarpBackend


@wp.kernel
def _group_contact_forces(
    active_count: wp.array(dtype=wp.int32),  # type: ignore[valid-type]
    world_ids: wp.array(dtype=wp.int32),  # type: ignore[valid-type]
    geoms: wp.array(dtype=wp.vec2i),  # type: ignore[valid-type]
    frames: wp.array(dtype=wp.mat33),  # type: ignore[valid-type]
    wrenches: wp.array(dtype=wp.spatial_vector),  # type: ignore[valid-type]
    sources: wp.array2d(dtype=wp.int32),  # type: ignore[valid-type]
    counterparts: wp.array2d(dtype=wp.int32),  # type: ignore[valid-type]
    normal_only: bool,
    output: wp.array4d(dtype=wp.float32),  # type: ignore[valid-type]
) -> None:
    contact = 0  # type: int
    source = 0  # type: int
    counterpart = 0  # type: int
    contact, source, counterpart = wp.tid()  # type: ignore[misc]
    if contact >= active_count[0]:
        return
    pair = geoms[contact]
    if pair[0] < 0 or pair[1] < 0:
        return
    sign = sources[source, pair[1]] * counterparts[counterpart, pair[0]]
    sign -= sources[source, pair[0]] * counterparts[counterpart, pair[1]]
    if sign == 0:
        return
    force = wp.spatial_top(wrenches[contact])
    if normal_only:
        frame = frames[contact]
        normal = wp.vec3(frame[0, 0], frame[0, 1], frame[0, 2])
        force = normal * wp.dot(force, normal)
    for axis in range(3):
        wp.atomic_add(output, world_ids[contact], source, counterpart, axis, float(sign) * force[axis])  # type: ignore[index]


class WarpContactReader:
    """Read contact pairs already solved by the backend's captured step."""

    def __init__(
        self,
        backend: WarpBackend,
        sources: npt.NDArray[np.int32],
        counterparts: npt.NDArray[np.int32],
        force_mode: str,
    ) -> None:
        self.backend = backend
        self.normal_only = force_mode == "normal"
        with wp.ScopedDevice(backend.mjw_device):
            self.sources = wp.array(sources, dtype=wp.int32)
            self.counterparts = wp.array(counterparts, dtype=wp.int32)
            self.output = wp.zeros((backend.num_envs, len(sources), len(counterparts), 3), dtype=wp.float32)
            self.torch_stream = torch.cuda.ExternalStream(
                wp.get_stream(backend.mjw_device).cuda_stream, device=backend.device
            )
        self.tensor = wp.to_torch(self.output)

    def __call__(self) -> torch.Tensor:
        backend = self.backend
        consumer = torch.cuda.current_stream(backend.device)
        self.torch_stream.wait_stream(consumer)
        with wp.ScopedDevice(backend.mjw_device):
            self.output.zero_()
            wp.launch(
                _group_contact_forces,
                dim=(backend._contact_ids.shape[0], self.sources.shape[0], self.counterparts.shape[0]),
                inputs=[
                    backend.mjw_data.nacon,
                    backend.mjw_data.contact.worldid,
                    backend.mjw_data.contact.geom,
                    backend.mjw_data.contact.frame,
                    backend._contact_wrenches,
                    self.sources,
                    self.counterparts,
                    self.normal_only,
                ],
                outputs=[self.output],
            )
        consumer.wait_stream(self.torch_stream)
        return self.tensor
