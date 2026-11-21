"""Terrain terms for simulator harnesses."""

from __future__ import annotations

from typing import Any

import trimesh

from holosoma.managers.terrain.base import TerrainTermBase
from holosoma.utils.safe_torch_import import torch


class FlatMeshTerrain(TerrainTermBase):
    """A terrain term implementing only :class:`TerrainTermBase`: a flat 100 m box, top at z=0."""

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        self._mesh = trimesh.creation.box(extents=(100.0, 100.0, 0.1))
        self._mesh.apply_translation((0.0, 0.0, -0.05))
        self._env_origins = torch.zeros(self.num_envs, 3, device=self.device)

    @property
    def terrain(self) -> Any:  # Only ``terrain.mesh`` is read.
        return self

    @property
    def mesh(self) -> trimesh.Trimesh:
        return self._mesh

    @property
    def env_origins(self) -> torch.Tensor:
        return self._env_origins

    @property
    def custom_origins(self) -> bool:
        return False

    @property
    def base_heights(self) -> torch.Tensor:
        return self._env_origins[:, 2]

    @property
    def feet_heights(self) -> torch.Tensor:
        return self._env_origins[:, :0]

    def setup(self) -> None:
        pass

    def draw_debug_viz(self) -> None:
        pass
