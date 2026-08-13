"""Base class for terrain terms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import trimesh

from holosoma.utils.safe_torch_import import torch

if TYPE_CHECKING:
    from holosoma.config_types.terrain import TerrainTermCfg


class TerrainTermBase(ABC):
    """Base class for stateful terrain terms."""

    def __init__(self, cfg: TerrainTermCfg, env: Any):
        """Initialize terrain term.

        Parameters
        ----------
        cfg : TerrainTermCfg
            Configuration for this terrain term
        env : Any
            Environment instance (typically BaseTask or subclass)
        """
        self._cfg = cfg
        self.env = env
        self.num_envs = env.num_envs
        self.device = env.device

    @abstractmethod
    def setup(self) -> None:
        """Setup hook called once during environment initialization.

        Override this to perform one-time setup operations like:
        - Initializing state variables
        - Storing initial environment parameters
        - Configuring adaptive schedules
        """

    @property
    @abstractmethod
    def terrain(self):
        """The terrain object."""

    @property
    @abstractmethod
    def env_origins(self) -> torch.Tensor:
        """Environment origins for multi-environment setups."""

    @property
    @abstractmethod
    def custom_origins(self) -> bool:
        """Whether the environment origins are custom."""

    @property
    @abstractmethod
    def base_heights(self) -> torch.Tensor:
        """Base heights."""

    @property
    @abstractmethod
    def feet_heights(self) -> torch.Tensor:
        """Feet heights."""

    @abstractmethod
    def update_heights(self, env_ids=None) -> None:
        """Update the base and feet heights."""

    @abstractmethod
    def draw_debug_viz(self) -> None:
        """Draw debug visualization."""

    @property
    def name(self) -> str:
        return self._cfg.name

    @property
    @abstractmethod
    def mesh(self) -> trimesh.Trimesh | None:
        """Terrain mesh."""

    @property
    def mesh_type(self) -> str:
        return self._cfg.mesh_type

    @property
    def static_friction(self) -> float:
        return self._cfg.static_friction

    @property
    def dynamic_friction(self) -> float:
        return self._cfg.dynamic_friction

    @property
    def restitution(self) -> float:
        return self._cfg.restitution

    @property
    def hide_visual(self) -> bool:
        return self._cfg.hide_visual
