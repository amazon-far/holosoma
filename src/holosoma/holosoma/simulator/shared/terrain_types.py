"""Terrain configuration data classes.

This module provides typed configuration data classes for terrain loading,
separating configuration structure from business logic and defining
protocols for terrain interfaces.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import trimesh


@runtime_checkable
class TerrainInterface(Protocol):
    """Protocol defining the terrain interface that all simulators must implement."""

    def sample_env_origins(self) -> npt.NDArray[np.float64]:
        """Environment origins for multi-environment setups.

        Returns
        -------
        torch.Tensor
            Shape [num_envs, 3] containing [x, y, z] origins for each environment.
            Must be float32 tensor on the same device as the simulator.
        """
        ...

    @property
    def mesh(self) -> trimesh.Trimesh:
        """Mesh of the terrain."""
        ...
