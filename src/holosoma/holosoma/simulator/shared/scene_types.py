"""Scene configuration data classes.

This module provides typed configuration data classes for scene loading,
separating configuration structure from business logic and defining
protocols for scene interfaces across different simulators.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from loguru import logger

from holosoma.utils.safe_torch_import import torch


@runtime_checkable
class SceneInterface(Protocol):
    """Protocol defining the scene interface that all simulators must implement.

    This protocol ensures consistent scene management across different simulator
    implementations by defining required properties and methods.
    """

    @property
    def env_origins(self) -> torch.Tensor:
        """Environment origins for multi-environment setups.

        Returns
        -------
        torch.Tensor
            Shape [num_envs, 3] containing [x, y, z] origins for each environment.
            Must be float32 tensor on the same device as the simulator.
        """
        ...


class EnvOriginsScene:
    """Minimal :class:`SceneInterface` wrapping env origins for the tensor backends.

    MuJoCo and IsaacGym have no rich scene object of their own (unlike IsaacSim's
    ``InteractiveScene``); their scene is just the per-env origins. This coerces the input to a
    float32 tensor on the sim device, validates its shape, and exposes it via ``env_origins``.
    """

    def __init__(self, env_origins: torch.Tensor, device: str) -> None:
        """Initialize the scene.

        Parameters
        ----------
        env_origins : torch.Tensor
            Environment origins with shape [num_envs, 3].
        device : str
            Device string ('cpu' or 'cuda').

        Raises
        ------
        ValueError
            If env_origins does not have shape [num_envs, 3].
        """
        if not isinstance(env_origins, torch.Tensor):
            env_origins = torch.tensor(env_origins, device=device, dtype=torch.float32)
        self._env_origins = env_origins.to(device=device, dtype=torch.float32)

        if self._env_origins.dim() != 2 or self._env_origins.shape[1] != 3:
            raise ValueError(f"env_origins must have shape [num_envs, 3], got {self._env_origins.shape}")

        logger.info(f"Scene initialized with {self._env_origins.shape[0]} environments on {device}")

    @property
    def env_origins(self) -> torch.Tensor:
        """Environment origins tensor, shape [num_envs, 3]."""
        return self._env_origins
