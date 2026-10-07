"""Terrain manager coordinating terrain hooks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from holosoma.config_types.terrain import TerrainManagerCfg
from holosoma.managers.utils import resolve_callable

from .base import TerrainTermBase

if TYPE_CHECKING:
    from holosoma.utils.safe_torch_import import torch


class TerrainManager:
    """Drive terrain terms at setup, reset, and step boundaries.

    Parameters
    ----------
    cfg : TerrainManagerCfg
        Terrain manager configuration specifying terms and parameters.
    env : Any
        Environment instance operated on by terrain terms.
    device : str
        Device identifier used by terrain terms.
    """

    def __init__(self, cfg: TerrainManagerCfg, env: Any, device: str):
        self.cfg = cfg
        self.env = env
        self.device = device
        self.logger = getattr(env, "logger", None)

        self.terrain_term: TerrainTermBase = resolve_callable(self.cfg.terrain_term.func, context="terrain term")(
            self.cfg.terrain_term, self.env
        )

    def setup(self) -> None:
        """Run setup hooks.

        Calls setup() on the terrain term.
        """
        self.terrain_term.setup()

    def update_heights(self, env_ids: torch.Tensor | None = None) -> None:
        self.terrain_term.update_heights(env_ids)

    def get_state(self, term_name: str) -> TerrainTermBase:
        """Retrieve a stateful terrain term by name.

        Parameters
        ----------
        term_name : str
            Name of the terrain term.

        Returns
        -------
        TerrainTermBase or None
            Stateful terrain term instance if it exists, otherwise ``None``.
        """
        del term_name
        return self.terrain_term
