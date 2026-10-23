"""Small API differences shared by the Isaac Sim integrations.

Shared builders default to Isaac Sim 5.1. Newer backends pass their API profile
explicitly, so importing shared code never selects or launches a simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from importlib import import_module
from typing import Any

import torch

from .state_utils import fullstate_wxyz_to_xyzw, fullstate_xyzw_to_wxyz
from .quat_convention import wxyz_to_xyzw


@dataclass(frozen=True)
class IsaacLabApi:
    quaternions_xyzw: bool = False
    env_regex: str = "/World/envs/env_.*"
    stage_module: str = "isaacsim.core.utils.stage"
    prim_module: str = "isaacsim.core.utils.prims"
    importer_preserves_fixed_joint_markers: bool = True

    def config_quaternion(self, wxyz: Sequence[float]) -> tuple[float, float, float, float]:
        x, y, z, w = wxyz_to_xyzw(wxyz)
        return (x, y, z, w) if self.quaternions_xyzw else (w, x, y, z)

    def read_state(self, state: torch.Tensor) -> torch.Tensor:
        return state if self.quaternions_xyzw else fullstate_wxyz_to_xyzw(state)

    def write_state(self, state: torch.Tensor) -> torch.Tensor:
        return state if self.quaternions_xyzw else fullstate_xyzw_to_wxyz(state)

    def stage_utils(self) -> Any:
        return import_module(self.stage_module)

    def prim_utils(self) -> Any:
        return import_module(self.prim_module)


ISAACSIM5 = IsaacLabApi()
ISAACLAB3 = IsaacLabApi(
    quaternions_xyzw=True,
    env_regex="/World/envs/env_[^/]+",
    stage_module="isaaclab.sim.utils.stage",
    prim_module="isaaclab.sim.utils",
    importer_preserves_fixed_joint_markers=False,
)
