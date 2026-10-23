"""Lab 3 quaternion convention for the shared root-state proxy."""

import torch
from holosoma.simulator.isaacsim.api import ISAACLAB3
from holosoma.simulator.isaacsim.proxy_utils import RootStatesProxy as _SharedProxy


class RootStatesProxy(_SharedProxy):
    def __init__(self, tensor_xyzw: torch.Tensor) -> None:
        super().__init__(tensor_xyzw, api=ISAACLAB3)
