"""Isaac Lab 3 state convention for the shared state adapter."""

from typing import Any

from holosoma.simulator.isaacsim.api import ISAACLAB3
from holosoma.simulator.isaacsim.state_adapter import IsaacSimStateAdapter as _SharedStateAdapter


class IsaacSimStateAdapter(_SharedStateAdapter):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, api=ISAACLAB3, **kwargs)
