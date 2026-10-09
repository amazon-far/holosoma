from __future__ import annotations

import sys
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any, cast

from .base import BasicSdk2Bridge

if TYPE_CHECKING:
    from holosoma.config_types.robot import RobotConfig
    from holosoma.config_types.simulator import BridgeConfig
    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator

# Auto-discover bridge implementations from installed extensions
# Handle Python 3.8/3.9 vs 3.10+ API difference for entry_points
if sys.version_info >= (3, 10):
    _bridge_registry = {ep.name: ep.load for ep in entry_points(group="holosoma.bridge")}
else:
    _eps = entry_points()
    _bridge_eps = _eps.get("holosoma.bridge", [])
    _bridge_registry = {ep.name: ep.load for ep in _bridge_eps}


def create_sdk2py_bridge(
    simulator: BaseSimulator,
    robot_config: RobotConfig,
    bridge_config: BridgeConfig,
    lcm: Any = None,
) -> BasicSdk2Bridge:
    """
    Factory function to create the appropriate SDK2Py bridge based on configuration.

    Uses entry points for SDK selection, allowing extensions to register their own
    bridge implementations without modifying the main codebase.

    Args:
        simulator: BaseSimulator instance (simulator-agnostic)
        robot_config: Robot configuration dataclass (with .bridge containing RobotBridgeConfig)
        bridge_config: Bridge configuration dataclass (simulator-level settings)
        lcm: LCM instance (optional, for LCM-based bridges)

    Returns:
        An instance of the appropriate bridge class
    """
    sdk_type = robot_config.bridge.sdk_type

    if bridge_config.dds_config is not None:
        if sdk_type != "unitree_mp":
            raise ValueError("Explicit dds_config requires the isolated unitree_mp backend")
        if bridge_config.domain_id != 0:
            raise ValueError("Explicit dds_config requires SDK domain_id=0 (not the ROS domain)")

    if sdk_type not in _bridge_registry:
        raise ValueError(f"Unsupported SDK type: {sdk_type}. Available: {list(_bridge_registry.keys())}")

    # Lazy load the bridge class
    bridge_cls = _bridge_registry[sdk_type]()
    return cast("BasicSdk2Bridge", bridge_cls(simulator, robot_config, bridge_config, lcm))


__all__ = [
    "BasicSdk2Bridge",
    "create_sdk2py_bridge",
]
