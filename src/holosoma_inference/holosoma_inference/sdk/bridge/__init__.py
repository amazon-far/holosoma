from .base import BasicSdk2Bridge, ElasticBand
from .booster import BoosterSdk2Bridge
from .unitree import UnitreeSdk2Bridge


def create_sdk2py_bridge(mj_model, mj_data, robot_config, lcm=None):
    """
    Factory function to create the appropriate SDK2Py bridge based on configuration.

    Args:
        mj_model: MuJoCo model
        mj_data: MuJoCo data
        robot_config: Robot configuration dictionary
        lcm: LCM instance (optional, for LCM-based bridges)

    Returns:
        An instance of the appropriate bridge class
    """
    sdk_type = robot_config.get("SDK_TYPE", "unitree")
    if sdk_type == "unitree":
        return UnitreeSdk2Bridge(mj_model, mj_data, robot_config, lcm)
    if sdk_type == "booster":
        return BoosterSdk2Bridge(mj_model, mj_data, robot_config, lcm)
    if sdk_type == "ros2":
        from .ros2 import ROS2Bridge

        return ROS2Bridge(mj_model, mj_data, robot_config, lcm)
    raise ValueError(f"Unsupported SDK type: {sdk_type}")


__all__ = [
    "BasicSdk2Bridge",
    "BoosterSdk2Bridge",
    "ElasticBand",
    "UnitreeSdk2Bridge",
    "create_sdk2py_bridge",
]
