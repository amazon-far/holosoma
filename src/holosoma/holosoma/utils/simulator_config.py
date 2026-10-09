"""Module for configuring the simulator type globally."""

from __future__ import annotations

from enum import Enum
from typing import cast

from typing_extensions import Self

import holosoma.config_types.simulator


class SimulatorType(Enum):
    """Enum for supported simulator types."""

    ISAACGYM = "isaacgym"
    ISAACSIM = "isaacsim"
    MUJOCO = "mujoco"

    def __str__(self) -> str:
        """Return the string value of the enum."""
        return self.value


class SimulatorConfig:
    """Singleton class to manage simulator configuration."""

    _instance: SimulatorConfig | None = None
    _simulator_type: SimulatorType | None = None
    _supported_simulator_names = {sim_type.value for sim_type in SimulatorType}

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cast("Self", cls._instance)

    @classmethod
    def set_simulator_type(cls, config: holosoma.config_types.simulator.SimulatorConfig) -> None:
        """Set the simulator type from config to use throughout the codebase."""
        simulator_type = config._target_
        simulator_type_name = simulator_type.split(".")[-1].lower()
        simulator_config_name = config.config.name
        if simulator_config_name != simulator_type_name:
            raise ValueError(
                f"Config mismatch: simulator._target_ type '{simulator_type}' inconsistent with "
                f"config name '{simulator_config_name}'"
            )
        cls._set_simulator_type_str(simulator_type_name)

    @classmethod
    def set_simulator_type_enum(cls, simulator_type: SimulatorType) -> None:
        """Set the simulator type directly using the SimulatorType enum.

        Args:
            simulator_type: The SimulatorType enum value to set
        """
        if not isinstance(simulator_type, SimulatorType):
            raise ValueError(f"Expected SimulatorType enum, got {type(simulator_type)}")
        cls._simulator_type = simulator_type

    @classmethod
    def _set_simulator_type_str(cls, simulator_type: str) -> None:
        """Set the simulator type to use throughout the codebase."""
        if simulator_type not in cls._supported_simulator_names:
            raise ValueError(f"Unsupported simulator type: {simulator_type}")

        # Convert string to enum
        for sim_type in SimulatorType:
            if sim_type.value == simulator_type:
                cls._simulator_type = sim_type
                return

        # This should never happen if _supported_simulator_names is kept in sync with SimulatorType
        raise ValueError(f"Failed to convert {simulator_type} to SimulatorType enum")

    @classmethod
    def get_simulator_type(cls) -> SimulatorType:
        """Get the currently configured simulator type."""
        if cls._simulator_type is None:
            raise RuntimeError("Simulator type not set. Call set_simulator_type() first.")
        return cls._simulator_type


# Create singleton instance
simulator_config = SimulatorConfig()


# Expose key functions at module level for backwards compatibility
def set_simulator_type(config: holosoma.config_types.simulator.SimulatorConfig) -> None:
    """Set the simulator type from config."""
    simulator_config.set_simulator_type(config)


def set_simulator_type_enum(simulator_type: SimulatorType) -> None:
    """Set the simulator type directly using the SimulatorType enum.

    Args:
        simulator_type: The SimulatorType enum value to set
    """
    simulator_config.set_simulator_type_enum(simulator_type)


def get_simulator_type() -> SimulatorType:
    """Get the currently configured simulator type."""
    return simulator_config.get_simulator_type()
