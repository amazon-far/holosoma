"""Isaac Sim 5 and 6 adapters for their corresponding Isaac Lab interfaces.

Keep selection import-safe: the launcher uses this module before Kit or Torch
may be imported. Native simulator imports stay inside backend methods.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from holosoma.config_types.full_sim import FullSimConfig
    from holosoma.config_types.simulator import IsaacLab3Config
    from holosoma.config_types.env import EnvConfig

    from .base import IsaacLabBackendAdapter


def _installed_isaaclab_major() -> int | None:
    # Isaac Lab's extension metadata versions (e.g. 0.54.2 and 28.0.0)
    # are unrelated to the v2/v3 release tags. The separate PhysX package
    # is the defining Lab 3 API boundary and can be detected without import.
    try:
        if importlib.util.find_spec("isaaclab_physx") is not None:
            return 3
    except ValueError:
        # External-boundary tests can provide modules without import specs.
        pass
    try:
        importlib.metadata.version("isaaclab")
    except importlib.metadata.PackageNotFoundError:
        return None
    return 2


def get_backend(
    backend: str, *, native: IsaacLab3Config | None = None, rendering: bool = True
) -> IsaacLabBackendAdapter:
    """Construct the explicit API family and validate the installed runtime.

    Selection comes from the simulator config. Installed package capabilities
    validate that choice; they never choose a backend. Each owner receives its
    own adapter and native options, without process-shared mutable state.
    ``rendering`` includes cameras, movies, viewers and explicit Kit settings;
    physics-only Isaac Sim 6 runs do not load their configured renderer preference.
    """
    lab_major = _installed_isaaclab_major()
    name = "isaaclab2" if backend == "isaacsim" else backend
    if name == "isaaclab2":
        from .isaaclab2 import IsaacLab2Backend

        selected: IsaacLabBackendAdapter = IsaacLab2Backend()
    elif name == "isaaclab3":
        from .isaaclab3 import IsaacLab3Backend

        selected = IsaacLab3Backend(native, rendering=rendering)
    else:
        raise ValueError(f"Unknown Isaac Sim backend {name!r}; expected 'isaaclab2' or 'isaaclab3'.")
    if lab_major is not None and lab_major != selected.isaaclab_major:
        raise RuntimeError(
            f"The {name} backend requires Isaac Lab {selected.isaaclab_major}.x; "
            f"Isaac Lab {lab_major}.x is installed. "
            "Activate the matching setup environment or select its simulator preset."
        )
    return selected


def get_backend_for_config(config: EnvConfig | FullSimConfig) -> IsaacLabBackendAdapter:
    """Select the same runtime requirements at launch and frontend construction."""
    from holosoma.config_types.sensor import CameraSensorConfig
    from holosoma.config_types.simulator import SimulatorConfig

    simulator = config.simulator
    if isinstance(simulator, SimulatorConfig):
        simulator = simulator.config
    rendering = (
        config.logger.video.enabled
        or config.logger.headless_recording
        or any(isinstance(sensor, CameraSensorConfig) for sensor in config.sensors.values())
        or not config.training.headless
        or bool(simulator.isaacsim.app_launcher_args())
    )
    return get_backend(simulator.isaaclab_backend, native=simulator.sim.isaaclab3, rendering=rendering)
