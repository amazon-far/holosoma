"""Robot communication package."""

from __future__ import annotations

from holosoma_inference.compat import entry_points

# Auto-discover SDK interfaces from installed packages using lazy loading.
# Lazy loading is to avoid errors from SDK dependencies from extensions (e.g. ROS2) when working with other SDKs.
_entry_points = {ep.name: ep for ep in entry_points(group="holosoma.sdk")}
_registry = {}  # Cache for loaded interfaces


def _load_builtin(sdk_type: str):
    """Fallback: import the stock SDK interface when entry-point metadata
    discovery turns up empty (happens in bazel runfiles where the
    egg-info-style sys.path scan doesn't always find the package).
    """
    if sdk_type == "unitree":
        from holosoma_inference.sdk.unitree.unitree_interface import UnitreeInterface

        return UnitreeInterface
    if sdk_type == "booster":
        from holosoma_inference.sdk.booster.booster_interface import BoosterInterface

        return BoosterInterface
    if sdk_type == "mujoco":
        from holosoma_inference.sdk.mujoco.mujoco_interface import MujocoInterface

        return MujocoInterface
    return None


def create_interface(robot_config, domain_id=0, interface_str=None, use_joystick=True):
    """Create interface from registry.

    If *interface_str* is ``"auto"``, the network interface is resolved
    automatically via :func:`holosoma_inference.utils.network.detect_robot_interface`.

    The env var ``HOLOSOMA_ROBOT_BACKEND`` overrides ``robot_config.sdk_type``
    without requiring callers to rebuild the RobotConfig. Example:
    ``HOLOSOMA_ROBOT_BACKEND=mujoco`` forces the MuJoCo virtual driver, even
    when the shipped config says ``sdk_type='unitree'``.
    """
    # Resolve "auto" interface before passing to the SDK backend
    if interface_str == "auto":
        from holosoma_inference.utils.network import detect_robot_interface

        interface_str = detect_robot_interface()

    import os as _os

    backend_override = _os.environ.get("HOLOSOMA_ROBOT_BACKEND")
    sdk_type = backend_override if backend_override else robot_config.sdk_type
    if sdk_type not in _entry_points and sdk_type not in _registry:
        # Fallback for environments where entry-point discovery via
        # importlib.metadata returns empty (e.g. bazel runfiles that
        # don't expose the egg-info on sys.path the way metadata expects).
        builtin = _load_builtin(sdk_type)
        if builtin is not None:
            _registry[sdk_type] = builtin
        else:
            raise ValueError(f"Unknown sdk_type: {sdk_type}. Available (entry_points): {sorted(_entry_points.keys())}")

    # Lazy load: only load the entry point when actually needed
    if sdk_type not in _registry:
        _registry[sdk_type] = _entry_points[sdk_type].load()

    return _registry[sdk_type](robot_config, domain_id, interface_str, use_joystick)
