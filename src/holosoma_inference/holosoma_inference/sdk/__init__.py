"""Robot communication package."""

from __future__ import annotations

from importlib.metadata import entry_points

# Auto-discover SDK interfaces from installed packages using lazy loading.
# Lazy loading is to avoid errors from SDK dependencies from extensions (e.g. ROS2) when working with other SDKs.
_entry_points = {ep.name: ep for ep in entry_points(group="holosoma.sdk")}
_registry = {}  # Cache for loaded interfaces


def create_interface(robot_config, domain_id=0, interface_str=None, use_joystick=True):
    """Create interface from registry.

    If *interface_str* is ``"auto"``, the network interface is resolved
    automatically via :func:`holosoma_inference.utils.network.detect_robot_interface`.
    """
    # Resolve "auto" interface before passing to the SDK backend
    if interface_str == "auto":
        from holosoma_inference.utils.network import detect_robot_interface

        interface_str = detect_robot_interface()

    sdk_type = robot_config.sdk_type
    if sdk_type not in _entry_points:
        raise ValueError(f"Unknown sdk_type: {sdk_type}. Available: {sorted(_entry_points.keys())}")

    # Lazy load: only load the entry point when actually needed
    if sdk_type not in _registry:
        _registry[sdk_type] = _entry_points[sdk_type].load()

    return _registry[sdk_type](robot_config, domain_id, interface_str, use_joystick)
