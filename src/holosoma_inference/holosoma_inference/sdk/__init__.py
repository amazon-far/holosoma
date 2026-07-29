"""Robot communication package."""

from __future__ import annotations

from holosoma_inference.pycompat import entry_points

from holosoma_inference.utils.network import detect_robot_interface

# Auto-discover SDK interfaces from installed packages using lazy loading.
# Lazy loading is to avoid errors from SDK dependencies from extensions (e.g. ROS2) when working with other SDKs.
_entry_points = {ep.name: ep for ep in entry_points(group="holosoma.sdk")}
_registry = {}  # Cache for loaded interfaces


def create_interface(robot_config, domain_id=0, interface_str="auto", use_joystick=True):
    """Create interface from registry.

    If *interface_str* is ``"auto"``, the network interface is resolved
    automatically via :func:`holosoma_inference.utils.network.detect_robot_interface`.
    """
    # Resolve "auto" interface before passing to the SDK backend
    if interface_str == "auto":
        interface_str = detect_robot_interface()

    sdk_type = robot_config.sdk_type
    if sdk_type not in _entry_points:
        raise ValueError(f"Unknown sdk_type: {sdk_type}. Available: {sorted(_entry_points.keys())}")

    # Lazy load: only load the entry point when actually needed
    if sdk_type not in _registry:
        _registry[sdk_type] = _entry_points[sdk_type].load()

    return _registry[sdk_type](robot_config, domain_id, interface_str, use_joystick)


def create_interface_with_default_config(robot_name, domain_id=0, interface_str="auto", use_joystick=True):
    """Create interface using a built-in robot config by name.

    Example::

        robot = create_interface_with_default_config("g1_29dof")
    """
    from holosoma_inference.config.config_values.robot import get_defaults

    defaults = get_defaults()
    key = robot_name.replace("_", "-")  # Accept both g1-29dof and g1_29dof
    if key not in defaults:
        raise ValueError(f"Unknown robot config: {robot_name!r}. Available: {sorted(defaults.keys())}")

    return create_interface(defaults[key], domain_id, interface_str, use_joystick)
