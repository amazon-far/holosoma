"""
Unitree SDK2Py bridge implementation.

Automatically selects the appropriate backend based on available SDK:
- unitree_interface (C++ bindings) for Linux
- unitree_sdk2py (pure Python) for Mac
"""

from loguru import logger


# from .unitree_sdk2py_bridge import UnitreeSdk2Bridge
def _get_unitree_bridge_class():
    """Return the appropriate bridge class based on available SDK."""
    # Try C++ bindings first (Linux)
    try:
        from unitree_interface import UnitreeInterface  # noqa: F401

        from .unitree_sdk2py_bridge import UnitreeSdk2Bridge

        logger.debug("Using unitree_interface (C++ bindings) backend")
        return UnitreeSdk2Bridge
    except ImportError:
        pass

    # Fall back to pure Python SDK (Mac)
    try:
        from unitree_sdk2py.core.channel import ChannelPublisher  # noqa: F401

        from .unitree_sdk2py_bridge_mac import UnitreeSdk2Bridge

        logger.debug("Using unitree_sdk2py (pure Python) backend")
        return UnitreeSdk2Bridge
    except ImportError:
        pass

    raise ImportError(
        "No Unitree SDK found. Install either:\n"
        "  - unitree_sdk2 (with unitree_interface) for Linux\n"
        "  - unitree_sdk2py for Mac"
    )


UnitreeSdk2Bridge = _get_unitree_bridge_class()

__all__ = ["UnitreeSdk2Bridge"]
