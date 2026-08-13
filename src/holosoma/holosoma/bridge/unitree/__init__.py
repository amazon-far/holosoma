"""
Unitree SDK2Py bridge implementation.

``UnitreeSdk2Bridge`` talks to the ``unitree_interface`` C++/CycloneDDS binding in-process;
``UnitreeMpSdk2Bridge`` runs that binding in a spawned child so it never shares the simulator's
address space with rclpy (ROS2 sensor egress). Both defer the binding import to construction time,
so importing this package stays CycloneDDS-free.
"""

from .unitree_sdk2py_bridge import UnitreeSdk2Bridge
from .unitree_sdk2py_bridge_mp import UnitreeMpSdk2Bridge

__all__ = ["UnitreeMpSdk2Bridge", "UnitreeSdk2Bridge"]
