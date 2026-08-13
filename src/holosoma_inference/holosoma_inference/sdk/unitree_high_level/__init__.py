"""G1 high-level clients (arm_sdk + loco), each runnable in an isolated process.

Convenience constructors wrap each client in an :class:`MPClientProxy` so the
unitree SDK's CycloneDDS stays out of the parent's (rclpy) address space.

The constructors are typed as the *concrete* client (via ``cast``) so callers
get the real method surface — ``arm.ctrl_dual_arm(...)`` checks and completes —
even though the object is a proxy and the SDK is never imported in the parent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from .multiprocess_proxy import MPClientProxy

if TYPE_CHECKING:
    # Import-only for type checkers; never executed at runtime, so the parent
    # process still never pulls in unitree_sdk2py.
    from .arm_client.arm_client import G1j29ArmController
    from .loco_client.loco_client import G1LocoClient

_ARM = "holosoma_inference.sdk.unitree_high_level.arm_client.arm_client:G1j29ArmController"
_LOCO = "holosoma_inference.sdk.unitree_high_level.loco_client.loco_client:G1LocoClient"


def make_mp_arm_client(**kwargs: Any) -> G1j29ArmController:
    """Arm controller in its own subprocess. kwargs -> G1j29ArmController."""
    return cast("G1j29ArmController", MPClientProxy(_ARM, **kwargs))


def make_mp_loco_client(**kwargs: Any) -> G1LocoClient:
    """Loco client in its own subprocess. kwargs -> G1LocoClient."""
    return cast("G1LocoClient", MPClientProxy(_LOCO, **kwargs))


__all__ = ["MPClientProxy", "make_mp_arm_client", "make_mp_loco_client"]
