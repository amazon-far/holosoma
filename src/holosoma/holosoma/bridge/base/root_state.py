"""Root-state reading for SDK bridges, as a collaborator rather than base-class machinery."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator


class RootStateReader:
    """Reads the robot's root state, holding one read for the span of a transport scope.

    ``robot_root_states`` is a backend-owned proxy, not a tensor: every index rebuilds the whole
    ``[pos, quat, lin_vel, ang_vel]`` 13-vector and rotates the angular velocity into the world
    frame. A transport update can read it several times -- for the IMU, and for odometry -- always
    at the same simulator state, since nothing steps the simulator inside the update.

    Used as a context manager over that update, one read then serves every consumer in it. Outside
    a scope every :meth:`read` indexes the proxy, which is what a bridge method invoked on its own
    has always done.
    """

    def __init__(self, simulator: BaseSimulator) -> None:
        self._simulator = simulator
        self._held: torch.Tensor | None = None
        self._holding = False

    def read(self) -> torch.Tensor:
        """Return the root-state 13-vector for this robot."""
        if self._held is not None:
            return self._held
        # Env 0: one bridge drives one robot over one SDK connection, and the torque it writes back
        # goes through apply_torques_at_dof, which takes no environment index at all.
        root: torch.Tensor = self._simulator.robot_root_states[0]
        if self._holding:
            self._held = root
        return root

    def __enter__(self) -> None:
        self._holding = True

    def __exit__(self, *exc_info: object) -> None:
        self._holding = False
        self._held = None
