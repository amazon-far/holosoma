"""Minimal G1 locomotion client — direct ``LocoClient`` wrapper.

A direct-DDS locomotion client with no TCP-bridge plumbing. It assumes it
runs ON THE JETSON, where the unitree SDK's CycloneDDS version works (a
bridge is only needed to escape a laptop/Docker DDS version mismatch — not
the case here).

Intended use: a service layer receives ROS2 commands and calls these
methods directly. The class owns DDS init + the ``LocoClient`` session;
callers just push velocity / FSM transitions.

Standalone (smoke test on Jetson):

    PYTHONPATH=~/unitree_sdk2 python3 loco_client.py
"""

from __future__ import annotations

import logging

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as hg_LowState

# Conventional-walk FSM. 501 decouples arms from legs so rt/arm_sdk works
# alongside locomotion; 200/801 (AMP gait) couples them and blocks arm_sdk.
FSM_ID_WALK = 501

# Network interface the G1 MCU is reachable on. unitree_sdk2py's
# ChannelFactoryInitialize takes the interface name positionally.
DEFAULT_IFACE = "eth0"


class G1LocoClient:
    """Thin wrapper around Unitree's ``LocoClient`` for base velocity control."""

    def __init__(self, iface: str = DEFAULT_IFACE, timeout: float = 5.0, logger=None):
        self._logger = logger or logging.getLogger(__name__)

        self._logger.info(f"Initializing DDS on {iface}...")
        ChannelFactoryInitialize(0, iface)

        # Wait for the MCU to come up on rt/lowstate before opening the
        # LocoClient RPC session (mirrors the bridge's startup gate).
        self._lowstate_sub = ChannelSubscriber("rt/lowstate", hg_LowState)
        self._lowstate_sub.Init()
        self._logger.info("Waiting for MCU DDS connection (rt/lowstate)...")
        waited = 0.0
        connect_timeout = 10.0
        while True:
            msg = self._lowstate_sub.Read(timeout=0.1)
            if msg is not None:
                self._logger.info(f"MCU connected! mode_machine={msg.mode_machine}")
                break
            waited += 0.1
            if waited >= connect_timeout:
                raise TimeoutError(
                    f"No rt/lowstate in {connect_timeout:.0f}s — MCU not reachable on the "
                    "loco DDS interface (eth0). Check the robot is up and the interface is right."
                )
            if round(waited, 1) % 2 == 0:
                self._logger.info(f"... still waiting for rt/lowstate ({waited:.0f}s)")

        self._loco = LocoClient()
        self._loco.SetTimeout(timeout)
        self._loco.Init()
        self._logger.info("LocoClient ready")

    def start(self) -> None:
        """Bring the robot to balance-stand and start the controller."""
        code = self._loco.BalanceStand(1)
        self._logger.info(f"BalanceStand(1) -> code={code}")
        code = self._loco.Start()
        self._logger.info(f"Start() -> code={code}")

    def set_walk_mode(self, fsm_id: int = FSM_ID_WALK) -> None:
        """Switch to conventional-walk FSM so arm_sdk works alongside loco."""
        fsm_before, fsm_data_before = self._loco._Call(7001, "{}")
        self._logger.info(f"FSM before SetFsmId({fsm_id}): code={fsm_before} data={fsm_data_before}")
        code = self._loco.SetFsmId(fsm_id)
        self._logger.info(f"SetFsmId({fsm_id}) -> code={code}")

    def set_velocity(self, vx: float, vy: float, vyaw: float) -> None:
        """Command base velocity (body frame). Continuous: holds until next call."""
        code = self._loco.Move(vx, vy, vyaw, continous_move=True)
        if abs(vx) > 0.01 or abs(vy) > 0.01 or abs(vyaw) > 0.01:
            self._logger.info(f"Move({vx:.3f},{vy:.3f},{vyaw:.3f},continuous) -> code={code}")

    def stop(self) -> None:
        """Stop base motion (zero velocity)."""
        self._loco.StopMove()

    def get_fsm_state(self):
        """Return raw (code, data) for the current FSM state (api 7001)."""
        return self._loco._Call(7001, "{}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    client = G1LocoClient()
    client.start()
    client.set_walk_mode()
    print("LocoClient up — FSM state:", client.get_fsm_state())
    client.stop()
