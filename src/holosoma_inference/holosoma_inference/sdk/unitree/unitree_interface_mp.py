"""Multiprocess proxy for UnitreeInterface.

Runs the real UnitreeInterface in a spawned child process so that the
``unitree_interface`` C++ binding never shares an address space with rclpy.
The proxy implements the same BaseInterface API via RPC-over-queues.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Any, NamedTuple

import numpy as np

from holosoma_inference.config.config_types import RobotConfig
from holosoma_inference.sdk.base.base_interface import BaseInterface


class JoystickMsg(NamedTuple):
    """Picklable stand-in for the C++ wireless-controller message."""

    lx: float
    ly: float
    rx: float
    keys: int


# Sentinel that tells the worker to shut down.
_STOP = None


# ── child process ──────────────────────────────────────────────────────


def _worker(
    robot_config: RobotConfig,
    domain_id: int,
    interface_str: str | None,
    use_joystick: bool,
    req_q: mp.Queue,
    res_q: mp.Queue,
):
    """Event loop that owns the real UnitreeInterface."""
    import ctypes
    import importlib.util
    import os
    from pathlib import Path

    # Preload unitree's bundled CycloneDDS before import to prevent
    # ROS2's version from being picked up via LD_LIBRARY_PATH.
    spec = importlib.util.find_spec("unitree_interface")
    if spec and spec.submodule_search_locations:
        ui_dir = Path(spec.submodule_search_locations[0])
        for lib in ["libddsc.so.0", "libddscxx.so.0"]:
            lib_path = ui_dir / lib
            if lib_path.exists():
                ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)

    from holosoma_inference.sdk.unitree.unitree_interface import UnitreeInterface

    robot = UnitreeInterface(robot_config, domain_id, interface_str, use_joystick)

    try:
        while True:
            msg = req_q.get()
            if msg is _STOP:
                break

            method, args, kwargs = msg
            try:
                if method == "__get_kp_level":
                    res_q.put(("ok", robot.kp_level))
                elif method == "__set_kp_level":
                    robot.kp_level = args[0]
                    res_q.put(("ok", None))
                elif method == "__get_kd_level":
                    res_q.put(("ok", robot.kd_level))
                elif method == "__set_kd_level":
                    robot.kd_level = args[0]
                    res_q.put(("ok", None))
                elif method == "get_joystick_msg":
                    wc = robot.get_joystick_msg()
                    if wc is None:
                        res_q.put(("ok", None))
                    else:
                        res_q.put(
                            (
                                "ok",
                                JoystickMsg(
                                    lx=getattr(wc, "lx", 0.0),
                                    ly=getattr(wc, "ly", 0.0),
                                    rx=getattr(wc, "rx", 0.0),
                                    keys=getattr(wc, "keys", 0),
                                ),
                            )
                        )
                elif method == "get_raw_motor_state":
                    state = robot.unitree_interface.read_low_state()
                    res_q.put(
                        (
                            "ok",
                            {
                                "q": list(state.motor.q),
                                "dq": list(state.motor.dq),
                                "tau_est": list(state.motor.tau_est),
                                "voltage": list(state.motor.voltage),
                                "temperature": list(state.motor.temperature),
                                "imu_quat": list(state.imu.quat),
                                "imu_omega": list(state.imu.omega),
                                "imu_accel": list(state.imu.accel),
                            },
                        )
                    )
                elif method == "update_config":
                    robot.update_config(*args, **kwargs)
                    res_q.put(("ok", None))
                else:
                    result = getattr(robot, method)(*args, **kwargs)
                    res_q.put(("ok", result))
            except Exception as exc:
                res_q.put(("err", exc))
    finally:
        # Drop the C++ binding ref before the worker function returns so its
        # destructor runs while the DDS event loop is still alive. Then bypass
        # Python's atexit chain — any lingering C++ teardown after this point
        # tends to surface as a misleading `Process SpawnProcess-1:` stderr
        # noise even though the parent already got a clean shutdown.
        del robot
        os._exit(0)


# ── parent-side proxy ──────────────────────────────────────────────────


class UnitreeInterfaceMP(BaseInterface):
    """Drop-in replacement for UnitreeInterface that runs in a child process."""

    def __init__(self, robot_config: RobotConfig, domain_id=0, interface_str: str | None = None, use_joystick=True):
        super().__init__(robot_config, domain_id, interface_str, use_joystick)

        ctx = mp.get_context("spawn")
        self._req_q = ctx.Queue()
        self._res_q = ctx.Queue()
        self._proc = ctx.Process(
            target=_worker,
            args=(robot_config, domain_id, interface_str, use_joystick, self._req_q, self._res_q),
            daemon=True,
        )
        self._proc.start()

    # ── RPC helper ─────────────────────────────────────────────────────

    def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        self._req_q.put((method, args, kwargs))
        tag, payload = self._res_q.get()
        if tag == "err":
            raise payload
        return payload

    # ── BaseInterface implementation ───────────────────────────────────

    def get_low_state(self) -> np.ndarray:
        return self._call("get_low_state")

    def send_low_command(self, cmd_q, cmd_dq, cmd_tau, dof_pos_latest=None, kp_override=None, kd_override=None):
        return self._call("send_low_command", cmd_q, cmd_dq, cmd_tau, dof_pos_latest, kp_override, kd_override)

    def get_joystick_msg(self):
        return self._call("get_joystick_msg")

    def get_joystick_key(self, wc_msg=None):
        if wc_msg is None:
            wc_msg = self.get_joystick_msg()
        if wc_msg is None:
            return None
        return self._wc_key_map.get(getattr(wc_msg, "keys", 0), None)

    def get_raw_motor_state(self) -> dict:
        """Get raw motor/IMU state as a dict (fields not in BaseInterface)."""
        return self._call("get_raw_motor_state")

    def update_config(self, robot_config: RobotConfig):
        super().update_config(robot_config)
        self._call("update_config", robot_config)

    @property
    def kp_level(self):
        return self._call("__get_kp_level")

    @kp_level.setter
    def kp_level(self, value):
        self._call("__set_kp_level", value)

    @property
    def kd_level(self):
        return self._call("__get_kd_level")

    @kd_level.setter
    def kd_level(self, value):
        self._call("__set_kd_level", value)

    # ── lifecycle ──────────────────────────────────────────────────────

    def close(self):
        self._req_q.put(_STOP)
        self._proc.join(timeout=5)
        if self._proc.is_alive():
            self._proc.kill()

    def __del__(self):
        if hasattr(self, "_proc") and self._proc.is_alive():
            self.close()
