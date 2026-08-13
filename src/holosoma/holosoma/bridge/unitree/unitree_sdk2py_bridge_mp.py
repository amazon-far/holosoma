"""Multiprocess Unitree bridge.

Runs the CycloneDDS-touching half of :class:`UnitreeSdk2Bridge` in a spawned child process so the
``unitree_interface`` C++ binding never shares an address space with the simulator's in-process
``rclpy`` (ROS2 sensor egress). Both bundle CycloneDDS; loading them into one process heap-corrupts
at SDK init ("free(): invalid pointer"). This mirrors the inference side's
``holosoma_inference.sdk.unitree.unitree_interface_mp`` — same spawn/RPC/teardown pattern.

Unlike the inference proxy (which runs its *entire* self-contained interface in the child), this
bridge is coupled to live simulator torch/GPU state and writes PD torques back into the sim, so that
half MUST stay in the parent. Only the four DDS operations move to the child:

    * constructing ``UnitreeInterface`` (opens CycloneDDS),
    * ``publish_low_state``       (parent computes the fields from sim state, ships plain lists),
    * ``publish_odom_state``      (parent computes base odom from sim state, ships plain lists),
    * ``read_incoming_command``   (child polls DDS, ships a picklable command back),
    * ``publish_wireless_controller`` (parent reads the joystick, ships the axes/keys).

``compute_torques`` and every ``_get_*`` simulator read stay inherited from :class:`UnitreeSdk2Bridge`
unchanged: the parent-side ``self.low_cmd`` is a picklable :class:`LowCommand` carrying exactly the
attributes ``compute_torques`` reads (``tau_ff``/``kp``/``kd``/``q_target``/``dq_target``).
"""

from __future__ import annotations

import multiprocessing as mp
import queue
from types import SimpleNamespace
from typing import Any, NamedTuple

from loguru import logger

from holosoma.bridge.unitree.unitree_sdk2py_bridge import UnitreeSdk2Bridge

# Seconds to wait for the child to construct the binding (DDS init) before declaring it dead.
_STARTUP_TIMEOUT_S = 30.0
# Poll interval for a per-RPC liveness check — an RPC that outlives this and whose child has died is
# turned into a raised error instead of a silent, permanent block of the simulator step thread.
_RPC_POLL_S = 1.0


class LowCommand(NamedTuple):
    """Picklable stand-in for the C++ low-level command returned by ``read_incoming_command``.

    Carries exactly the attributes :meth:`UnitreeSdk2Bridge.compute_torques` reads, so the inherited
    torque computation runs against it in the parent unchanged.
    """

    tau_ff: list[float]
    kp: list[float]
    kd: list[float]
    q_target: list[float]
    dq_target: list[float]


# Sentinel that tells the worker to shut down.
_STOP = None


# ── child process ──────────────────────────────────────────────────────


def _worker(
    interface_name: str,
    robot_type_name: str,
    message_type_name: str,
    num_motor: int,
    req_q: mp.Queue,
    res_q: mp.Queue,
):
    """Event loop that owns the real ``unitree_interface`` binding (and its CycloneDDS)."""
    import ctypes
    import importlib.util
    import os
    from pathlib import Path

    # Preload unitree's bundled CycloneDDS before import so ROS2's version is not picked up via
    # LD_LIBRARY_PATH (identical to the inference-side proxy).
    spec = importlib.util.find_spec("unitree_interface")
    if spec and spec.submodule_search_locations:
        ui_dir = Path(spec.submodule_search_locations[0])
        for lib in ["libddsc.so.0", "libddscxx.so.0"]:
            lib_path = ui_dir / lib
            if lib_path.exists():
                ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)

    # Construct the binding (opens CycloneDDS) BEFORE signalling readiness, so a DDS init failure is
    # reported to the parent as an error instead of leaving it to block on the first RPC forever.
    try:
        from unitree_interface import (
            LowState,
            MessageType,
            OdomState,
            RobotType,
            UnitreeInterface,
            WirelessController,
        )

        interface = UnitreeInterface(
            interface_name,
            getattr(RobotType, robot_type_name),
            getattr(MessageType, message_type_name),
        )
        # The child owns only the C++ objects the DDS calls need: a reusable LowState the parent fills
        # each publish, a WirelessController for joystick publishing, and an OdomState for base
        # odometry publishing. The incoming command lives in the parent (as a picklable LowCommand),
        # so no MotorCommand is held here.
        low_state = LowState(num_motor)
        wireless_controller = WirelessController()
        odom_state = OdomState()
    except Exception as exc:
        res_q.put(("err", exc))
        os._exit(0)
    res_q.put(("ready", None))

    try:
        while True:
            msg = req_q.get()
            if msg is _STOP:
                break

            method, args, kwargs = msg
            try:
                if method == "publish_low_state":
                    q, dq, ddq, tau_est, quat, omega, accel, tick = args
                    low_state.motor.q = q
                    low_state.motor.dq = dq
                    low_state.motor.ddq = ddq
                    low_state.motor.tau_est = tau_est
                    low_state.imu.quat = quat
                    low_state.imu.omega = omega
                    low_state.imu.accel = accel
                    low_state.tick = tick
                    interface.publish_low_state(low_state)  # CRC calculated in C++
                    res_q.put(("ok", None))
                elif method == "publish_odom_state":
                    position, velocity, yaw_speed, quat = args
                    odom_state.position = position
                    odom_state.velocity = velocity
                    odom_state.yaw_speed = yaw_speed
                    odom_state.quat = quat
                    interface.publish_odom_state(odom_state)
                    res_q.put(("ok", None))
                elif method == "read_incoming_command":
                    cmd = interface.read_incoming_command()
                    res_q.put(
                        (
                            "ok",
                            LowCommand(
                                tau_ff=list(cmd.tau_ff),
                                kp=list(cmd.kp),
                                kd=list(cmd.kd),
                                q_target=list(cmd.q_target),
                                dq_target=list(cmd.dq_target),
                            ),
                        )
                    )
                elif method == "publish_wireless_controller":
                    lx, ly, rx, ry, keys = args
                    wireless_controller.lx = lx
                    wireless_controller.ly = ly
                    wireless_controller.rx = rx
                    wireless_controller.ry = ry
                    wireless_controller.keys = keys
                    interface.publish_wireless_controller(wireless_controller)
                    res_q.put(("ok", None))
                else:
                    res_q.put(("err", ValueError(f"Unknown method '{method}'")))
            except Exception as exc:
                res_q.put(("err", exc))
    finally:
        # Drop the binding ref before the worker returns so its destructor runs while the DDS event
        # loop is still alive, then bypass Python's atexit chain — lingering C++ teardown otherwise
        # surfaces as misleading `Process SpawnProcess-1:` stderr noise (mirrors the inference proxy).
        del interface
        os._exit(0)


# ── parent-side bridge ───────────────────────────────────────────────────


class UnitreeMpSdk2Bridge(UnitreeSdk2Bridge):
    """Unitree bridge that runs the ``unitree_interface`` binding in a spawned child process.

    Drop-in for :class:`UnitreeSdk2Bridge` (same ``holosoma.bridge`` factory + ``compute_torques``);
    isolates CycloneDDS from the rest of the process. Select it via ``sdk_type="unitree_mp"`` when the
    simulator process also loads rclpy (a ROS2 sensor egress) — the two CycloneDDS runtimes cannot
    coexist in one process.
    """

    def _init_sdk_components(self):
        """Spawn the DDS child; keep only picklable, binding-free stand-ins in the parent."""
        robot_type = self.robot.asset.robot_type
        if robot_type not in self.SUPPORTED_ROBOT_TYPES:
            raise ValueError(f"Invalid robot type '{robot_type}'. Unitree SDK supports: {self.SUPPORTED_ROBOT_TYPES}")

        interface_name = self.bridge_config.interface or "eth0"

        ctx = mp.get_context("spawn")
        self._req_q = ctx.Queue()
        self._res_q = ctx.Queue()
        self._proc = ctx.Process(
            target=_worker,
            args=(
                interface_name,
                self._ROBOT_TYPE_NAMES[robot_type],
                self._MESSAGE_TYPE_NAMES[robot_type],
                self.num_motor,
                self._req_q,
                self._res_q,
            ),
            daemon=True,
        )
        self._proc.start()

        # Block until the child has constructed the binding (or failed), so a DDS init error raises
        # here — like the in-process bridge's synchronous construction — instead of hanging a later
        # step. (Mirrors the high-level MPClientProxy's readiness handshake.)
        try:
            tag, payload = self._res_q.get(timeout=_STARTUP_TIMEOUT_S)
        except queue.Empty:
            self.close()
            raise RuntimeError("Unitree MP bridge: child process did not start within timeout") from None
        if tag == "err":
            self.close()
            raise payload

        # Parent-side stand-ins (never the C++ objects): a zero command truthy for compute_torques'
        # guard, and a mutable wireless-controller the base joystick code writes its axes/keys onto.
        zeros = [0.0] * self.num_motor
        self.low_cmd = LowCommand(tau_ff=zeros, kp=zeros, kd=zeros, q_target=zeros, dq_target=zeros)
        self.wireless_controller = SimpleNamespace(lx=0.0, ly=0.0, rx=0.0, ry=0.0, keys=0)

    # ── RPC helper ─────────────────────────────────────────────────────

    def _call(self, method: str, *args: Any) -> Any:
        """Send one request and block for its reply. Raises if the child has died (never hangs).

        Without the liveness check a child that crashed at the C level (segfault, or the very
        heap-corruption this module isolates) — which never sends a reply — would block the simulator
        step thread forever. Poll instead, and convert a dead child into a raised error.
        """
        self._req_q.put((method, args, {}))
        while True:
            try:
                tag, payload = self._res_q.get(timeout=_RPC_POLL_S)
                break
            except queue.Empty:
                if not self._proc.is_alive():
                    raise RuntimeError(f"Unitree MP bridge: child died during '{method}'") from None
        if tag == "err":
            raise payload
        return payload

    # ── overridden DDS operations (everything else inherited) ──────────

    def low_cmd_handler(self, msg=None):
        """Poll the child for the latest incoming command; store its picklable carrier."""
        self.low_cmd = self._call("read_incoming_command")

    def publish_low_state(self):
        """Compute the state fields from sim state (parent), ship them to the child to publish."""
        positions, velocities, accelerations = self._get_dof_states()
        actuator_forces = self._get_actuator_forces()
        quaternion, gyro, acceleration = self._get_base_imu_data()

        # _get_base_imu_data already returns the quaternion in SDK order [w, x, y, z]; just floatify
        # it into a plain list (same as the direct bridge does before assigning imu.quat).
        quat_array = quaternion.detach().cpu().numpy()
        quat = [float(quat_array[0]), float(quat_array[1]), float(quat_array[2]), float(quat_array[3])]

        self._call(
            "publish_low_state",
            positions.tolist(),
            velocities.tolist(),
            accelerations.tolist(),
            actuator_forces.tolist(),
            quat,
            gyro.detach().cpu().numpy().tolist(),
            acceleration.detach().cpu().numpy().tolist(),
            int(self.sim_time * 1e3),
        )

    def publish_odom(self):
        """Compute base odom from sim state (parent), ship it to the child to publish.

        Mirrors publish_low_state: the inherited _get_base_odometry reads robot_root_states and
        rotates world->body velocity in the parent (binding-free), then plain float lists cross to
        the child, which owns the C++ OdomState and writes SportModeState on rt/odommodestate.
        """
        position, quat_wxyz, lin_vel_body, yaw_speed = self._get_base_odometry()
        self._call("publish_odom_state", position, lin_vel_body, yaw_speed, quat_wxyz)

    def publish_wireless_controller(self):
        """Populate the parent stand-in from the joystick (base class), ship it to the child."""
        # Skip UnitreeSdk2Bridge (it touches self.interface, which lives in the child) and reach
        # BasicSdk2Bridge, which reads pygame and writes lx/ly/rx/ry/keys onto self.wireless_controller.
        super(UnitreeSdk2Bridge, self).publish_wireless_controller()

        if self.joystick is not None:
            wc = self.wireless_controller
            self._call("publish_wireless_controller", wc.lx, wc.ly, wc.rx, wc.ry, wc.keys)

    # ── lifecycle ──────────────────────────────────────────────────────

    def close(self):
        """Stop the child process (idempotent)."""
        if not hasattr(self, "_proc"):
            return
        if self._proc.is_alive():
            self._req_q.put(_STOP)
            self._proc.join(timeout=5)
            if self._proc.is_alive():
                self._proc.kill()

    def __del__(self):
        try:
            self.close()
        except Exception as exc:  # never raise from GC
            logger.debug(f"UnitreeMpSdk2Bridge teardown ignored error: {exc}")
