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
attributes ``compute_torques`` reads (``tau_ff``/``kp``/``kd``/``q_target``/``dq_target``). The
command object remains cached between child-process polls while inherited PD feedback recomputes
against fresh simulator state on every physics step.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import re
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


def _safe_startup_reason(exc: Exception) -> str:
    """Return only known diagnostic literals, never native context or user-supplied values.

    SDK 0.1.8 / bundled CycloneDDS diagnostics and common loader failures are allowlisted.
    Even quoted, escaped, truncated or unquoted XML context is discarded, not tag-stripped.
    Unknown future diagnostics deliberately lose detail; this sanitizes the startup queue
    payload only, not output the native libraries may independently write to stderr.
    """
    try:
        message = str(exc)[:4096]
    except Exception:
        return "native reason withheld (unreadable diagnostic)"
    # Released SDK wraps CycloneDDS initialization errors before pybind exposes them.
    wrapper = "Catch dds::core exception. Class:::dds::core::Error, Message:Error Error - "
    if message.startswith(wrapper):
        message = message[len(wrapper) :]
    for reason in (
        "dds_config must be nonempty inline CycloneDDS XML (no filename, URI or BOM)",
        'dds_config must be inline CycloneDDS XML with exactly one Domain Id="0" or Id="any"',
        "Could not allocate dds_config XML parser",
        "DDS initialization previously failed; start a fresh SDK process",
        "DDS already initialized with incompatible configuration; start a fresh SDK process",
        "Installed SDK does not support explicit DDS configuration API version 1",
        "Failed to create domain explicitly.",
        "Could not create DomainParticipant.",
        "Failed to create useful DomainParticipant",
        "Could not create topic.",
        "Could not create DataWriter.",
        "Could not create DataReader.",
    ):
        if message.startswith(reason):
            return reason
    # Match only known library names (including wheel hash renames). Never return the path,
    # arbitrary basenames, symbol names, or any suffix that could contain configuration data.
    for library in ("libddsc.so.0", "libddscxx.so.0", "libddsc.so", "libddscxx.so", "libstdc++.so.6", "libgcc_s.so.1"):
        pattern = re.escape(library).replace(r"\.so", r"(?:-[0-9a-f]{8})?\.so")
        match = re.search(r"(?:^|/)" + pattern + r": (.*)", message)
        if match:
            for cause in (
                "cannot open shared object file: No such file or directory",
                "cannot open shared object file: Permission denied",
                "file too short",
                "invalid ELF header",
                "wrong ELF class: ELFCLASS32",
            ):
                if match.group(1).startswith(cause):
                    return f"{library}: {cause}"
    return "native reason withheld (unrecognized diagnostic)"


# ── child process ──────────────────────────────────────────────────────


def _worker(
    interface_name: str,
    robot_type_name: str,
    message_type_name: str,
    num_motor: int,
    req_q: mp.Queue[Any],
    res_q: mp.Queue[Any],
    dds_config: str | None = None,
) -> None:
    """Event loop that owns the real ``unitree_interface`` binding (and its CycloneDDS)."""
    import ctypes
    import importlib.util
    import os
    from pathlib import Path

    # Include preload/import failures in the startup handshake, not just native construction.
    stage = "SDK library preload/import"
    try:
        # Preload unitree's bundled CycloneDDS before import so ROS2's version is not picked up via
        # LD_LIBRARY_PATH (identical to the inference-side proxy).
        spec = importlib.util.find_spec("unitree_interface")
        if spec and spec.submodule_search_locations:
            ui_dir = Path(spec.submodule_search_locations[0])
            for lib in ["libddsc.so.0", "libddscxx.so.0"]:
                lib_path = ui_dir / lib
                if lib_path.exists():
                    ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)

        import unitree_interface as sdk
        from unitree_interface import (
            LowState,
            MessageType,
            OdomState,
            RobotType,
            UnitreeInterface,
            WirelessController,
        )

        kwargs = {}
        if dds_config is not None:
            stage = "DDS capability check (requires DDS_CONFIG_API_VERSION=1)"
            if getattr(sdk, "DDS_CONFIG_API_VERSION", None) != 1:
                raise RuntimeError("Installed SDK does not support explicit DDS configuration API version 1")
            kwargs["dds_config"] = dds_config

        stage = "SDK construction"
        interface = UnitreeInterface(
            interface_name,
            getattr(RobotType, robot_type_name),
            getattr(MessageType, message_type_name),
            **kwargs,
        )
        # SIM ONLY: this bridge is always the sim fake robot. Enable the MotionSwitcher RPC
        # responder in the CHILD (which owns the DDS binding + server thread) so the operator
        # driver's startup motion-release RPC is answered.
        stage = "motion-switcher responder startup"
        interface.enable_motion_switcher_responder()
        stage = "SDK state allocation"
        # The child owns only the C++ objects the DDS calls need: a reusable LowState the parent fills
        # each publish, a WirelessController for joystick publishing, and an OdomState for base
        # odometry publishing. The incoming command lives in the parent (as a picklable LowCommand),
        # so no MotorCommand is held here.
        low_state = LowState(num_motor)
        wireless_controller = WirelessController()
        odom_state = OdomState()
    except Exception as exc:
        # Native errors may contain the entire profile or unpicklable binding-specific types.
        # On opt-in, transport only a built-in error with a bounded, recognized diagnostic.
        error = (
            RuntimeError(
                f"Unitree MP bridge: {stage} failed ({type(exc).__name__}); "
                f"{_safe_startup_reason(exc)}; explicit DDS startup aborted"
            )
            if dds_config is not None
            else exc
        )
        res_q.put(("err", error))
        # Queue.put hands the payload to a background feeder thread; flush it before exiting or
        # the message is lost and the parent times out instead of seeing the error.
        res_q.close()
        res_q.join_thread()
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

    def _init_sdk_components(self) -> None:
        """Spawn the DDS child; keep only picklable, binding-free stand-ins in the parent."""
        robot_type = self.sdk_robot_type
        if robot_type not in self.SUPPORTED_ROBOT_TYPES:
            raise ValueError(f"Invalid robot type '{robot_type}'. Unitree SDK supports: {self.SUPPORTED_ROBOT_TYPES}")

        interface_name = self.bridge_config.interface or "eth0"
        dds_config = self.bridge_config.dds_config

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
                dds_config,
            ),
            daemon=True,
        )
        # Block until construction/responder startup succeeds; clean partial child resources on
        # every startup failure because the outer simulator cannot own a failed constructor.
        try:
            self._proc.start()
            try:
                tag, payload = self._res_q.get(timeout=_STARTUP_TIMEOUT_S)
            except queue.Empty:
                raise RuntimeError("Unitree MP bridge: child process did not start within timeout") from None
            if tag == "err":
                raise payload
            if tag != "ready":
                raise RuntimeError("Unitree MP bridge: invalid child startup response")
        except BaseException:
            self.close()
            raise

        # Parent-side stand-ins (never the C++ objects): an initial zero command available before
        # the first child poll, and a mutable wireless-controller for the base joystick code.
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
        tag: str
        payload: Any
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

    def low_cmd_handler(self, msg: Any = None) -> None:
        """Poll the child for the latest incoming command; store its picklable carrier."""
        self.low_cmd = self._call("read_incoming_command")

    def publish_low_state(self) -> None:
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

    def publish_odom(self) -> None:
        """Compute base odom from sim state (parent), ship it to the child to publish.

        Mirrors publish_low_state: the inherited _get_base_odometry reads robot_root_states and
        rotates world->body velocity in the parent (binding-free), then plain float lists cross to
        the child, which owns the C++ OdomState and writes SportModeState on rt/odommodestate.
        """
        position, quat_wxyz, lin_vel_body, yaw_speed = self._get_base_odometry()
        self._call("publish_odom_state", position, lin_vel_body, yaw_speed, quat_wxyz)

    def publish_wireless_controller(self) -> None:
        """Populate the parent stand-in from the joystick (base class), ship it to the child."""
        # Skip UnitreeSdk2Bridge (it touches self.interface, which lives in the child) and reach
        # BasicSdk2Bridge, which reads pygame and writes lx/ly/rx/ry/keys onto self.wireless_controller.
        super(UnitreeSdk2Bridge, self).publish_wireless_controller()

        if self.joystick is not None:
            wc = self.wireless_controller
            self._call("publish_wireless_controller", wc.lx, wc.ly, wc.rx, wc.ry, wc.keys)

    # ── lifecycle ──────────────────────────────────────────────────────

    def close(self) -> None:
        """Stop the child process (idempotent)."""
        if not hasattr(self, "_proc") or getattr(self, "_queues_closed", False):
            return
        if self._proc.pid is not None:
            if self._proc.is_alive():
                self._req_q.put(_STOP)
            self._proc.join(timeout=5)
            if self._proc.is_alive():
                self._proc.kill()
                self._proc.join(timeout=5)
        # A failed child may not consume STOP; do not wait on its abandoned queue feeder.
        for channel in (self._req_q, self._res_q):
            channel.cancel_join_thread()
            channel.close()
        self._queues_closed = True

    def __del__(self) -> None:
        try:
            self.close()
        except Exception as exc:  # never raise from GC
            logger.debug(f"UnitreeMpSdk2Bridge teardown ignored error: {exc}")
