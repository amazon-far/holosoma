"""Child-process half of the multiprocess Unitree bridge; kept torch-free so spawn starts fast."""

from __future__ import annotations

import multiprocessing as mp
import re
from typing import Any, NamedTuple


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
# Pseudo-method that carries a whole transport update's operations in one request. Not a real SDK
# call, so it cannot collide with a method name the child dispatches.
_BATCH = "__batch__"


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
    message = message.removeprefix(wrapper)
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
                # A _BATCH request carries a whole transport update's operations. Running them in a
                # loop here (rather than one request each) keeps the DDS writes in the order the
                # parent queued them, while costing the parent a single blocking round trip.
                operations = list(args[0]) if method == _BATCH else [(method, args)]
                payloads: list[Any] = []
                for op_method, op_args in operations:
                    if op_method == "publish_low_state":
                        q, dq, ddq, tau_est, quat, omega, accel, tick = op_args
                        low_state.motor.q = q
                        low_state.motor.dq = dq
                        low_state.motor.ddq = ddq
                        low_state.motor.tau_est = tau_est
                        low_state.imu.quat = quat
                        low_state.imu.omega = omega
                        low_state.imu.accel = accel
                        low_state.tick = tick
                        interface.publish_low_state(low_state)  # CRC calculated in C++
                        payloads.append(None)
                    elif op_method == "publish_odom_state":
                        position, velocity, yaw_speed, quat = op_args
                        odom_state.position = position
                        odom_state.velocity = velocity
                        odom_state.yaw_speed = yaw_speed
                        odom_state.quat = quat
                        interface.publish_odom_state(odom_state)
                        payloads.append(None)
                    elif op_method == "read_incoming_command":
                        cmd = interface.read_incoming_command()
                        payloads.append(
                            LowCommand(
                                tau_ff=list(cmd.tau_ff),
                                kp=list(cmd.kp),
                                kd=list(cmd.kd),
                                q_target=list(cmd.q_target),
                                dq_target=list(cmd.dq_target),
                            )
                        )
                    elif op_method == "publish_wireless_controller":
                        lx, ly, rx, ry, keys = op_args
                        wireless_controller.lx = lx
                        wireless_controller.ly = ly
                        wireless_controller.rx = rx
                        wireless_controller.ry = ry
                        wireless_controller.keys = keys
                        interface.publish_wireless_controller(wireless_controller)
                        payloads.append(None)
                    else:
                        raise ValueError(f"Unknown method '{op_method}'")
                res_q.put(("ok", payloads if method == _BATCH else payloads[0]))
            except Exception as exc:
                res_q.put(("err", exc))
    finally:
        # Drop the binding ref before the worker returns so its destructor runs while the DDS event
        # loop is still alive, then bypass Python's atexit chain — lingering C++ teardown otherwise
        # surfaces as misleading `Process SpawnProcess-1:` stderr noise (mirrors the inference proxy).
        del interface
        os._exit(0)
