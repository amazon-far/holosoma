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

Each round trip blocks the simulator's step thread while the child is scheduled, so the publishes
are queued and sent together with the command read as a single batched request per transport update
(see ``transport_exchange``); the child still runs them one at a time, in the order they were
queued.

``compute_torques`` and every ``_get_*`` simulator read stay inherited from :class:`UnitreeSdk2Bridge`
unchanged: the parent-side ``self.low_cmd`` is a picklable :class:`LowCommand` carrying exactly the
attributes ``compute_torques`` reads (``tau_ff``/``kp``/``kd``/``q_target``/``dq_target``). The
command object remains cached between child-process polls while inherited PD feedback recomputes
against fresh simulator state on every physics step.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
from types import SimpleNamespace
from typing import Any

from loguru import logger

from holosoma.bridge.unitree.unitree_sdk2py_bridge import UnitreeSdk2Bridge
from holosoma.utils.unitree_mp_worker import _BATCH, _STOP, LowCommand, _worker

# Seconds to wait for the child to construct the binding (DDS init) before declaring it dead.
_STARTUP_TIMEOUT_S = 30.0
# Poll interval for a per-RPC liveness check — an RPC that outlives this and whose child has died is
# turned into a raised error instead of a silent, permanent block of the simulator step thread.
_RPC_POLL_S = 1.0


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

        # Calls queued by the exchange in flight; see transport_exchange.
        self._pending: list[tuple[str, tuple[Any, ...]]] = []
        self._batching = False

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

    def transport_exchange(self) -> None:
        """Run the whole exchange as ONE round trip to the SDK child.

        Each call to the child blocks the simulator's step thread while the child is scheduled, and
        the inherited sequence makes three (state, odometry, command read). Queue the publishes
        instead and send them with the command read, which the child runs in queue order -- so the
        DDS writes reach the wire in the same order, and the same exchange, as one call each.

        Only an exchange batches. A publish called on its own goes straight out, because nothing
        would flush it.
        """
        self._batching = True
        try:
            super().transport_exchange()
        finally:
            self._batching = False
            # Covers an exchange that raised before the command read, which unbatched would have
            # sent its earlier publishes already.
            self._send(self._pending)

    def _queue(self, method: str, *args: Any) -> None:
        """Add one call to the exchange in flight, or send it now if there is no exchange."""
        if self._batching:
            self._pending.append((method, args))
        else:
            self._call(method, *args)

    def _send(self, operations: list[tuple[str, tuple[Any, ...]]]) -> Any:
        """Send queued calls as one request; return the last one's reply."""
        self._pending = []
        if not operations:
            return None
        if len(operations) == 1:
            method, args = operations[0]
            return self._call(method, *args)
        return self._call(_BATCH, operations)[-1]

    # ── overridden DDS operations (everything else inherited) ──────────

    def low_cmd_handler(self, msg: Any = None) -> None:
        """Flush this update's queued publishes and take the latest incoming command back."""
        self.low_cmd = self._send([*self._pending, ("read_incoming_command", ())])

    def publish_low_state(self) -> None:
        """Compute the state fields from sim state (parent), ship them to the child to publish."""
        positions, velocities, accelerations = self._get_dof_states()
        actuator_forces = self._get_actuator_forces()
        quaternion, gyro, acceleration = self._get_base_imu_data()

        # _get_base_imu_data already returns the quaternion in SDK order [w, x, y, z]; just floatify
        # it into a plain list (same as the direct bridge does before assigning imu.quat).
        quat_array = quaternion.detach().cpu().numpy()
        quat = [float(quat_array[0]), float(quat_array[1]), float(quat_array[2]), float(quat_array[3])]

        self._queue(
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
        self._queue("publish_odom_state", position, lin_vel_body, yaw_speed, quat_wxyz)

    def publish_wireless_controller(self) -> None:
        """Populate the parent stand-in from the joystick (base class), ship it to the child."""
        # Skip UnitreeSdk2Bridge (it touches self.interface, which lives in the child) and reach
        # BasicSdk2Bridge, which reads pygame and writes lx/ly/rx/ry/keys onto self.wireless_controller.
        super(UnitreeSdk2Bridge, self).publish_wireless_controller()

        if self.joystick is not None:
            wc = self.wireless_controller
            self._queue("publish_wireless_controller", wc.lx, wc.ly, wc.rx, wc.ry, wc.keys)

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
