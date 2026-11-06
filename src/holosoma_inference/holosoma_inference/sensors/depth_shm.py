"""Shared-memory depth sensor.

Attaches to a POSIX shared-memory block published by an external depth server
(e.g. the MuJoCo image server in the ``holosoma`` sim package, or an on-robot
camera daemon) and hands the latest frame stack to a policy.

The producer owns the block and writes a float32 array of shape
``(num_cameras, 1, height, width)`` containing depth that is already resized,
clipped and normalized — this sensor performs no preprocessing, it only reads.
Reading is a plain memory copy, so it adds no measurable latency to the control
loop.

Implements :class:`~holosoma_inference.sensors.base.Sensor`, so a policy polls
it through the same ``get_latest()`` seam as any other injected sensor.

The block itself cannot say whether the producer is still writing, so the sensor
can also follow the producer's heartbeat (see
:mod:`~holosoma_inference.sensors.depth_heartbeat`) and report how stale the
frame is through ``liveness()``.
"""

from __future__ import annotations

import math

import numpy as np
from loguru import logger

from holosoma_inference.sensors.base import Sensor
from holosoma_inference.sensors.depth_heartbeat import STATE_FAILED, DepthHeartbeatMonitor, DepthLiveness


class DepthShmSensor(Sensor):
    """Read-only view of a depth-image shared-memory block.

    Parameters
    ----------
    shape
        Expected array shape ``(num_cameras, channels, height, width)``. Must
        match the producer exactly — a mismatch means the two sides disagree
        about resolution, which would silently corrupt the depth latent.
    name
        Shared-memory block name (the producer's ``shared_memory`` name).
    required
        When ``True``, a missing block raises at ``start()``. When ``False``,
        the sensor logs a warning and serves zero frames, which keeps the
        control loop runnable without a depth producer.
    heartbeat_endpoint
        zmq endpoint of the producer's heartbeat. ``None`` disables liveness
        tracking, so ``liveness()`` returns ``None``.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        name: str = "depth_img_shm",
        required: bool = True,
        heartbeat_endpoint: str | None = None,
    ):
        self._shape = tuple(shape)
        self._name = name
        self._required = required
        self._heartbeat_endpoint = heartbeat_endpoint
        self._heartbeat: DepthHeartbeatMonitor | None = None
        self._reattach_pending = False
        self._shm = None
        self._array: np.ndarray | None = None
        self._zeros = np.zeros(self._shape, dtype=np.float32)
        self._warned_missing = False
        self._warned_reattach = False

    def start(self) -> None:
        """Attach to the shared-memory block and, if configured, the producer's heartbeat.

        Raises
        ------
        FileNotFoundError
            If the block does not exist and ``required`` is True.
        ValueError
            If the block size differs from ``shape`` — i.e. the producer and the
            policy disagree on the image dimensions.
        """
        if self._heartbeat_endpoint is not None:
            self._heartbeat = DepthHeartbeatMonitor(self._heartbeat_endpoint)
        shm = self._open_block()
        if shm is not None:
            self._adopt(shm)

    def try_attach(self) -> bool:
        """Map the block if it exists by now; returns whether the sensor is attached.

        For consumers that start before the producer: unlike :meth:`start`, a missing block is
        not reported.
        """
        if self._array is None:
            shm = self._open_block(quiet=True)
            if shm is not None:
                self._adopt(shm)
        return self.attached

    def _open_block(self, quiet: bool = False):
        from multiprocessing import shared_memory

        try:
            shm = shared_memory.SharedMemory(name=self._name)
        except FileNotFoundError:
            if self._required:
                raise FileNotFoundError(
                    f"Depth shared memory '{self._name}' not found. Start the depth producer "
                    f"before the policy, or make the sensor optional to serve zero depth frames."
                ) from None
            if not quiet:
                logger.warning(f"[DepthShmSensor] '{self._name}' not found yet; waiting for the depth producer")
            return None

        expected_bytes = int(np.prod(self._shape)) * np.dtype(np.float32).itemsize
        if shm.size != expected_bytes:
            actual = shm.size
            shm.close()
            raise ValueError(
                f"Depth shared memory '{self._name}' is {actual} bytes but shape {self._shape} "
                f"needs {expected_bytes}. The producer's camera resolution does not match this "
                f"policy's camera config."
            )
        return shm

    def _adopt(self, shm) -> None:
        self._shm = shm
        self._array = np.ndarray(self._shape, dtype=np.float32, buffer=shm.buf)
        logger.info(f"[DepthShmSensor] attached to '{self._name}' shape={self._shape}")

    @property
    def attached(self) -> bool:
        """Whether the producer's block is mapped; until then ``get_latest`` serves zeros."""
        return self._array is not None

    def get_latest(self) -> np.ndarray:
        """Return a copy of the latest depth stack, shape ``self._shape``.

        Copied because the producer may overwrite the buffer mid-read; the
        policy must operate on a stable frame.
        """
        if self._array is None:
            if not self._warned_missing:
                logger.warning(f"[DepthShmSensor] '{self._name}' unavailable — using zeros")
                self._warned_missing = True
            return self._zeros.copy()
        return self._array.copy()

    def liveness(self) -> DepthLiveness | None:
        """Poll the producer's heartbeat; ``None`` when heartbeat tracking is disabled.

        A producer that restarts recreates the block under the same name, while this
        sensor would keep reading the old, unlinked one. A restart shows up as the
        heartbeat sequence going backwards, so the sensor re-attaches then. If that
        fails, the producer is reported as failed: the old block no longer receives
        frames, so treating it as live would feed the policy a frozen view.
        """
        if self._heartbeat is None:
            return None
        status = self._heartbeat.poll()
        if status.restarted:
            logger.warning(f"[DepthShmSensor] depth producer restarted; re-attaching to '{self._name}'")
            self._reattach_pending = True
        if self._reattach_pending:
            # Polled every tick until the restarted producer recreates its block: log once.
            try:
                shm = self._open_block(quiet=True)
            except (FileNotFoundError, ValueError) as exc:
                shm = None
                if not self._warned_reattach:
                    logger.error(f"[DepthShmSensor] re-attach failed: {exc}")
            if shm is None:
                if not self._warned_reattach:
                    logger.warning(f"[DepthShmSensor] waiting for the restarted producer to recreate '{self._name}'")
                    self._warned_reattach = True
                return DepthLiveness(
                    checked=True, state=STATE_FAILED, stale_s=math.inf, seq=status.seq, restarted=status.restarted
                )
            self._detach()
            self._adopt(shm)
            self._reattach_pending = False
            self._warned_reattach = False
        return status

    def stop(self) -> None:
        """Detach from the block. Does not unlink — the producer owns it."""
        self._detach()
        if self._heartbeat is not None:
            self._heartbeat.close()
            self._heartbeat = None

    def _detach(self) -> None:
        self._array = None
        if self._shm is not None:
            self._shm.close()
            self._shm = None
