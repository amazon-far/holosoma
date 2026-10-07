"""Clock synchronization module for simulation and policy synchronization.

This module provides ZMQ-based clock synchronization between the simulation
and policy inference to resolve timing issues when simulation rates vary.
"""

from __future__ import annotations

import time

import zmq
from loguru import logger


class ClockPub:
    """Clock publisher that publishes elapsed milliseconds via ZMQ.

    Used by the simulation to broadcast timing information to policies.
    Enables motion timestep synchronization for WBT policies.
    """

    def __init__(self, port: int = 5555) -> None:
        """Initialize the clock publisher.

        Parameters
        ----------
        port : int, default=5555
            ZMQ port to publish on.
        """
        self.port: int = port
        self.context: zmq.SyncContext | None = None
        self.socket: zmq.SyncSocket | None = None
        self.start_time: float | None = None
        self.enabled: bool = False

    def start(self) -> None:
        """Start the clock publisher."""
        if self.enabled:
            return
        context: zmq.SyncContext | None = None
        socket: zmq.SyncSocket | None = None
        try:
            context = zmq.Context()
            socket = context.socket(zmq.PUB)
            socket.bind(f"tcp://*:{self.port}")
        except Exception as e:
            logger.error(f"Failed to start clock publisher: {e}")
            if socket is not None:
                try:
                    socket.close()
                except Exception:
                    logger.exception("Clock socket rollback failed")
            if context is not None:
                try:
                    context.term()
                except Exception:
                    logger.exception("Clock context rollback failed")
            return

        self.context = context
        self.socket = socket
        self.start_time = time.time()
        self.enabled = True
        logger.info(f"Clock publisher started on port {self.port}")

    def restart(self) -> None:
        """Restart the clock publisher (reset start time)."""
        if self.enabled:
            self.start_time = time.time()
            logger.info("Clock publisher restarted")

    def publish(self, sim_time: float) -> None:
        """Publish simulation time in milliseconds.

        Parameters
        ----------
        sim_time : float
            Current simulation time in seconds.
        """
        if not self.enabled or not self.socket:
            return

        try:
            sim_time_ms = int(sim_time * 1000)
            self.socket.send_string(str(sim_time_ms), zmq.NOBLOCK)
        except zmq.Again:
            # Non-blocking send failed, skip this publish
            pass
        except Exception as e:
            logger.warning(f"Clock publish failed: {e}")

    def close(self) -> None:
        """Close the clock publisher and cleanup resources."""
        socket, context = self.socket, self.context
        self.socket = None
        self.context = None
        self.start_time = None
        self.enabled = False
        failures: list[Exception] = []
        if socket is not None:
            try:
                socket.close()
            except Exception as exc:
                failures.append(exc)
        if context is not None:
            try:
                context.term()
            except Exception as exc:
                failures.append(exc)
        if failures:
            details = "; ".join(repr(exc) for exc in failures)
            raise RuntimeError(f"Failed to close clock publisher: {details}")


class ClockSub:
    """Clock subscriber that receives elapsed milliseconds via ZMQ.

    Used by policies to get synchronized timing information from the simulator.
    Essential for WBT policies to advance motion timesteps in sync with simulation.
    """

    def __init__(self, port: int = 5555) -> None:
        """Initialize the clock subscriber.

        Parameters
        ----------
        port : int, default=5555
            ZMQ port to subscribe to.
        """
        self.port: int = port
        self.context: zmq.SyncContext | None = None
        self.socket: zmq.SyncSocket | None = None
        self.last_clock: int = 0
        self._offset: int = 0

    def start(self) -> None:
        """Start the clock subscriber."""
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://localhost:{self.port}")
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")  # Subscribe to all messages
        self.socket.setsockopt(zmq.RCVTIMEO, 10)  # 10ms timeout
        logger.info(f"Clock subscriber started, connecting to port {self.port}")

    def _drain_messages(self) -> None:
        """Drain all pending clock messages from the socket."""
        if self.socket is None:
            return

        while True:
            try:
                message = self.socket.recv_string(zmq.NOBLOCK)
                self.last_clock = int(message)
            except zmq.Again:  # noqa: PERF203
                break

    def get_clock(self) -> int:
        """Get current elapsed milliseconds.

        Returns
        -------
        int
            Elapsed milliseconds since simulation start.
        """
        # Try to receive the latest clock value (drain all pending messages)
        self._drain_messages()

        self._offset = min(self._offset, self.last_clock)

        adjusted_clock = self.last_clock - self._offset
        return max(adjusted_clock, 0)

    def reset_origin(self) -> None:
        """Reset the clock origin to the latest received timestamp."""
        self._drain_messages()
        self._offset = self.last_clock

    def close(self) -> None:
        """Close the clock subscriber and cleanup resources."""
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()
