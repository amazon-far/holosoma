"""
Clock synchronization module for simulation and policy synchronization.

This module provides ZMQ-based clock synchronization between the simulation
and policy inference to resolve timing issues when simulation rates vary.
"""

import time

import zmq
from loguru import logger


class ClockPub:
    """
    Clock publisher that publishes elapsed milliseconds via ZMQ.
    Used by the simulation to broadcast timing information.
    """

    def __init__(self, port=5555):
        """Initialize the clock publisher.

        Args:
            port (int): ZMQ port to publish on (default: 5555)
        """
        self.port = port
        self.context = None
        self.socket = None
        self.start_time = None
        self.enabled = False

    def start(self):
        """Start the clock publisher."""
        try:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.PUB)
            self.socket.bind(f"tcp://*:{self.port}")
            self.start_time = time.time()
            self.enabled = True
            logger.info(f"Clock publisher started on port {self.port}")
        except Exception as e:
            logger.error(f"Failed to start clock publisher: {e}")
            self.enabled = False

    def restart(self):
        """Restart the clock publisher (reset start time)."""
        if self.enabled:
            self.start_time = time.time()
            logger.info("Clock publisher restarted")

    def publish(self, sim_time):
        """Publish simulation time in milliseconds.

        Args:
            sim_time (float): Current simulation time in seconds
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

    def close(self):
        """Close the clock publisher."""
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()
        self.enabled = False


class ClockSub:
    """
    Clock subscriber that receives elapsed milliseconds via ZMQ.
    Used by the policy to get synchronized timing information.
    """

    def __init__(self, port=5555):
        """Initialize the clock subscriber.

        Args:
            port (int): ZMQ port to subscribe to (default: 5555)
        """
        self.port = port
        self.context = None
        self.socket = None
        self.last_clock = 0
        self._offset = 0

    def start(self):
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
            except zmq.Again:
                break

    def get_clock(self):
        """Get current elapsed milliseconds.

        Returns:
            int: Elapsed milliseconds since simulation start
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

    def close(self):
        """Close the clock subscriber."""
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()
