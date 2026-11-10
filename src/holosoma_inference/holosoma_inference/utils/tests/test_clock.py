"""
Unit tests for holosoma_inference.utils.clock module.
"""

from unittest import mock

import zmq

from holosoma_inference.utils.clock import ClockSub


class TestClockSub:
    """Test cases for ClockSub class."""

    def test_get_clock_with_messages(self):
        """Test get_clock with available messages."""
        mock_socket = mock.Mock()
        mock_socket.recv_string.side_effect = ["2500", zmq.Again()]

        clock_sub = ClockSub()
        clock_sub.socket = mock_socket
        clock_sub.last_clock = 1000

        result = clock_sub.get_clock()

        assert result == 2500
        assert clock_sub.last_clock == 2500

    def test_get_clock_no_messages(self):
        """Test get_clock when no messages are available."""
        mock_socket = mock.Mock()
        mock_socket.recv_string.side_effect = zmq.Again()

        clock_sub = ClockSub()
        clock_sub.socket = mock_socket
        clock_sub.last_clock = 1000

        result = clock_sub.get_clock()

        assert result == 1000

    def test_reset_origin(self):
        """Clock origin reset should zero the adjusted time."""
        mock_socket = mock.Mock()
        mock_socket.recv_string.side_effect = ["5000", zmq.Again()]

        clock_sub = ClockSub()
        clock_sub.socket = mock_socket

        clock_sub.reset_origin()
        assert clock_sub._offset == 5000

        mock_socket.recv_string.side_effect = ["6500", zmq.Again()]
        assert clock_sub.get_clock() == 1500

    def test_get_clock_handles_restart(self):
        """Clock subscriber should detect simulator restarts without explicit reset."""
        mock_socket = mock.Mock()
        clock_sub = ClockSub()
        clock_sub.socket = mock_socket

        # Simulate having previously seen a large timestamp with a matching offset
        clock_sub.last_clock = 4000
        clock_sub._offset = 4000

        # Now the simulator restarts and emits a small timestamp
        mock_socket.recv_string.side_effect = ["100", zmq.Again()]
        assert clock_sub.get_clock() == 0
        assert clock_sub._offset == 100

        # Subsequent timestamps should advance from the new origin
        mock_socket.recv_string.side_effect = ["350", zmq.Again()]
        assert clock_sub.get_clock() == 250
