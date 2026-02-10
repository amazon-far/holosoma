"""
Unit tests for holosoma_inference.policies.wbt_utils module.

Tests focus on timing logic for MotionClockUtil and TimestepUtil classes.
"""

from unittest import mock

from holosoma_inference.policies.wbt_utils import MotionClockUtil, TimestepUtil


class TestMotionClockUtil:
    """Test cases for MotionClockUtil class."""

    def test_elapsed_ms_first_call_returns_zero(self):
        """First call to elapsed_ms should return 0 (anchor is set to current time)."""
        mock_clock_sub = mock.Mock()
        mock_clock_sub.get_clock.return_value = 100

        clock_util = MotionClockUtil(mock_clock_sub)
        result = clock_util.elapsed_ms()

        assert result == 0  # 100 - 100 = 0

    def test_elapsed_ms_normal_progression(self):
        """Normal clock progression should return correct elapsed time."""
        mock_clock_sub = mock.Mock()
        mock_clock_sub.get_clock.side_effect = [100, 150, 200]

        clock_util = MotionClockUtil(mock_clock_sub)

        assert clock_util.elapsed_ms() == 0  # Anchor at 100: 100-100=0
        assert clock_util.elapsed_ms() == 50  # 150-100=50
        assert clock_util.elapsed_ms() == 100  # 200-100=100

    def test_elapsed_ms_handles_backward_clock_jump(self):
        """Backward clock jump (e.g., sim reset) should preserve elapsed progress."""
        mock_clock_sub = mock.Mock()
        # Clock sequence: 100 -> 200 -> 50 (jump back)
        mock_clock_sub.get_clock.side_effect = [100, 200, 50]

        clock_util = MotionClockUtil(mock_clock_sub)

        assert clock_util.elapsed_ms() == 0  # Anchor at 100: 100-100=0
        assert clock_util.elapsed_ms() == 100  # Normal: 200-100=100
        # After jump: elapsed_at_anchor becomes 100, new anchor=50
        # Result: 100 + (50-50) = 100
        assert clock_util.elapsed_ms() == 100

    def test_elapsed_ms_backward_jump_continues_correctly(self):
        """After backward jump, subsequent calls should continue from new anchor."""
        mock_clock_sub = mock.Mock()
        # Clock: 100 -> 200 -> 50 (jump) -> 80 -> 120
        mock_clock_sub.get_clock.side_effect = [100, 200, 50, 80, 120]

        clock_util = MotionClockUtil(mock_clock_sub)

        assert clock_util.elapsed_ms() == 0  # Anchor at 100
        assert clock_util.elapsed_ms() == 100  # 200-100
        assert clock_util.elapsed_ms() == 100  # Jump: 100 + (50-50)
        assert clock_util.elapsed_ms() == 130  # 100 + (80-50)
        assert clock_util.elapsed_ms() == 170  # 100 + (120-50)

    def test_elapsed_ms_logs_warning_on_backward_jump(self):
        """Backward clock jump should log a warning when logger is provided."""
        mock_clock_sub = mock.Mock()
        mock_clock_sub.get_clock.side_effect = [100, 200, 50]
        mock_logger = mock.Mock()

        clock_util = MotionClockUtil(mock_clock_sub)

        clock_util.elapsed_ms(log=mock_logger)  # Anchor
        clock_util.elapsed_ms(log=mock_logger)  # Normal
        clock_util.elapsed_ms(log=mock_logger)  # Jump back

        mock_logger.warning.assert_called_once_with("Clock jumped back; re-anchoring.")

    def test_reset_clears_state_and_calls_clock_sub_reset(self):
        """Reset should clear internal state and call clock_sub.reset_origin()."""
        mock_clock_sub = mock.Mock()
        mock_clock_sub.get_clock.side_effect = [100, 200, 0, 50]

        clock_util = MotionClockUtil(mock_clock_sub)

        # Build up some state
        assert clock_util.elapsed_ms() == 0
        assert clock_util.elapsed_ms() == 100

        # Reset
        clock_util.reset()

        # Verify reset_origin was called
        mock_clock_sub.reset_origin.assert_called_once()

        # After reset, should start fresh
        assert clock_util.elapsed_ms() == 0  # New anchor at 0
        assert clock_util.elapsed_ms() == 50  # 50-0=50

    def test_reset_clears_elapsed_at_anchor(self):
        """Reset should clear the accumulated elapsed_ms_at_anchor."""
        mock_clock_sub = mock.Mock()
        # Sequence: 100 -> 200 -> 50 (jump, accumulates 100) -> reset -> 0 -> 30
        mock_clock_sub.get_clock.side_effect = [100, 200, 50, 0, 30]

        clock_util = MotionClockUtil(mock_clock_sub)

        clock_util.elapsed_ms()  # Anchor at 100
        clock_util.elapsed_ms()  # 100ms elapsed
        clock_util.elapsed_ms()  # Jump back, elapsed_at_anchor=100

        clock_util.reset()

        # After reset, elapsed_at_anchor should be 0
        assert clock_util.elapsed_ms() == 0  # Fresh anchor at 0
        assert clock_util.elapsed_ms() == 30  # 30-0=30, not 130


class TestTimestepUtil:
    """Test cases for TimestepUtil class."""

    def test_get_timestep_normal_progression(self):
        """Normal timestep progression based on elapsed time."""
        mock_clock = mock.Mock()
        mock_clock.elapsed_ms.side_effect = [0, 20, 40, 60]

        timestep_util = TimestepUtil(mock_clock, interval_ms=20.0, start_timestep=0)

        assert timestep_util.get_timestep() == 0  # 0/20 = 0
        assert timestep_util.get_timestep() == 1  # 20/20 = 1
        assert timestep_util.get_timestep() == 2  # 40/20 = 2
        assert timestep_util.get_timestep() == 3  # 60/20 = 3

    def test_get_timestep_with_start_timestep(self):
        """Timesteps should be offset by start_timestep."""
        mock_clock = mock.Mock()
        mock_clock.elapsed_ms.side_effect = [0, 20, 40]

        timestep_util = TimestepUtil(mock_clock, interval_ms=20.0, start_timestep=10)

        assert timestep_util.get_timestep() == 10  # 0/20 + 10 = 10
        assert timestep_util.get_timestep() == 11  # 20/20 + 10 = 11
        assert timestep_util.get_timestep() == 12  # 40/20 + 10 = 12

    def test_get_timestep_handles_forward_jump_at_start(self):
        """Forward jump at start should trigger reset to prevent frame skipping."""
        mock_clock = mock.Mock()
        # Simulate realistic clock behavior:
        # - First call: 500ms elapsed (jumped ahead, > 1 step, triggers reset)
        # - After reset() re-anchors, subsequent elapsed values are relative to new anchor
        # - Values 10, 30 represent time since the re-anchor point
        mock_clock.elapsed_ms.side_effect = [500, 10, 30]

        timestep_util = TimestepUtil(mock_clock, interval_ms=20.0, start_timestep=0)

        result = timestep_util.get_timestep()
        assert result == 0  # Should return start_timestep, not jump to step 25
        mock_clock.reset.assert_called_once()

        # After reset re-anchors the clock, elapsed times are small again
        assert timestep_util.get_timestep() == 0  # 10/20 = 0 steps
        assert timestep_util.get_timestep() == 1  # 30/20 = 1 step

    def test_get_timestep_forward_jump_only_at_start(self):
        """Forward jump detection only applies when at start_timestep."""
        mock_clock = mock.Mock()
        # Start normal, then jump (should NOT reset since we're past start)
        mock_clock.elapsed_ms.side_effect = [0, 20, 500]

        timestep_util = TimestepUtil(mock_clock, interval_ms=20.0, start_timestep=0)

        assert timestep_util.get_timestep() == 0  # At start
        assert timestep_util.get_timestep() == 1  # Past start
        assert timestep_util.get_timestep() == 25  # Jump allowed (500/20=25)

        mock_clock.reset.assert_not_called()

    def test_get_timestep_logs_warning_on_forward_jump(self):
        """Forward jump at start should log a warning when logger is provided."""
        mock_clock = mock.Mock()
        mock_clock.elapsed_ms.return_value = 500
        mock_logger = mock.Mock()

        timestep_util = TimestepUtil(mock_clock, interval_ms=20.0, start_timestep=0)
        timestep_util.get_timestep(log=mock_logger)

        mock_logger.warning.assert_called_once_with("Clock jumped ahead at start; re-anchoring.")

    def test_reset_without_start_timestep(self):
        """Reset without parameter should use original start_timestep."""
        mock_clock = mock.Mock()
        mock_clock.elapsed_ms.side_effect = [0, 20, 0, 20]

        timestep_util = TimestepUtil(mock_clock, interval_ms=20.0, start_timestep=5)

        assert timestep_util.get_timestep() == 5  # 0/20 + 5 = 5
        assert timestep_util.get_timestep() == 6  # 20/20 + 5 = 6

        timestep_util.reset()  # No parameter
        mock_clock.reset.assert_called_once()

        assert timestep_util.get_timestep() == 5  # Back to original start_timestep
        assert timestep_util.get_timestep() == 6

    def test_reset_with_new_start_timestep(self):
        """Reset with parameter should update start_timestep."""
        mock_clock = mock.Mock()
        mock_clock.elapsed_ms.side_effect = [0, 20, 0, 20]

        timestep_util = TimestepUtil(mock_clock, interval_ms=20.0, start_timestep=5)

        assert timestep_util.get_timestep() == 5  # 0/20 + 5 = 5
        assert timestep_util.get_timestep() == 6  # 20/20 + 5 = 6

        timestep_util.reset(start_timestep=100)  # New start
        mock_clock.reset.assert_called_once()

        assert timestep_util.get_timestep() == 100  # 0/20 + 100 = 100
        assert timestep_util.get_timestep() == 101  # 20/20 + 100 = 101

    def test_timestep_property_returns_cached_value(self):
        """The timestep property should return cached value without clock update."""
        mock_clock = mock.Mock()
        # Use 20ms which is exactly 1 step - won't trigger forward jump detection
        mock_clock.elapsed_ms.return_value = 20

        timestep_util = TimestepUtil(mock_clock, interval_ms=20.0, start_timestep=0)

        # Before any get_timestep call, should return start_timestep
        assert timestep_util.timestep == 0

        # After get_timestep, should return updated value
        timestep_util.get_timestep()
        assert timestep_util.timestep == 1

        # Property access should not call elapsed_ms again
        call_count = mock_clock.elapsed_ms.call_count
        _ = timestep_util.timestep
        assert mock_clock.elapsed_ms.call_count == call_count

    def test_timestep_calculation_with_fractional_interval(self):
        """Timestep calculation should handle fractional intervals correctly."""
        mock_clock = mock.Mock()
        mock_clock.elapsed_ms.side_effect = [0, 33, 66, 100]

        timestep_util = TimestepUtil(mock_clock, interval_ms=33.33, start_timestep=0)

        assert timestep_util.get_timestep() == 0  # 0/33.33 = 0
        assert timestep_util.get_timestep() == 0  # 33/33.33 = 0 (floor)
        assert timestep_util.get_timestep() == 1  # 66/33.33 = 1 (floor)
        assert timestep_util.get_timestep() == 3  # 100/33.33 = 3 (floor)
