"""Unit tests for camera controller."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from holosoma.config_types.video import (
    CartesianCameraConfig,
    FixedCameraConfig,
    SphericalCameraConfig,
)
from holosoma.simulator.shared.camera_controller import CameraController, CameraParameters


@pytest.fixture
def mock_simulator() -> Mock:
    """Create mock simulator for testing."""
    simulator = Mock(spec=["robot_root_states", "find_rigid_body_indice"])
    # Mock robot position at origin
    simulator.robot_root_states = np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]])
    # Mock body resolution to succeed
    simulator.find_rigid_body_indice = Mock(return_value=1)
    return simulator


class TestCameraParameters:
    """Test CameraParameters dataclass."""

    def test_camera_parameters_creation(self):
        """Test creating camera parameters."""
        params = CameraParameters(
            position=(1.0, 2.0, 3.0),
            target=(0.0, 0.0, 0.0),
            distance=3.74,
            azimuth=45.0,
            elevation=30.0,
        )
        assert params.position == (1.0, 2.0, 3.0)
        assert params.target == (0.0, 0.0, 0.0)
        assert params.distance == 3.74
        assert params.azimuth == 45.0
        assert params.elevation == 30.0


class TestFixedCameraMode:
    """Test fixed camera mode."""

    def test_fixed_camera_returns_static_position(self, mock_simulator):
        """Test that fixed camera returns configured static position."""
        config = FixedCameraConfig(
            position=[5.0, 5.0, 3.0],
            target=[0.0, 0.0, 1.0],
        )
        controller = CameraController(config, mock_simulator)

        params = controller.update()

        assert params.position == (5.0, 5.0, 3.0)
        assert params.target == (0.0, 0.0, 1.0)
        # Distance and angles should be calculated
        assert params.distance > 0
        assert isinstance(params.azimuth, float)
        assert isinstance(params.elevation, float)

    def test_fixed_camera_no_robot_tracking(self, mock_simulator):
        """Test that fixed camera doesn't resolve robot body."""
        config = FixedCameraConfig()
        controller = CameraController(config, mock_simulator)

        # Robot body should not be resolved for fixed cameras
        assert controller.robot_body_id is None

    def test_fixed_camera_consistent_output(self, mock_simulator):
        """Test that fixed camera returns same output on multiple updates."""
        config = FixedCameraConfig(
            position=[3.0, 3.0, 2.0],
            target=[0.0, 0.0, 0.5],
        )
        controller = CameraController(config, mock_simulator)

        params1 = controller.update()
        params2 = controller.update()

        assert params1.position == params2.position
        assert params1.target == params2.target
        assert params1.distance == params2.distance


class TestSphericalCameraMode:
    """Test spherical camera mode."""

    def test_spherical_camera_calculates_position(self, mock_simulator):
        """Test spherical camera calculates correct position from coordinates."""
        config = SphericalCameraConfig(
            distance=3.0,
            azimuth=90.0,
            elevation=30.0,
            smoothing=0.0,  # Disable smoothing for deterministic test
            tracking_body_name="Trunk",
        )
        controller = CameraController(config, mock_simulator)

        params = controller.update()

        # Should have calculated position from spherical coordinates
        assert isinstance(params.position, tuple)
        assert len(params.position) == 3
        # Target should be robot position (smoothed)
        assert isinstance(params.target, tuple)
        # Distance and angles should match config
        assert params.distance == 3.0
        assert params.azimuth == 90.0
        assert params.elevation == 30.0

    def test_spherical_camera_resolves_tracking_body(self, mock_simulator):
        """Test spherical camera resolves tracking body."""
        config = SphericalCameraConfig(
            distance=3.0,
            azimuth=45.0,
            elevation=20.0,
            tracking_body_name="Trunk",
        )
        controller = CameraController(config, mock_simulator)

        # Robot body should be resolved
        controller._resolve_tracking_body()
        assert controller.robot_body_id == 1
        mock_simulator.find_rigid_body_indice.assert_called()

    def test_spherical_camera_auto_tracking(self, mock_simulator):
        """Test spherical camera with auto tracking body name."""
        config = SphericalCameraConfig(
            distance=3.0,
            azimuth=45.0,
            elevation=20.0,
            tracking_body_name="auto",
        )
        controller = CameraController(config, mock_simulator)

        # Should try default body names
        controller._resolve_tracking_body()
        assert controller.robot_body_id == 1

    def test_spherical_camera_tracking_failure(self, mock_simulator):
        """Test spherical camera fails if no tracking body found."""
        # Mock find_rigid_body_indice to always fail
        mock_simulator.find_rigid_body_indice = Mock(side_effect=ValueError("Body not found"))

        config = SphericalCameraConfig(
            distance=3.0,
            azimuth=45.0,
            elevation=20.0,
            tracking_body_name="NonExistent",
        )

        with pytest.raises(ValueError, match="No suitable tracking body found"):
            CameraController(config, mock_simulator)._resolve_tracking_body()

    def test_spherical_camera_smoothing(self, mock_simulator):
        """Test spherical camera applies smoothing."""
        config = SphericalCameraConfig(
            distance=3.0,
            azimuth=90.0,
            elevation=30.0,
            smoothing=0.9,
            tracking_body_name="Trunk",
        )
        controller = CameraController(config, mock_simulator)

        # First update initializes smoothing
        params1 = controller.update()
        target1 = params1.target

        # Move robot
        mock_simulator.robot_root_states = np.array([[2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]])

        # Second update should have smoothed target
        params2 = controller.update()
        target2 = params2.target

        # Target should have changed but be smoothed
        assert target1 != target2
        # Smoothed target should be between old and new position
        assert target2[0] > 0.0  # Moved toward new position
        assert target2[0] < 2.0  # But not all the way (due to smoothing)


class TestCartesianCameraMode:
    """Test cartesian camera mode."""

    def test_cartesian_camera_offset_from_robot(self, mock_simulator):
        """Test cartesian camera calculates position as offset from robot."""
        config = CartesianCameraConfig(
            offset=[2.0, 2.0, 1.0],
            target_offset=[0.0, 0.0, 0.5],
            smoothing=0.0,  # Disable smoothing
            tracking_body_name="Trunk",
        )
        controller = CameraController(config, mock_simulator)

        params = controller.update()

        # Robot is at (0, 0, 1), so camera should be at offset
        # Note: smoothing=0.0 means no smoothing on first frame
        assert isinstance(params.position, tuple)
        assert isinstance(params.target, tuple)

    def test_cartesian_camera_resolves_tracking_body(self, mock_simulator):
        """Test cartesian camera resolves tracking body."""
        config = CartesianCameraConfig(
            offset=[2.0, 2.0, 1.0],
            target_offset=[0.0, 0.0, 0.5],
            tracking_body_name="Trunk",
        )
        controller = CameraController(config, mock_simulator)

        # Robot body should be resolved
        controller._resolve_tracking_body()
        assert controller.robot_body_id == 1

    def test_cartesian_camera_smoothing(self, mock_simulator):
        """Test cartesian camera applies smoothing to both position and target."""
        config = CartesianCameraConfig(
            offset=[2.0, 0.0, 1.0],
            target_offset=[0.0, 0.0, 0.5],
            smoothing=0.9,
            tracking_body_name="Trunk",
        )
        controller = CameraController(config, mock_simulator)

        # First update initializes smoothing
        params1 = controller.update()
        pos1 = params1.position

        # Move robot
        mock_simulator.robot_root_states = np.array([[3.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]])

        # Second update should have smoothed position
        params2 = controller.update()
        pos2 = params2.position

        # Position should have changed but be smoothed
        assert pos1 != pos2
        # Smoothed position should be between old and new
        assert pos2[0] > pos1[0]  # Moved toward new position
        # But not jumped all the way to target (2.0 + 3.0 = 5.0)
        assert pos2[0] < 5.0


class TestCameraReset:
    """Test camera reset functionality."""

    def test_reset_clears_smoothing_state(self, mock_simulator):
        """Test that reset clears smoothing state."""
        config = CartesianCameraConfig(
            offset=[2.0, 0.0, 1.0],
            target_offset=[0.0, 0.0, 0.5],
            smoothing=0.9,
            tracking_body_name="Trunk",
        )
        controller = CameraController(config, mock_simulator)

        # Update to initialize smoothing
        controller.update()
        assert controller.smoothed_cam_pos is not None
        assert controller.smoothed_cam_target is not None

        # Reset should clear state
        controller.reset()
        assert controller.smoothed_cam_pos is None
        assert controller.smoothed_cam_target is None

    def test_reset_allows_fresh_smoothing(self, mock_simulator):
        """Test that reset allows smoothing to start fresh."""
        config = CartesianCameraConfig(
            offset=[2.0, 0.0, 1.0],
            target_offset=[0.0, 0.0, 0.5],
            smoothing=0.9,
            tracking_body_name="Trunk",
        )
        controller = CameraController(config, mock_simulator)

        # First update
        params1 = controller.update()

        # Reset
        controller.reset()

        # Second update should start fresh (not smoothed from previous)
        params2 = controller.update()

        # Positions should be very close (both starting fresh with same robot pos)
        assert params1.position == params2.position


class TestCoordinateConversion:
    """Test coordinate conversion utilities."""

    def test_cartesian_to_spherical_conversion(self):
        """Test cartesian to spherical coordinate conversion."""
        position = (3.0, 3.0, 2.0)
        target = (0.0, 0.0, 0.0)

        distance, azimuth, elevation = CameraController._cartesian_to_spherical(position, target)

        # Distance should be sqrt(3^2 + 3^2 + 2^2) = sqrt(22) ≈ 4.69
        assert pytest.approx(distance, 0.01) == 4.69
        # Azimuth for (3, 3) should be 45 degrees
        assert pytest.approx(azimuth, 0.1) == 45.0
        # Elevation should be positive (camera above target)
        assert elevation > 0

    def test_spherical_to_cartesian_conversion(self):
        """Test spherical to cartesian coordinate conversion."""
        distance = 5.0
        azimuth = 90.0  # Points in +Y direction
        elevation = 0.0  # Horizontal
        target = (0.0, 0.0, 1.0)

        position = CameraController._spherical_to_cartesian(distance, azimuth, elevation, target)

        # Should be 5 meters in +Y direction from target
        assert pytest.approx(position[0], 0.01) == 0.0
        assert pytest.approx(position[1], 0.01) == 5.0
        assert pytest.approx(position[2], 0.01) == 1.0  # Same Z as target

    def test_conversion_round_trip(self):
        """Test that converting back and forth preserves position."""
        original_pos = (3.0, 4.0, 2.0)
        target = (0.0, 0.0, 0.0)

        # Convert to spherical
        distance, azimuth, elevation = CameraController._cartesian_to_spherical(original_pos, target)

        # Convert back to cartesian
        reconstructed_pos = CameraController._spherical_to_cartesian(distance, azimuth, elevation, target)

        # Should match original (within floating point precision)
        assert pytest.approx(reconstructed_pos[0], 0.001) == original_pos[0]
        assert pytest.approx(reconstructed_pos[1], 0.001) == original_pos[1]
        assert pytest.approx(reconstructed_pos[2], 0.001) == original_pos[2]


class TestRobotPositionFetching:
    """Test robot position fetching."""

    def test_get_robot_position_from_simulator(self, mock_simulator):
        """Test getting robot position from simulator state."""
        mock_simulator.robot_root_states = np.array([[1.5, 2.5, 3.5, 0.0, 0.0, 0.0, 1.0]])

        config = SphericalCameraConfig(
            distance=3.0,
            azimuth=45.0,
            elevation=20.0,
            tracking_body_name="Trunk",
        )
        controller = CameraController(config, mock_simulator)

        robot_pos = controller._get_robot_position()

        assert robot_pos == (1.5, 2.5, 3.5)

    def test_update_with_precomputed_robot_position(self, mock_simulator):
        """Test that update can use pre-fetched robot position."""
        config = SphericalCameraConfig(
            distance=3.0,
            azimuth=90.0,
            elevation=30.0,
            smoothing=0.0,
            tracking_body_name="Trunk",
        )
        controller = CameraController(config, mock_simulator)

        # Update with explicit robot position
        robot_pos = (5.0, 0.0, 2.0)
        params = controller.update(robot_pos=robot_pos)

        # Target should use the provided robot position
        # (with smoothing=0.0, first update uses position directly)
        assert params.target[0] == pytest.approx(5.0, 0.01)
        assert params.target[2] == pytest.approx(2.0, 0.01)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_unsupported_camera_config_type(self, mock_simulator):
        """Test that unsupported camera config raises error."""
        # Create invalid config type
        invalid_config = Mock()
        controller = CameraController(invalid_config, mock_simulator)

        with pytest.raises(ValueError, match="Unsupported camera config type"):
            controller.update()

    def test_zero_distance_spherical_coordinate(self):
        """Test spherical conversion with zero distance."""
        position = (0.0, 0.0, 0.0)
        target = (0.0, 0.0, 0.0)

        distance, azimuth, elevation = CameraController._cartesian_to_spherical(position, target)

        assert distance == 0.0
        # Elevation should be 0 when distance is 0 (handled in conversion)
        assert elevation == 0.0
