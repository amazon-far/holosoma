"""The calibration fixture must measure its panel, not unrelated red geometry."""

from types import SimpleNamespace

import pytest
import torch

from tests.simulators.camera_geometry_assert import _check_geometry, _measure, _red_mask

pytestmark = pytest.mark.no_sim


def test_small_disconnected_robot_pixels_do_not_change_panel_measurement():
    image = torch.zeros(64, 64, 3, dtype=torch.uint8)
    image[20:44, 20:44, 0] = 255
    image[0:4, 56:64, 0] = 255
    assert _measure(_red_mask(image)) == (576, 31.5, 31.5, 24.0, 24.0)


def test_target_identification_still_rejects_displacement_and_wrong_fov():
    image = torch.zeros(64, 64, 3, dtype=torch.uint8)
    image[0:24, 0:24, 0] = 255
    failures = _check_geometry(image, SimpleNamespace(vertical_fov=60.0), 0.89, 0.2, "offset")
    assert any("centroid" in failure for failure in failures)
    image.zero_()
    image[10:54, 10:54, 0] = 255
    failures = _check_geometry(image, SimpleNamespace(vertical_fov=60.0), 0.89, 0.2, "fov")
    assert any("FOV" in failure for failure in failures)
