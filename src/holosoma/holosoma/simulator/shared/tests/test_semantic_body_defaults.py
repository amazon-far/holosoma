"""Semantic body-tag defaults shared by cameras and the virtual gantry."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from holosoma.config_types.simulator import VirtualGantryCfg
from holosoma.config_types.video import SphericalCameraConfig
from holosoma.simulator.base_simulator.base_simulator import BaseSimulator
from holosoma.simulator.shared.camera_controller import CameraController
from holosoma.simulator.shared.virtual_gantry import create_virtual_gantry

pytestmark = pytest.mark.no_sim


class _Sim(BaseSimulator):
    def __init__(self) -> None:
        self.robot_config = SimpleNamespace(
            body_names=["pelvis", "torso_link"],
            body_tags={"#base": ["torso_link"]},
            dof_names=[],
            asset=SimpleNamespace(fix_base_link=False),
        )  # type: ignore[assignment]
        self.body_ids = [4, 9]
        self.num_envs = 1

    def find_rigid_body_indice(self, body_name: str) -> int:
        return self.body_names.index(body_name)


def test_semantic_defaults_use_base_tag() -> None:
    assert VirtualGantryCfg().attachment_body_names == ["#base"]
    camera = CameraController(SphericalCameraConfig(), _Sim())
    camera._resolve_tracking_body()
    assert camera.robot_body_id == 1

    gantry = create_virtual_gantry(_Sim(), enable=True)
    assert gantry.body_link_id == 1


def test_explicit_camera_body_falls_back_to_base_tag() -> None:
    camera = CameraController(SphericalCameraConfig(tracking_body_name="missing"), _Sim())
    camera._resolve_tracking_body()
    assert camera.robot_body_id == 1
