"""Isaac Lab camera projection adapter tests."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest

from holosoma.config_types.sensor import (
    ISAACSIM_FISHEYE_PROJECTION_TYPES,
    CameraSensorConfig,
    IsaacSimCameraConfig,
    IsaacSimCameraProjectionType,
    IsaacSimFisheyeConfig,
    SensorMountConfig,
)
from holosoma.simulator.isaacsim.camera_projection import camera_spawn_cfg_for

pytestmark = pytest.mark.no_sim


class _PinholeCameraCfg:
    projection_type: str
    focal_length: float = 24.0
    f_stop: float = 0.0
    focus_distance: float = 400.0
    clipping_range: tuple[float, float]
    vertical_aperture: float
    horizontal_aperture: float

    def __init__(self, **kwargs: Any) -> None:
        self.projection_type = "pinhole"
        self.__dict__.update(kwargs)


class _FisheyeCameraCfg(_PinholeCameraCfg):
    fisheye_nominal_width: float = 1936.0
    fisheye_nominal_height: float = 1216.0
    fisheye_optical_centre_x: float = 970.94244
    fisheye_optical_centre_y: float = 600.37482
    fisheye_max_fov: float = 200.0
    fisheye_polynomial_a: float = 0.0
    fisheye_polynomial_b: float = 0.00245
    fisheye_polynomial_c: float = 0.0
    fisheye_polynomial_d: float = 0.0
    fisheye_polynomial_e: float = 0.0
    fisheye_polynomial_f: float = 0.0

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)


_SIM_UTILS = SimpleNamespace(PinholeCameraCfg=_PinholeCameraCfg, FisheyeCameraCfg=_FisheyeCameraCfg)


def _camera(isaacsim: IsaacSimCameraConfig | None = None) -> CameraSensorConfig:
    backend_config: dict[str, Any] = {} if isaacsim is None else {"isaacsim": isaacsim}
    return CameraSensorConfig(
        mount=SensorMountConfig(target_kind="world"),
        width=320,
        height=240,
        vertical_fov=60.0,
        near=0.1,
        far=50.0,
        **backend_config,
    )


def test_pinhole_spawn_config_derives_aperture_from_shared_vertical_fov() -> None:
    config = camera_spawn_cfg_for(
        _camera(
            IsaacSimCameraConfig(
                fisheye=IsaacSimFisheyeConfig(max_fov=220.0, polynomial_b=0.25),
            )
        ),
        _SIM_UTILS,
    )

    assert isinstance(config, _PinholeCameraCfg)
    assert not isinstance(config, _FisheyeCameraCfg)
    assert config.projection_type == "pinhole"
    assert config.focal_length == 24.0
    assert config.f_stop == 0.0
    assert config.focus_distance == 400.0
    assert config.__dict__["f_stop"] == 0.0
    assert config.__dict__["focus_distance"] == 400.0
    assert config.clipping_range == (0.1, 50.0)
    assert config.vertical_aperture == pytest.approx(2.0 * 24.0 * math.tan(math.radians(60.0) / 2.0))
    assert config.horizontal_aperture == pytest.approx(config.vertical_aperture * 320.0 / 240.0)
    assert not hasattr(config, "fisheye_max_fov")
    assert not hasattr(config, "fisheye_polynomial_b")


@pytest.mark.parametrize("projection_type", ISAACSIM_FISHEYE_PROJECTION_TYPES)
def test_each_fisheye_model_selects_native_fisheye_config(
    projection_type: IsaacSimCameraProjectionType,
) -> None:
    config = camera_spawn_cfg_for(
        _camera(IsaacSimCameraConfig(projection_type=projection_type)),
        _SIM_UTILS,
    )

    assert isinstance(config, _FisheyeCameraCfg)
    assert config.projection_type == projection_type
    assert config.fisheye_nominal_width == 1936.0
    assert config.fisheye_polynomial_b == 0.00245
    assert config.__dict__["fisheye_nominal_width"] == 1936.0
    assert config.__dict__["fisheye_polynomial_b"] == 0.00245


def test_fisheye_spawn_config_passes_native_projection_and_calibration() -> None:
    config = camera_spawn_cfg_for(
        _camera(
            IsaacSimCameraConfig(
                projection_type="fisheyePolynomial",
                focal_length=18.0,
                f_stop=2.8,
                focus_distance=2.0,
                fisheye=IsaacSimFisheyeConfig(
                    nominal_width=1920.0,
                    nominal_height=1080.0,
                    optical_centre_x=950.0,
                    optical_centre_y=530.0,
                    max_fov=210.0,
                    polynomial_a=1.0,
                    polynomial_b=2.0,
                    polynomial_c=3.0,
                    polynomial_d=4.0,
                    polynomial_e=5.0,
                    polynomial_f=6.0,
                ),
            )
        ),
        _SIM_UTILS,
    )

    assert isinstance(config, _FisheyeCameraCfg)
    assert config.projection_type == "fisheyePolynomial"
    assert config.clipping_range == (0.1, 50.0)
    assert config.focal_length == 18.0
    assert config.f_stop == 2.8
    assert config.focus_distance == 2.0
    assert config.fisheye_nominal_width == 1920.0
    assert config.fisheye_nominal_height == 1080.0
    assert config.fisheye_optical_centre_x == 950.0
    assert config.fisheye_optical_centre_y == 530.0
    assert config.fisheye_max_fov == 210.0
    assert config.fisheye_polynomial_a == 1.0
    assert config.fisheye_polynomial_b == 2.0
    assert config.fisheye_polynomial_c == 3.0
    assert config.fisheye_polynomial_d == 4.0
    assert config.fisheye_polynomial_e == 5.0
    assert config.fisheye_polynomial_f == 6.0
