"""``RunSimConfig.viewer_dt`` is deprecated in favour of ``viewer_update_decimation``."""

from __future__ import annotations

import dataclasses
import warnings

import pytest

from holosoma.config_types.frequency import resolve_decimation
from holosoma.config_types.run_sim import RunSimConfig

pytestmark = pytest.mark.no_sim


def test_default_is_silent() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        config = dataclasses.replace(RunSimConfig(), time_scale=0.0)

    assert config.viewer_dt is None


@pytest.mark.parametrize("fps", [200.0, 500.0])
@pytest.mark.parametrize("viewer_dt", [1 / 60.0, 0.02, 0.033333, 0.1])
def test_viewer_dt_keeps_its_decimation(viewer_dt: float, fps: float) -> None:
    with pytest.warns(FutureWarning, match="viewer_dt is deprecated"):
        config = RunSimConfig(viewer_dt=viewer_dt)

    decimation = resolve_decimation(config.viewer_update_decimation, fps, field="viewer")
    assert decimation == max(1, int(viewer_dt / (1 / fps)))
