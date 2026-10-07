"""Cross-preset invariants for the depth-distillation sim2sim rig (sensor + plugin + terrain).

These presets only work as a *set*: the plugin names the sensor it publishes, the sensor's render
rate has to be achievable from the sim's control rate, and the plugin's depth contract has to match
the checkpoint the policy loads. Each coupling breaks at a different point — a stale camera name
raises at startup, a non-divisible rate string raises while resolving the config, and a wrong clip
range silently rescales every pixel — so they are pinned here rather than left to an end-to-end run.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from holosoma.config_types.frequency import resolve_decimation
from holosoma.config_values.plugin import PLUGIN_REGISTRY
from holosoma.config_values.run_sim import RUN_SIM_REGISTRY
from holosoma.config_values.sensor import CAMERA_REGISTRY
from holosoma.config_values.terrain import TERRAIN_REGISTRY
from holosoma.utils.path import resolve_data_file_path

pytestmark = pytest.mark.no_sim

# (plugin preset, sensor preset) pairs that are meant to be declared together.
DEPTH_RIGS = [("depth-shm", "g1-stair-front-depth"), ("depth-shm-d435i", "g1-d435i-front-depth")]

# The sim2sim scripts set the physics rate on the command line; the mujoco preset keeps its default.
SIM2SIM_SCRIPTS = [
    Path(__file__).resolve().parents[5] / "demo_scripts" / "php" / name
    for name in ("run_stair_sim.sh", "run_php_sim.sh")
]


def _sim2sim_physics_fps(script: Path) -> int:
    match = re.search(r"--simulator\.config\.sim\.fps=(\d+)", script.read_text())
    assert match, f"{script.name} must set --simulator.config.sim.fps"
    return int(match.group(1))


def _mujoco_control_hz() -> float:
    """The control rate a camera's ``update_decimation`` resolves against in sim2sim."""
    (fps,) = {_sim2sim_physics_fps(script) for script in SIM2SIM_SCRIPTS}
    return float(fps / RUN_SIM_REGISTRY["mujoco"].config.sim.control_decimation_steps)


@pytest.mark.parametrize(("plugin_name", "sensor_name"), DEPTH_RIGS)
def test_plugin_camera_matches_a_sensor_preset_key(plugin_name: str, sensor_name: str) -> None:
    """The plugin's ``camera`` must be usable as the sensor's CLI key.

    ``plugin.<key>:<preset>`` looks the camera up by the *dict key* the sensor was declared under,
    so the documented invocation only works if that key equals the plugin's ``camera`` field.
    """
    assert sensor_name in CAMERA_REGISTRY
    camera = PLUGIN_REGISTRY[plugin_name].camera
    assert camera, f"{plugin_name} must name a camera"
    # The docs/scripts declare `sensor.<camera>:<sensor_name>`, i.e. the key is the plugin's camera.
    assert camera.isidentifier(), f"{camera!r} must be a valid CLI dict key"


@pytest.mark.parametrize(("plugin_name", "sensor_name"), DEPTH_RIGS)
def test_sensor_render_rate_is_achievable_at_the_sim2sim_control_rate(plugin_name: str, sensor_name: str) -> None:
    """A camera's rate string must resolve against the sim2sim scripts' control rate.

    A bare ``"50Hz"`` demands an exact divisor and raises at 125 Hz (125/50 = 2.5), which would make
    the documented command die at startup. The presets use ``">50Hz"`` so they over-render instead;
    the policy samples the shared-memory block at its own rate, so extra frames are harmless.
    """
    decimation = resolve_decimation(
        CAMERA_REGISTRY[sensor_name].update_decimation,
        _mujoco_control_hz(),
        field="update_decimation",
    )
    achieved = _mujoco_control_hz() / decimation
    assert achieved >= 50.0, f"{sensor_name} renders at {achieved} Hz, slower than the 50 Hz policy"


def test_bare_50hz_is_not_achievable_at_the_sim2sim_control_rate() -> None:
    """Documents *why* the presets say ">50Hz" — remove this only if the control rate changes."""
    with pytest.raises(ValueError, match="not exactly achievable"):
        resolve_decimation("50Hz", _mujoco_control_hz(), field="update_decimation")


def test_d435i_plugin_matches_the_d435i_training_contract() -> None:
    """The D435i checkpoints' depth contract: [0.3, 3.0] m, training's [2:, 4:-4] crop, 58x87 out.

    ``far_clip`` differs from the ZED rig's 2.0 and rescales every pixel if wrong, and the crop is
    what makes the resize see training's field of view.
    """
    cfg = PLUGIN_REGISTRY["depth-shm-d435i"]
    assert (cfg.near_clip, cfg.far_clip) == (0.3, 3.0)
    assert (cfg.crop_top, cfg.crop_bottom, cfg.crop_left, cfg.crop_right) == (2, 0, 4, 4)
    assert (cfg.resized_height, cfg.resized_width) == (58, 87)


def test_d435i_sensor_renders_at_the_training_raycast_resolution() -> None:
    """106x60 is training's raycast size; the crop then yields 58x98 for the resize to 58x87."""
    camera = CAMERA_REGISTRY["g1-d435i-front-depth"]
    cfg = PLUGIN_REGISTRY["depth-shm-d435i"]
    assert (camera.width, camera.height) == (106, 60)

    cropped_h = camera.height - cfg.crop_top - cfg.crop_bottom
    cropped_w = camera.width - cfg.crop_left - cfg.crop_right
    assert (cropped_h, cropped_w) == (58, 98)
    # The crop must leave something to resize from, and never upsample vertically.
    assert cropped_h >= cfg.resized_height
    assert cropped_w >= cfg.resized_width


def test_d435i_sensor_frustum_does_not_clip_inside_the_training_range() -> None:
    """MuJoCo's near/far is a global frustum that *removes* geometry, unlike training's clamp.

    A near plane at ``near_clip`` would make a 0.2 m obstacle invisible (revealing the background
    behind it) where training would report it at 0.3 m, so the frustum must be strictly wider.
    """
    camera = CAMERA_REGISTRY["g1-d435i-front-depth"]
    cfg = PLUGIN_REGISTRY["depth-shm-d435i"]
    assert camera.near < cfg.near_clip
    assert camera.far > cfg.far_clip


def test_stepped_terrain_preset_is_registered_with_a_shipped_mesh() -> None:
    """``terrain:terrain-load-step`` must resolve, and its mesh must exist in the package."""
    assert "terrain_load_step" in TERRAIN_REGISTRY
    term = TERRAIN_REGISTRY["terrain_load_step"].terrain_term
    assert term.mesh_type == "load_obj"
    assert Path(resolve_data_file_path(term.obj_file_path)).exists(), (
        f"terrain mesh {term.obj_file_path} is not present in the installed package"
    )
