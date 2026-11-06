"""Configuration compatibility, independent of an installed simulator."""

from dataclasses import replace

import pytest
import tyro
from pydantic import TypeAdapter

from holosoma.config_types.isaac import IsaacSimRenderConfig
from holosoma.config_types.run_sim import RunSimConfig

pytestmark = pytest.mark.no_sim


def test_existing_saved_launcher_settings_keep_51_defaults():
    cfg = TypeAdapter(IsaacSimRenderConfig).validate_python({"rendering_mode": "quality", "kit_args": "--/app/test=1"})
    assert cfg.runtime == "5.1"
    assert cfg.physics == "isaacsim_physx"
    assert cfg.app_launcher_args() == {"rendering_mode": "quality", "kit_args": "--/app/test=1"}


@pytest.mark.parametrize(
    ("preset", "runtime", "physics"),
    [
        ("isaacsim", "5.1", "isaacsim_physx"),
        ("isaacsim6", "6.x", "isaacsim_physx"),
        ("isaacsim6-newton", "6.x", "newton"),
        ("isaacsim6-ovphysx", "6.x", "ovphysx"),
    ],
)
def test_presets_share_frontend_and_roundtrip(preset, runtime, physics):
    cfg = tyro.cli(RunSimConfig, args=[f"simulator:{preset}"])
    assert cfg.simulator._target_ == "holosoma.simulator.isaacsim.isaacsim.IsaacSim"
    selection = cfg.simulator.config.isaacsim
    assert (selection.runtime, selection.physics) == (runtime, physics)
    adapter = TypeAdapter(IsaacSimRenderConfig)
    assert adapter.validate_json(adapter.dump_json(selection)) == selection


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"physics": "newton"}, "5.1 supports Isaac Sim PhysX only"),
        ({"renderer": "ovrtx"}, "5.1 supports the Isaac RTX renderer only"),
        ({"physics_options": {"num_substeps": 2}}, "Native backend options"),
        ({"runtime": "6.x", "solver": "xpbd"}, "apply only to Newton"),
        ({"runtime": "6.x", "visualizer_options": {"viser": {"port": 8123}}}, "explicit visualizer selection"),
        ({"runtime": "6.x", "renderer": "none", "renderer_options": {"foo": 1}}, "requires a selected renderer"),
    ],
)
def test_incompatible_configuration_is_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        IsaacSimRenderConfig(**kwargs)


def test_headless_never_selects_a_visualizer():
    cfg = IsaacSimRenderConfig(runtime="6.x", physics="newton")
    assert cfg.selected_visualizers(headless=True) == []
    assert cfg.selected_visualizers(headless=False) == ["newton_gl"]
    with pytest.raises(ValueError, match="Headless"):
        replace(cfg, visualizers=["newton_gl"]).selected_visualizers(headless=True)


def test_cli_native_options_survive_saved_configuration():
    cfg = tyro.cli(
        RunSimConfig,
        args=[
            "simulator:isaacsim6-newton",
            "--simulator.config.isaacsim.solver-options",
            '{"iterations": 23, "ls_iterations": 7}',
        ],
    )
    selection = cfg.simulator.config.isaacsim
    adapter = TypeAdapter(IsaacSimRenderConfig)
    restored = adapter.validate_json(adapter.dump_json(selection))
    assert restored.solver_options == {"iterations": 23, "ls_iterations": 7}
