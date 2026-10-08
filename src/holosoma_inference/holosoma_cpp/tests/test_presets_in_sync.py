"""The C++ presets must equal the Python holosoma_inference presets they are exported from."""

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.no_sim

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_presets.py"


def _export_presets():
    spec = importlib.util.spec_from_file_location("export_presets", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["g1-29dof-loco", "g1-29dof-wbt"])
def test_cpp_preset_matches_python_registry(name):
    exporter = _export_presets()
    assert name in exporter.EXPORTED_PRESETS
    path = exporter.PRESET_DIR / f"{name}.yaml"
    assert path.read_text() == exporter.render_preset(name), (
        f"{path} is stale; run python src/holosoma_inference/holosoma_cpp/scripts/export_presets.py"
    )
