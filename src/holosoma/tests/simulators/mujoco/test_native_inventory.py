"""Snapshot the supported upstream API, independently of configuration application."""

import importlib.util
import json
from pathlib import Path

import mujoco
import pytest

pytestmark = pytest.mark.mujoco_classic


def test_supported_upstream_definitions_stay_accounted_for():
    pytest.importorskip("mujoco_warp")
    root = Path(__file__).resolve().parents[5]
    spec = importlib.util.spec_from_file_location("inventory", root / "scripts/inventory_mujoco.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    saved = json.loads((root / "docs/mujoco-parameter-inventory.json").read_text())
    assert module.inventory() == saved, (
        "Supported upstream API changed: review coverage, compatibility, then regenerate inventory"
    )


def test_every_public_option_has_explicit_generic_or_existing_owner_path():
    from holosoma.config_types.mujoco import MujocoConfig, MujocoWarpConfig

    root = Path(__file__).resolve().parents[5]
    saved = json.loads((root / "docs/mujoco-parameter-inventory.json").read_text())
    upstream = {name for name in dir(mujoco.MjOption) if isinstance(getattr(mujoco.MjOption, name), property)}
    assert {r["upstream"].removeprefix("MjOption.") for r in saved["options"]} == upstream
    for record in saved["options"]:
        assert record["path"]
        assert record["notes"]
    assert "timestep" not in MujocoConfig.__dataclass_fields__
    assert saved["extra_name_collisions"] == [], "Ambiguous bare names require dependency compatibility review"
    assert "warp" in MujocoConfig.__dataclass_fields__
    for category in ("warp_options", "allocation", "conversion", "compilation", "capture", "block_dimensions"):
        for record in saved[category]:
            assert record["path"] or record["exclusion"], record["upstream"]
    # The named surface stays focused; less common settings use API-validated mappings.
    assert set(MujocoWarpConfig.__dataclass_fields__) == {
        "nconmax_per_env",
        "njmax_per_env",
        "naconmax",
        "graph_conditional",
        "extras",
    }
