"""Format x backend support matrix for scene rigid objects (pure, no simulator).

Every backend loads a different subset of the tri-format sibling fields
(usd_file / urdf_file / xml_file). This locks the FULL matrix in one place via the
shared selector: each SUPPORTED cell picks the right format+path (honouring per-backend
preference order), and each UNSUPPORTED cell raises a clear error naming what was
provided vs. what the backend supports — never silently dropped.

    backend    usd   urdf  xml
    -------    ---   ----  ---
    MuJoCo      x     ok   ok    (prefers xml over urdf)
    IsaacGym    x     ok    x
    IsaacSim   ok     ok    x    (prefers usd over urdf)

Live spawn of the supported cells is covered per backend by test_scene_spawn_mujoco.py
(MuJoCo urdf+xml) and the scene_spawn_assert.py harness (IsaacGym urdf, IsaacSim usd+urdf).
"""

from __future__ import annotations

import pytest

from holosoma.config_types.scene import RigidObjectConfig
from holosoma.simulator.shared.asset_format import select_asset_format

# format -> the RigidObjectConfig kwarg that carries it, with a sample path.
_FIELD = {"usd": "usd_file", "urdf": "urdf_file", "xml": "xml_file"}
_PATH = {"usd": "a.usd", "urdf": "a.urdf", "xml": "a.xml"}

# The expected support matrix. These values are hand-copied from each backend's
# get_supported_scene_formats() return literal:
#   MuJoCo.get_supported_scene_formats()    -> ["xml", "urdf"]   (mujoco.py)
#   IsaacGym.get_supported_scene_formats()  -> ["urdf"]          (isaacgym.py)
#   IsaacSim.get_supported_scene_formats()  -> ["usd", "urdf"]   (isaacsim.py)
# They CANNOT be bound to source here: get_supported_scene_formats is a plain instance
# method whose only access path is through the backend modules, and importing those drags
# in cluster-only SDKs (isaacgym -> `from isaacgym import gymapi`, isaacsim -> `import
# isaaclab.sim`) and torch (mujoco.py top-level), none of which are importorskip-guarded.
# This file is deliberately pure (no simulator), so the matrix is mirrored by hand and the
# selection contract is asserted against select_asset_format. If a backend's supported list
# drifts in source, update the corresponding row here.
BACKENDS = {
    "mujoco": ["xml", "urdf"],
    "isaacgym": ["urdf"],
    "isaacsim": ["usd", "urdf"],
}
ALL_FORMATS = ["usd", "urdf", "xml"]


def _obj(fmt):
    return RigidObjectConfig(**{_FIELD[fmt]: _PATH[fmt]})


@pytest.mark.parametrize(("backend", "supported"), BACKENDS.items())
@pytest.mark.parametrize("fmt", ALL_FORMATS)
def test_single_format_cell(backend, supported, fmt):
    """Every (format, backend) cell: supported -> selected; unsupported -> loud error."""
    obj = _obj(fmt)
    if fmt in supported:
        chosen_fmt, chosen_path = select_asset_format(obj, supported)
        assert chosen_fmt == fmt
        assert chosen_path == _PATH[fmt]
    else:
        with pytest.raises(ValueError, match="but this backend loads") as ei:
            select_asset_format(obj, supported)
        msg = str(ei.value)
        # The message must name what was PROVIDED (the unsupported fmt) AND what the backend
        # loads (every supported fmt), with the contrasting wording — never silently dropped.
        # A vacuous/empty message would fail all of these.
        assert fmt in msg
        assert "but this backend loads" in msg
        assert all(s in msg for s in supported)
        assert "Set one of the supported formats" in msg


def test_object_with_no_asset_errors():
    """An object providing no asset file at all fails loud for every backend."""
    obj = RigidObjectConfig()
    for supported in BACKENDS.values():
        with pytest.raises(ValueError, match="(?i)none"):
            select_asset_format(obj, supported)


def test_preference_order_mujoco_prefers_xml():
    """MuJoCo picks xml over urdf when both are present."""
    obj = RigidObjectConfig(urdf_file="a.urdf", xml_file="a.xml")
    assert select_asset_format(obj, BACKENDS["mujoco"]) == ("xml", "a.xml")


def test_preference_order_isaacsim_prefers_usd():
    """IsaacSim picks usd over urdf when both are present."""
    obj = RigidObjectConfig(usd_file="a.usd", urdf_file="a.urdf")
    assert select_asset_format(obj, BACKENDS["isaacsim"]) == ("usd", "a.usd")


def test_unsupported_format_falls_back_to_supported_sibling():
    """An object with both an unsupported and a supported format uses the supported one."""
    # USD (unsupported on MuJoCo) + urdf (supported) -> urdf, not an error.
    obj = RigidObjectConfig(usd_file="a.usd", urdf_file="a.urdf")
    assert select_asset_format(obj, BACKENDS["mujoco"]) == ("urdf", "a.urdf")
