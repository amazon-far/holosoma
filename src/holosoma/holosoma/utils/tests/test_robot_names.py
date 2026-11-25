from __future__ import annotations

import pytest

from holosoma.utils.robot_names import validate_asset_names


def test_exact_names_allow_reordering_and_extra_bodies() -> None:
    validate_asset_names(["tip", "base"], ["base", "helper", "tip"], kind="body")
    validate_asset_names(["b", "a"], ["a", "b"], kind="joint", require_all=True)
    validate_asset_names(["joint.0", "vendor[0]"], ["vendor[0]", "joint.0"], kind="joint", require_all=True)


@pytest.mark.parametrize(
    ("names", "asset", "message"),
    [
        (["a"], ["b"], "no asset matches"),
        (["joint.0"], ["jointx0"], "names must match exactly"),
        (["vendor[0]"], ["vendor0"], "names must match exactly"),
        (["a"], ["a", "a"], "not unique"),
        (["a"], ["a", "b"], "unconfigured"),
    ],
)
def test_invalid_joint_mapping_fails(names: list[str], asset: list[str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_asset_names(names, asset, kind="joint", require_all=True)
