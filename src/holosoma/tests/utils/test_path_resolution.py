"""Unit tests for asset/data path resolution (pure, no simulator).

resolve_data_file_path / resolve_asset_path are the single path-classification layer that
every backend funnels scene + robot assets through (package "holosoma/..." paths, the
"@holosoma/..." alias, s3://, absolute, and asset_root-relative). Two real bugs this branch
fixed lived exactly here (raw ``Path(asset_root)/...`` joins crashing on package paths), so
each classification branch is locked down below.
"""

from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 9):
    from importlib.resources import files
else:
    from importlib_resources import files  # type: ignore[import-not-found]

import pytest

from holosoma.utils.path import resolve_asset_path, resolve_data_file_path

PKG = str(files("holosoma"))  # absolute path of the installed holosoma package


# ----- resolve_data_file_path: one assertion per input class -----


def test_s3_path_returned_as_is():
    assert resolve_data_file_path("s3://bucket/path/to/box.usd") == "s3://bucket/path/to/box.usd"


def test_package_path_resolves_under_holosoma():
    assert (
        resolve_data_file_path("holosoma/data/scene_objects/boxes/small_box.urdf")
        == f"{PKG}/data/scene_objects/boxes/small_box.urdf"
    )


def test_at_holosoma_alias_resolves_identically():
    plain = resolve_data_file_path("holosoma/data/scene_objects/boxes/small_box.urdf")
    alias = resolve_data_file_path("@holosoma/data/scene_objects/boxes/small_box.urdf")
    assert alias == plain == f"{PKG}/data/scene_objects/boxes/small_box.urdf"


def test_package_root_with_trailing_slash_resolves_to_package_dir():
    # The "holosoma/..." rule keys on the trailing slash; "holosoma/" -> the package dir.
    # (Bare "holosoma" with no slash is NOT a package path — it's treated as relative.)
    assert resolve_data_file_path("holosoma/") == PKG


def test_absolute_path_returned_as_is():
    assert resolve_data_file_path("/home/user/custom.npz") == "/home/user/custom.npz"


def test_relative_path_resolved_against_cwd():
    assert resolve_data_file_path("my_data/custom.npz") == str(Path.cwd() / "my_data/custom.npz")


# ----- resolve_asset_path: asset_root applies ONLY to plain relative paths -----


@pytest.mark.parametrize(
    "asset_file",
    [
        "holosoma/data/scene_objects/boxes/small_box.urdf",
        "@holosoma/data/scene_objects/boxes/small_box.urdf",
        "s3://bucket/box.usd",
        "/abs/box.urdf",
    ],
)
def test_self_locating_paths_ignore_asset_root(asset_file):
    """Package/alias/s3/absolute paths are self-locating — asset_root must NOT be joined
    (joining would produce a bogus filesystem path)."""
    assert resolve_asset_path(asset_file, asset_root="/some/root") == resolve_data_file_path(asset_file)


def test_relative_path_joined_onto_asset_root():
    assert resolve_asset_path("box.urdf", asset_root="/some/root") == resolve_data_file_path("/some/root/box.urdf")


def test_relative_path_strips_trailing_slash_on_root():
    assert resolve_asset_path("box.urdf", asset_root="/some/root/") == resolve_data_file_path("/some/root/box.urdf")


def test_relative_path_without_root_falls_back_to_cwd():
    assert resolve_asset_path("box.urdf", asset_root=None) == str(Path.cwd() / "box.urdf")
