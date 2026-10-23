"""The USD conversion cache must stay out of the source tree and out of the other lane.

Both Isaac Sim lanes install the same holosoma worktree editable, so a cache keyed on the URDF's
own directory -- which lives inside the installed package -- was shared between generations that
produce different USD for the same input.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from holosoma.utils.path import holosoma_deps_dir, urdf_conversion_cache_dir

pytestmark = pytest.mark.no_sim

_SOURCE_COMMON = pathlib.Path(__file__).resolve().parents[5] / "scripts" / "source_common.sh"


def test_holosoma_deps_dir_matches_source_common(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Python default must equal the shell default in ``scripts/source_common.sh``.

    They are two independent spellings of one path; if they drift, assets resolve to different
    places depending on whether the caller went through an activate script.
    """
    monkeypatch.delenv("HOLOSOMA_DEPS_DIR", raising=False)

    shell = _SOURCE_COMMON.read_text()
    match = re.search(r"HOLOSOMA_DEPS_DIR=\$\{HOLOSOMA_DEPS_DIR:-\$HOME/([^\}\s]+)\}", shell)
    assert match, f"could not find the HOLOSOMA_DEPS_DIR default in {_SOURCE_COMMON}"

    assert holosoma_deps_dir() == pathlib.Path.home() / match.group(1)


def test_cache_is_outside_the_source_tree(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """A URDF inside the installed package must not get its conversions written beside it."""
    monkeypatch.setenv("HOLOSOMA_DEPS_DIR", str(tmp_path / "deps"))
    monkeypatch.delenv("HOLOSOMA_USD_CACHE_DIR", raising=False)

    urdf = pathlib.Path(__file__).resolve().parents[1] / "data" / "robots" / "g1" / "g1.urdf"
    cache = pathlib.Path(urdf_conversion_cache_dir(str(urdf)))

    assert urdf.parent not in cache.parents, "conversions must not land next to the URDF"
    assert tmp_path / "deps" in cache.parents


def test_lanes_do_not_share_a_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """Different dependency workspaces -- i.e. different lanes -- must yield different caches.

    This is the property that keeps Isaac Sim 5.1 from reusing 6.1's converted USD and vice versa.
    """
    monkeypatch.delenv("HOLOSOMA_USD_CACHE_DIR", raising=False)
    urdf = "/some/pkg/data/robots/g1/g1.urdf"

    monkeypatch.setenv("HOLOSOMA_DEPS_DIR", str(tmp_path / "holosoma_deps"))
    lane_51 = urdf_conversion_cache_dir(urdf)

    monkeypatch.setenv("HOLOSOMA_DEPS_DIR", str(tmp_path / "holosoma_deps_sim6"))
    lane_61 = urdf_conversion_cache_dir(urdf)

    assert lane_51 != lane_61


def test_robots_do_not_share_a_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """Two robots in one lane must not collide, since the cache is keyed per URDF directory."""
    monkeypatch.setenv("HOLOSOMA_DEPS_DIR", str(tmp_path / "deps"))
    monkeypatch.delenv("HOLOSOMA_USD_CACHE_DIR", raising=False)

    g1 = urdf_conversion_cache_dir("/pkg/data/robots/g1/g1.urdf")
    t1 = urdf_conversion_cache_dir("/pkg/data/robots/t1/t1.urdf")

    assert g1 != t1


def test_explicit_override_wins(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """``HOLOSOMA_USD_CACHE_DIR`` must take precedence, for shared read-only asset caches."""
    monkeypatch.setenv("HOLOSOMA_DEPS_DIR", str(tmp_path / "deps"))
    monkeypatch.setenv("HOLOSOMA_USD_CACHE_DIR", str(tmp_path / "explicit"))

    cache = pathlib.Path(urdf_conversion_cache_dir("/pkg/data/robots/g1/g1.urdf"))

    assert tmp_path / "explicit" in cache.parents
