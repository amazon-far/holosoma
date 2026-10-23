"""The Isaac Sim 6.1 image must agree with the lane it packages.

Why assert on a Dockerfile from pytest
--------------------------------------
Every fact in ``docker/isaacsim6.Dockerfile`` is a second spelling of something that already
exists elsewhere -- the Isaac Sim version in ``scripts/versions.sh``, the dependency workspace in
``scripts/setup_isaacsim6.sh``, the image registry in ``docker/docker_images.json``. Nothing
re-checks those spellings, and the cost of drift is high and slow to discover: building the image
takes tens of minutes and most mismatches produce a *working* image that is quietly wrong (the
5.1 base, or the 5.1 dependency workspace).

The ``HOLOSOMA_DEPS_DIR`` case is the dangerous one. ``setup_isaacsim6.sh`` deletes and re-clones
the IsaacLab checkout under whatever that variable points at, so an image that inherited the 5.1
lane's default would wipe the other generation's checkout. That is a one-line mistake with no
symptom until it happens, which is exactly what a cheap test is for.

These parse text rather than build anything, so they run on the CPU lane in milliseconds.
"""

from __future__ import annotations

import json
import pathlib
import re

import pytest

pytestmark = pytest.mark.no_sim

_ROOT = pathlib.Path(__file__).resolve().parents[3]
_DOCKERFILE = _ROOT / "docker" / "isaacsim6.Dockerfile"
_SETUP = _ROOT / "scripts" / "setup_isaacsim6.sh"
_VERSIONS = _ROOT / "scripts" / "versions.sh"
_IMAGES_JSON = _ROOT / "docker" / "docker_images.json"
_BUILD_AND_PUSH = _ROOT / "docker" / "build_and_push.sh"


def _dockerfile() -> str:
    assert _DOCKERFILE.is_file(), f"missing {_DOCKERFILE}"
    return _DOCKERFILE.read_text()


def _instructions() -> str:
    """The Dockerfile with comment lines removed.

    Needed because this file's header *explains* what it deliberately omits ("there is no
    miniconda install here"), so a plain substring search for an omitted tool matches the prose
    that says it is absent. Asserting on instructions keeps the checks about what the image does.
    """
    return "\n".join(line for line in _dockerfile().splitlines() if not line.lstrip().startswith("#"))


def _env(dockerfile: str, name: str) -> str:
    match = re.search(rf"^ENV {name}=(\S+)", dockerfile, re.MULTILINE)
    assert match, f"{name} is not set as an ENV in {_DOCKERFILE.name}"
    return match.group(1)


def test_base_image_matches_the_pinned_isaac_sim_version() -> None:
    """The base tag and ``ISAACSIM6_VERSION`` are the same fact written twice.

    A mismatch yields an image that builds and runs against a different Isaac Sim than the lane
    was measured on -- the kind of difference that shows up as unexplained physics or render
    behaviour rather than as a build failure.
    """
    version = re.search(r"^ISAACSIM6_VERSION=(\S+)", _VERSIONS.read_text(), re.MULTILINE)
    assert version, "ISAACSIM6_VERSION not found in scripts/versions.sh"
    assert f"nvcr.io/nvidia/isaac-sim:{version.group(1)}" in _dockerfile()


def test_deps_dir_is_the_sim6_workspace_not_the_5_1_one() -> None:
    """The image must not point this lane's setup at the 5.1 dependency workspace.

    ``setup_isaacsim6.sh`` removes and re-clones the IsaacLab checkout under
    ``HOLOSOMA_DEPS_DIR``, so inheriting the 5.1 default would delete the other generation's
    checkout. The script's own default names the sim6 directory; the image must agree.
    """
    default = re.search(r"HOLOSOMA_DEPS_DIR:-\$HOME/([^\}\s]+)\}", _SETUP.read_text())
    assert default, "could not find setup_isaacsim6.sh's HOLOSOMA_DEPS_DIR default"
    assert _env(_dockerfile(), "HOLOSOMA_DEPS_DIR").endswith("/" + default.group(1))


def test_the_venv_bin_is_on_path() -> None:
    """Without it, ``python`` in the container is Isaac Sim's bundled interpreter, not hssim6.

    The conda lane gets this from putting its ``bin`` on PATH; a uv venv needs the same treatment
    and there is no activate step in a Dockerfile CMD.
    """
    dockerfile = _dockerfile()
    assert "hssim6/bin" in _env(dockerfile, "PATH")


def test_eula_is_accepted_so_the_app_can_launch_unattended() -> None:
    """Isaac Sim blocks on the EULA prompt otherwise, which in CI reads as a hang."""
    assert _env(_dockerfile(), "OMNI_KIT_ACCEPT_EULA") == "1"


def test_conda_is_absent() -> None:
    """Isaac Lab 3.0 rejects conda outright, so the 5.1 image's miniconda step must not be copied.

    A non-empty ``CONDA_PREFIX`` combined with a downloaded Isaac Sim package is a hard error in
    3.0, and its venv check requires ``CONDA_PREFIX`` to be empty.
    """
    instructions = _instructions()
    assert "miniconda" not in instructions.lower()
    assert "CONDA_ROOT" not in instructions


def test_uv_is_version_pinned() -> None:
    """uv resolves this lane's whole dependency set, so an unpinned installer un-pins the image."""
    assert re.search(r"^ARG UV_VERSION=\d+\.\d+", _dockerfile(), re.MULTILINE)


def test_the_setup_script_it_copies_is_the_one_it_runs() -> None:
    """A COPY/RUN mismatch fails the build, but a stale *extra* copy silently bloats the layer."""
    instructions = _instructions()
    assert "scripts/setup_isaacsim6.sh" in instructions
    assert "setup_isaacsim_delta.sh" not in instructions, "that is the 5.1 lane's setup script"


def test_both_build_stages_are_declared() -> None:
    """Downstream images reuse the source-free ``dependencies`` target across source-only releases.

    Collapsing the stages would make every holosoma commit rebuild the whole dependency layer.
    """
    dockerfile = _dockerfile()
    assert "AS dependencies" in dockerfile
    assert "FROM dependencies AS runtime" in dockerfile


def test_the_image_is_registered_for_building_and_publishing() -> None:
    """An unregistered image is never built, so the Dockerfile would silently rot."""
    entry = json.loads(_IMAGES_JSON.read_text())["images"]
    assert "hs-isaacsim6" in entry, "add hs-isaacsim6 to docker/docker_images.json"
    assert entry["hs-isaacsim6"].startswith("holosoma-isaacsim6:")

    table = _BUILD_AND_PUSH.read_text()
    assert "docker/isaacsim6.Dockerfile|holosoma-isaacsim6|hs-isaacsim6" in table, (
        "add an isaacsim6 row to build_and_push.sh's IMAGES table"
    )


def test_the_json_key_and_build_table_agree_on_the_image_name() -> None:
    """Two tables name the same ECR image; a mismatch publishes to the wrong repository."""
    entry = json.loads(_IMAGES_JSON.read_text())["images"]["hs-isaacsim6"]
    image_name = entry.split(":", 1)[0]
    assert f"|{image_name}|hs-isaacsim6|" in _BUILD_AND_PUSH.read_text()
