"""Keep image builds, test callers, and release promotion in agreement."""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.no_sim

_ROOT = Path(__file__).resolve().parents[3]
_WORKFLOWS = _ROOT / ".github" / "workflows"


def _workflow(name: str) -> dict:
    workflow = yaml.safe_load((_WORKFLOWS / name).read_text())
    # PyYAML interprets GitHub Actions' `on` key as a YAML 1.1 bool.
    if True in workflow:
        workflow["on"] = workflow.pop(True)
    return workflow


def _build_entries() -> list[dict]:
    return _workflow("build-docker-images.yaml")["jobs"]["build"]["strategy"]["matrix"]["include"]


def test_every_build_entry_names_existing_stages_and_a_runner() -> None:
    """An inserted matrix row must not steal its neighbor's stage and runner fields."""
    for entry in _build_entries():
        assert entry["runner_prefix"], entry
        dockerfile = (_ROOT / entry["dockerfile"]).read_text()
        stages = re.findall(r"^FROM\s+.*?\s+AS\s+(\S+)", dockerfile, re.IGNORECASE | re.MULTILINE)
        assert entry["dependency_target"] in stages, entry
        assert entry["target"] in stages, entry


@pytest.mark.parametrize("name", ["docker-build.yaml", "pytest.yaml", "promote-docker-images.yaml"])
def test_release_steps_include_every_built_image(name: str) -> None:
    """A built image must participate in reuse checks, tested tags, and latest promotion."""
    expected = {entry["image"] for entry in _build_entries()}
    arrays = re.findall(r"^\s*images=\(\n(.*?)^\s*\)", (_WORKFLOWS / name).read_text(), re.MULTILINE | re.DOTALL)
    assert arrays, f"No release image arrays found in {name}"
    for array in arrays:
        images = shlex.split(array)
        assert len(images) == len(expected), (name, images)
        assert set(images) == expected, (name, images)


def test_reusable_test_workflow_callers_supply_all_required_images() -> None:
    """Adding a required image input must update both PR and main-release callers."""
    inputs = _workflow("test-images.yaml")["on"]["workflow_call"]["inputs"]
    required = {name for name, config in inputs.items() if config.get("required") is True}
    callers = []
    for path in _WORKFLOWS.glob("*.yaml"):
        for job in _workflow(path.name)["jobs"].values():
            if job.get("uses") == "./.github/workflows/test-images.yaml":
                callers.append(path.name)
                assert required <= job.get("with", {}).keys(), path.name
    assert callers


def test_promoted_digests_match_image_order() -> None:
    """The positional image/digest arrays must publish each image's own verified digest."""
    workflow = _workflow("promote-docker-images.yaml")
    steps = workflow["jobs"]["promote-latest"]["steps"]
    script = next(step["run"] for step in steps if step["name"] == "Tag as latest")
    image_array = re.search(r"images=\(\n(.*?)^\s*\)", script, re.MULTILINE | re.DOTALL)
    digest_array = re.search(r"digests=\(\n(.*?)^\s*\)", script, re.MULTILINE | re.DOTALL)
    assert image_array and digest_array
    images = shlex.split(image_array.group(1))
    digests = re.findall(r"needs\.verify-images\.outputs\.(\w+)", digest_array.group(1))
    assert digests == [image.replace("-", "_") + "_digest" for image in images]
    outputs = workflow["jobs"]["verify-images"]["outputs"]
    assert all(outputs[digest] == f"${{{{ steps.verify.outputs.{digest} }}}}" for digest in digests)
