"""The new runtime must participate in the same test and release gates."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github/workflows"


def _jobs(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text())["jobs"]


@pytest.mark.parametrize(
    ("simulator", "expected"),
    [
        ("isaaclab2", ["--backend", "isaaclab2"]),
        ("isaaclab3", ["--backend", "isaaclab3"]),
        ("isaacsim", ["--backend", "isaaclab2"]),
        ("mujoco", []),
    ],
)
def test_reusable_workflow_passes_explicit_version_only_to_isaac_suite(
    tmp_path: Path, simulator: str, expected: list[str]
) -> None:
    workflow = _jobs("test-simulator-image.yaml")["simulator-tests"]
    assert "HOLOSOMA_ISAACSIM_BACKEND" not in workflow["container"]["env"]
    command = workflow["steps"][0]["run"]
    assert command.count("cd /workspace/holosoma") == 1
    command = command.replace("cd /workspace/holosoma", 'cd "$WORKFLOW_CHECKOUT"').replace(
        "${{ inputs.simulator }}", simulator
    )
    scripts = tmp_path / "tests/ci"
    scripts.mkdir(parents=True)
    suite = "isaaclab" if simulator in ("isaaclab2", "isaaclab3", "isaacsim") else simulator
    (scripts / f"{suite}_ci_tests.sh").write_text('printf "%s\\n" "$@" > "$WORKFLOW_ARG_LOG"\n')
    log = tmp_path / "args"
    result = subprocess.run(
        ["bash", "-e", "-c", command],
        env={
            **os.environ,
            "WORKFLOW_CHECKOUT": str(tmp_path),
            "WORKFLOW_ARG_LOG": str(log),
            "HOLOSOMA_ISAACSIM_BACKEND": "isaaclab2" if simulator == "isaaclab3" else "isaaclab3",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert [line for line in log.read_text().splitlines() if line] == expected


def test_pr_runs_full_isaac_suite_on_both_new_runtime_gpu_shapes_before_tagging() -> None:
    jobs = _jobs("pytest.yaml")
    assert json.loads(jobs["build-isaaclab3"]["with"]["environments"]) == ["isaaclab3"]
    test = jobs["test-isaaclab3"]
    assert "build-isaaclab3" in test["needs"]
    assert test["with"]["simulator"] == "isaaclab3"
    assert json.loads(test["with"]["multigpu"]) == ["False", "True"]
    assert "/holosoma-isaaclab3:candidate-" in test["with"]["image"]
    gate = jobs["pr-image-gate"]
    assert {"build-isaaclab3", "test-isaaclab3"}.issubset(set(gate["needs"]))
    required_results = gate["steps"][0]["env"]["JOB_RESULTS"]
    assert "build-isaaclab3=${{ needs.build-isaaclab3.result }}" in required_results
    assert "test-isaaclab3=${{ needs.test-isaaclab3.result }}" in required_results


def test_release_runs_new_runtime_suite_using_the_verified_immutable_digest() -> None:
    promotion = _jobs("promote-docker-images.yaml")
    assert "holosoma_isaaclab3_digest" in promotion["verify-images"]["outputs"]
    image = promotion["test-images"]["with"]["isaaclab3_image"]
    assert (
        image == "${{ vars.ECR_REGISTRY || '982423663241.dkr.ecr.us-west-2.amazonaws.com' }}"
        "/holosoma-isaaclab3@${{ needs.verify-images.outputs.holosoma_isaaclab3_digest }}"
    )
    test = _jobs("test-images.yaml")["isaaclab3"]
    assert test["with"]["image"] == "${{ inputs.isaaclab3_image }}"
    assert test["with"]["simulator"] == "isaaclab3"
    assert json.loads(test["with"]["multigpu"]) == ["False", "True"]


def test_nightly_covers_new_runtime_locomotion_and_tracking_with_distinct_image() -> None:
    training = _jobs("nightly-training.yaml")["training"]
    matrix = training["strategy"]["matrix"]
    assert set(matrix["simulator"]) == {"isaaclab2", "isaaclab3"}
    assert set(matrix["multigpu"]) == {"True", "False"}
    tracking = {cell["experiment"] for cell in matrix["include"] if cell["simulator"] == "isaaclab3"}
    assert tracking == {"g1-29dof-wbt", "g1-29dof-wbt-fast-sac"}
    run = training["steps"][1]["run"]
    assert "matrix.simulator == 'isaaclab3' && 'holosoma-isaaclab3'" in run
    assert "source scripts/source_${{ matrix.simulator }}_setup.sh" in run
    assert "'IsaacLab/3.0/pip3-envs/default-3.12'" in run
    assert "'Isaac-Sim/5.1/pip3-envs/default'" in run


def test_image_installs_native_shim_before_starting_python_dependency_setup(tmp_path: Path) -> None:
    """Execute the Docker RUN with external shim/uv boundaries stopped before installation."""
    dockerfile = (ROOT / "docker/isaaclab3.Dockerfile").read_text()
    lines = dockerfile.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("RUN --mount=type=cache,target=/root/.cache/uv"))
    command = []
    for line in lines[start + 1 :]:
        command.append(line.removesuffix("\\").strip())
        if not line.endswith("\\"):
            break

    runtime = tmp_path / "isaac-sim"
    runtime.mkdir()
    (runtime / "VERSION").write_text("6.2.0-rc.14+release\n")
    (runtime / "setup_python_env.sh").write_text("")
    lab = tmp_path / "lab"
    installer = lab / "docker/scripts/install_carb_env_shim.sh"
    installer.parent.mkdir(parents=True)
    installer.write_text('#!/bin/bash\nset -e\ntest "$1" = "$ISAACSIM_ROOT_PATH"\necho shim >> "$IMAGE_SETUP_EVENTS"\n')
    for args in [
        ["init", "-q"],
        ["add", "."],
        ["-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "release"],
        ["tag", "v3.0.0"],
    ]:
        subprocess.run(["git", "-C", str(lab), *args], check=True, capture_output=True)
    binaries = tmp_path / "bin"
    binaries.mkdir()
    uv = binaries / "uv"
    uv.write_text('#!/bin/bash\necho "uv:$1" >> "$IMAGE_SETUP_EVENTS"\nexit 77\n')
    uv.chmod(0o755)
    events = tmp_path / "events"
    result = subprocess.run(
        ["bash", "-c", " ".join(command)],
        cwd=ROOT,
        env={
            **os.environ,
            "PATH": f"{binaries}:{os.environ['PATH']}",
            "ISAACLAB3_DIR": str(lab),
            "ISAACSIM_ROOT_PATH": str(runtime),
            "HOLOSOMA_DEPS_DIR": str(tmp_path / "deps"),
            "IMAGE_SETUP_EVENTS": str(events),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 77, result.stderr
    assert events.read_text().splitlines() == ["shim", "uv:venv"]
