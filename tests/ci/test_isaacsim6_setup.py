"""Installer contracts runnable without downloading or launching Isaac Sim."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import venv
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SETUP = ROOT / "scripts/setup_isaacsim6.sh"
SOURCE = ROOT / "scripts/source_isaacsim6_setup.sh"


def _shell(script: str, *args: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", script, "test", *(str(arg) for arg in args)],
        env={**os.environ, **env},
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("version", ["5.1.0", "6.1.0", "6.2.1"])
def test_setup_rejects_wrong_image_runtime_before_installing(tmp_path: Path, version: str) -> None:
    runtime = tmp_path / "sim"
    runtime.mkdir()
    (runtime / "VERSION").write_text(version)
    (runtime / "setup_python_env.sh").write_text("")
    result = _shell(
        'source "$1"; isaacsim6_validate_runtime',
        SETUP,
        env={"HOLOSOMA_DEPS_DIR": str(tmp_path / "deps"), "ISAACSIM_ROOT_PATH": str(runtime)},
    )
    assert result.returncode != 0
    assert f"Expected Isaac Sim 6.2.0; found {version}" in result.stderr
    assert not (tmp_path / "deps").exists()


def test_setup_accepts_supported_runtime_build_suffix(tmp_path: Path) -> None:
    runtime = tmp_path / "sim"
    runtime.mkdir()
    (runtime / "VERSION").write_text("6.2.0-rc.1+release\n")
    (runtime / "setup_python_env.sh").write_text("")
    result = _shell(
        'source "$1"; isaacsim6_validate_runtime',
        SETUP,
        env={"HOLOSOMA_DEPS_DIR": str(tmp_path / "deps"), "ISAACSIM_ROOT_PATH": str(runtime)},
    )
    assert result.returncode == 0, result.stderr


def test_setup_rejects_missing_native_activation(tmp_path: Path) -> None:
    runtime = tmp_path / "sim"
    runtime.mkdir()
    (runtime / "VERSION").write_text("6.2.0")
    result = _shell(
        'source "$1"; isaacsim6_validate_runtime',
        SETUP,
        env={"HOLOSOMA_DEPS_DIR": str(tmp_path / "deps"), "ISAACSIM_ROOT_PATH": str(runtime)},
    )
    assert result.returncode != 0
    assert "Isaac Sim runtime not found" in result.stderr


@pytest.mark.parametrize("already_installed", [False, True])
def test_setup_rejects_changed_isaaclab_checkout(tmp_path: Path, already_installed: bool) -> None:
    checkout = tmp_path / "lab"
    checkout.mkdir()
    for args in [
        ["init", "-q"],
        ["-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "--allow-empty", "-qm", "release"],
        ["tag", "v3.0.0"],
        ["-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "--allow-empty", "-qm", "changed"],
    ]:
        subprocess.run(["git", "-C", str(checkout), *args], check=True, capture_output=True)
    before = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True)
    result = _shell(
        'source "$1"; '
        + ('mkdir -p "$WORKSPACE_DIR"; touch "$ISAACSIM6_SENTINEL"; ' if already_installed else "")
        + "isaacsim6_setup_main",
        SETUP,
        env={
            "HOLOSOMA_DEPS_DIR": str(tmp_path / "deps"),
            "ISAACSIM6_ISAACLAB_DIR": str(checkout),
            "ISAACSIM_ROOT_PATH": "",
        },
    )
    assert result.returncode != 0
    assert "Isaac Lab checkout for isaacsim6 is not at v3.0.0" in result.stderr
    assert subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True) == before


@pytest.mark.parametrize("native", [False, True], ids=["standalone", "native"])
@pytest.mark.parametrize("current_stamp", [False, True], ids=["old-setup", "missing-environment"])
def test_setup_syncs_locked_video_extra_after_previous_setup(tmp_path: Path, native: bool, current_stamp: bool) -> None:
    checkout = tmp_path / "lab"
    checkout.mkdir()
    for args in [
        ["init", "-q"],
        ["-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "--allow-empty", "-qm", "release"],
        ["tag", "v3.0.0"],
    ]:
        subprocess.run(["git", "-C", str(checkout), *args], check=True, capture_output=True)
    lock_packages = ("isaacsim", "isaacsim-kernel", "isaacsim-robot-schema", "isaacsim-asset-isolated")
    wheel_hash = "sha256:" + "a" * 64
    (checkout / "uv.lock").write_text(
        "".join(
            f'[[package]]\nname = "{name}"\nversion = "6.2.0.0"\nwheels = [{{ hash = "{wheel_hash}" }}]\n'
            for name in lock_packages
        )
    )
    deps = tmp_path / "deps"
    deps.mkdir()
    stamp = ".env_setup_finished_hssim6_isaacsim-6.2.0_isaaclab-v3.0.0"
    if current_stamp:
        stamp += "_video_ov_importers_activation"
    (deps / stamp).touch()
    runtime = tmp_path / "sim"
    runtime.mkdir()
    (runtime / "VERSION").write_text("6.2.0")
    (runtime / "setup_python_env.sh").write_text("")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    uv = binaries / "uv"
    uv.write_text(
        """#!/bin/bash
"$INSTALLER_TEST_PYTHON" - "$@" <<'PY'
import json, os, sys, venv
with open(os.environ["INSTALLER_TEST_LOG"], "a") as log:
    log.write(json.dumps(sys.argv[1:]) + "\\n")
if sys.argv[1] == "venv":
    venv.EnvBuilder(with_pip=False).create(sys.argv[-1])
if sys.argv[1] == "sync":
    sys.exit(77)
PY
"""
    )
    uv.chmod(0o755)
    log = tmp_path / "invocations.jsonl"
    env_root = deps / "env"
    result = _shell(
        'source "$1"; isaacsim6_setup_main',
        SETUP,
        env={
            "PATH": f"{binaries}:{os.environ['PATH']}",
            "HOLOSOMA_DEPS_DIR": str(deps),
            "ISAACSIM6_ISAACLAB_DIR": str(checkout),
            "ISAACSIM_ROOT_PATH": str(runtime) if native else "",
            "ISAACSIM6_ENV_ROOT": str(env_root),
            "INSTALLER_TEST_PYTHON": sys.executable,
            "INSTALLER_TEST_LOG": str(log),
        },
    )
    assert result.returncode == 77, result.stderr
    invocations = [json.loads(line) for line in log.read_text().splitlines()]
    expected = [
        [
            "venv",
            "--seed",
            "--allow-existing",
            "--python",
            str(runtime / "kit/python/bin/python3") if native else "3.12",
            str(env_root),
        ]
    ]
    if not native:
        expected.append(["pip", "uninstall", "--python", str(env_root / "bin/python"), "isaacsim", "isaacsim-kernel"])
    expected.append(
        [
            "sync",
            "--frozen",
            "--inexact",
            "--extra",
            "video",
            "--extra",
            "ov",
            "--extra",
            "importers",
            *([] if native else ["--extra", "isaacsim"]),
            *([] if native else ["--no-install-package", "isaacsim", "--no-install-package", "isaacsim-kernel"]),
        ],
    )
    assert invocations == expected
    if not native:
        requirements = (env_root / "isaacsim-runtime-requirements.txt").read_text()
        assert "isaacsim-kernel==6.2.0.0 --hash=sha256:" in requirements
        assert "isaacsim-robot-schema==6.2.0.0 --hash=sha256:" in requirements
        assert "isaacsim-asset-isolated" not in requirements
    new_stamp = ".env_setup_finished_hssim6_isaacsim-6.2.0_isaaclab-v3.0.0_video_ov_importers_activation"
    assert not (deps / new_stamp).exists()


@pytest.mark.parametrize("inherited_backend", [None, "isaacsim5", "isaacsim6", "unsupported"])
def test_activation_keeps_kit_python_packages_deferred(tmp_path: Path, inherited_backend: str | None) -> None:
    env_root = tmp_path / "venv"
    venv.EnvBuilder(with_pip=False).create(env_root)
    runtime = tmp_path / "sim"
    runtime.mkdir()
    prebundle = runtime / "pip_prebundle"
    prebundle.mkdir()
    (prebundle / "runtime_stack_probe.py").write_text('VERSION = "kit"\n')
    site = subprocess.check_output(
        [str(env_root / "bin/python"), "-c", "import site; print(site.getsitepackages()[0])"], text=True
    ).strip()
    (Path(site) / "runtime_stack_probe.py").write_text('VERSION = "venv"\n')
    (runtime / "setup_python_env.sh").write_text(
        'export PYTHONPATH="$ISAAC_PATH/pip_prebundle"\nexport LD_LIBRARY_PATH="$ISAAC_PATH/kit"\n'
    )
    backend_env = {"HOLOSOMA_ISAACSIM_BACKEND": inherited_backend} if inherited_backend is not None else {}
    result = _shell(
        ("unset HOLOSOMA_ISAACSIM_BACKEND; " if inherited_backend is None else "")
        + 'source "$1"; python -c \'import json, os, runtime_stack_probe, sys; '
        'print(json.dumps([runtime_stack_probe.VERSION, os.environ.get("HOLOSOMA_ISAACSIM_BACKEND"), '
        'sys.prefix, os.environ["LD_LIBRARY_PATH"], '
        'os.environ["ISAACSIM6_KIT_PYTHONPATH"], os.environ.get("PYTHONPATH", "")]))\'',
        SOURCE,
        env={
            "HOLOSOMA_DEPS_DIR": str(tmp_path / "deps"),
            "ISAACSIM6_ENV_ROOT": str(env_root),
            "ISAACSIM_ROOT_PATH": str(runtime),
            **backend_env,
        },
    )
    assert result.returncode == 0, result.stderr
    version, backend, prefix, native_paths, deferred_paths, active_paths = json.loads(result.stdout.strip())
    assert [version, backend, prefix] == ["venv", inherited_backend, str(env_root)]
    assert native_paths == str(runtime / "kit")
    assert deferred_paths == f"{site}:{prebundle}"
    assert str(prebundle) not in active_paths.split(os.pathsep)


@pytest.mark.parametrize("kit", [False, True], ids=["standalone", "kit"])
def test_runtime_selection_uses_normal_package_discovery(tmp_path: Path, kit: bool) -> None:
    env_root = tmp_path / "venv"
    venv.EnvBuilder(with_pip=False).create(env_root)
    site = Path(
        subprocess.check_output(
            [str(env_root / "bin/python"), "-c", "import site; print(site.getsitepackages()[0])"], text=True
        ).strip()
    )
    importer = site / "isaacsim/asset/importer/urdf"
    importer.mkdir(parents=True)
    (importer / "__init__.py").write_text('ORIGIN = "standalone"\n')
    runtime = tmp_path / "sim"
    full_root = runtime / "python_packages/isaacsim"
    full_root.mkdir(parents=True)
    (full_root / "__init__.py").write_text('ORIGIN = "kit"\n')
    (runtime / "setup_python_env.sh").write_text(
        'export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$ISAAC_PATH/python_packages"\n'
    )
    activate = ROOT / "src/holosoma/holosoma/simulator/isaacsim/backends/runtime.py"
    consumer = tmp_path / "consumer.py"
    consumer.write_text(
        """import importlib.machinery
import json
import runpy
import sys

if sys.argv[1] == "kit":
    runpy.run_path(sys.argv[2])["activate_kit"]()
import isaacsim
spec = importlib.machinery.PathFinder.find_spec("isaacsim")
if sys.argv[1] == "kit":
    print(json.dumps([isaacsim.ORIGIN, spec.origin == isaacsim.__file__]))
else:
    from isaacsim.asset.importer import urdf
    print(json.dumps([urdf.ORIGIN, spec.origin is None,
                      isinstance(isaacsim.__spec__.loader, importlib.machinery.NamespaceLoader)]))
"""
    )
    result = _shell(
        'source "$1"; python "$2" "$3" "$4"',
        SOURCE,
        consumer,
        Path("kit" if kit else "standalone"),
        activate,
        env={
            "HOLOSOMA_DEPS_DIR": str(tmp_path / "deps"),
            "ISAACSIM6_ENV_ROOT": str(env_root),
            "ISAACSIM_ROOT_PATH": str(runtime),
            # An inherited binary activation must not shadow the namespace.
            "PYTHONPATH": str(runtime / "python_packages"),
        },
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == (["kit", True] if kit else ["standalone", True, True])
