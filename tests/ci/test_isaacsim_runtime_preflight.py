"""Exercise the CI validator in isolated processes at the native runtime boundary."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PREFLIGHT = Path(__file__).resolve().parents[2] / "scripts/validate_isaacsim_runtime.py"


def _runtime(
    tmp_path: Path,
    *,
    backend: str = "isaaclab3",
    torch_version: str = "2.11.0+cu130",
    warp_version: str = "1.17.0",
    cuda: bool = True,
    devices: int = 1,
    multigpu: bool = False,
    requested_backend: str | None = None,
    missing_video: str | None = None,
) -> subprocess.CompletedProcess[str]:
    lab_version = "28.0.0" if backend == "isaaclab3" else "0.54.2"
    for name, version in [("isaaclab", lab_version), ("torch", torch_version), ("warp-lang", warp_version)]:
        metadata = tmp_path / f"{name.replace('-', '_')}-{version}.dist-info"
        metadata.mkdir()
        (metadata / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
    unavailable = []
    if backend == "isaaclab3":
        (tmp_path / "isaaclab_physx").mkdir()
        (tmp_path / "isaaclab_physx/__init__.py").write_text("")
    else:
        unavailable.append("isaaclab_physx")
    for package in ("moviepy", "imageio_ffmpeg"):
        if backend == "isaaclab3" and package != missing_video:
            (tmp_path / f"{package}.py").write_text("")
        else:
            unavailable.append(package)
    # PYTHONPATH does not hide packages installed in the interpreter's own
    # site-packages. Model absence at the external module boundary.
    (tmp_path / "sitecustomize.py").write_text(
        "import sys\n" + "".join(f"sys.modules[{package!r}] = None\n" for package in unavailable)
    )
    (tmp_path / "torch.py").write_text(
        f"from types import SimpleNamespace\ncuda = SimpleNamespace(is_available=lambda: {cuda}, "
        f"device_count=lambda: {devices})\n"
    )
    return subprocess.run(
        [
            sys.executable,
            str(PREFLIGHT),
            "--backend",
            requested_backend or backend,
            *(["--multigpu"] if multigpu else []),
        ],
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    ("backend", "torch_version", "warp_version", "devices", "multigpu"),
    [
        ("isaaclab2", "2.7.0+cu128", "1.16.0", 1, False),
        ("isaaclab3", "2.11.0+cu130", "1.17.0", 1, False),
        ("isaaclab3", "2.11.0+cu130", "1.17.0", 2, True),
    ],
)
def test_preflight_accepts_matching_native_profiles(
    tmp_path: Path,
    backend: str,
    torch_version: str,
    warp_version: str,
    devices: int,
    multigpu: bool,
) -> None:
    result = _runtime(
        tmp_path,
        backend=backend,
        torch_version=torch_version,
        warp_version=warp_version,
        devices=devices,
        multigpu=multigpu,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "backend": backend,
        "isaaclab_package": "28.0.0" if backend == "isaaclab3" else "0.54.2",
        "torch": torch_version,
        "warp": warp_version,
        "cuda_devices": devices,
    }


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"cuda": False}, "requires a usable CUDA device"),
        ({"devices": 0}, "requires 1 CUDA devices; found 0"),
        ({"devices": 1, "multigpu": True}, "requires 2 CUDA devices; found 1"),
        ({"torch_version": "2.7.0"}, "requires torch 2.11.0; found 2.7.0"),
        ({"warp_version": "1.16.0"}, "requires warp-lang 1.17.0; found 1.16.0"),
        ({"missing_video": "moviepy"}, "Isaac Lab video extra is missing moviepy"),
        ({"missing_video": "imageio_ffmpeg"}, "Isaac Lab video extra is missing imageio_ffmpeg"),
    ],
)
def test_preflight_rejects_environments_that_could_hide_runtime_failures_as_skips(
    tmp_path: Path,
    options: dict,
    message: str,
) -> None:
    result = _runtime(tmp_path, **options)
    assert result.returncode != 0
    assert message in result.stderr
    assert result.stdout == ""


def test_preflight_rejects_wrong_isaac_api_family(tmp_path: Path) -> None:
    result = _runtime(tmp_path, requested_backend="isaaclab2")
    assert result.returncode != 0
    assert "requires Isaac Lab 2.x" in result.stderr
    assert "Isaac Lab 3.x is installed" in result.stderr
