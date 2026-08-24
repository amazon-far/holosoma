from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import wandb

from holosoma.utils.wandb_registry import download_motion_and_terrain, resolve_tag

pytestmark = pytest.mark.no_sim


def test_resolve_tag_preserves_explicit_tag() -> None:
    assert resolve_tag("entity/project/name") == "entity/project/name:latest"
    assert resolve_tag("entity/project/name:v2") == "entity/project/name:v2"


def test_local_registry_returns_only_matching_files(tmp_path: Path) -> None:
    motion = tmp_path / "walk_motion.npz"
    terrain = tmp_path / "walk_terrain.npy"
    motion.touch()
    terrain.touch()
    (tmp_path / "motion_directory").mkdir()

    motion_paths, terrain_paths = download_motion_and_terrain(f"file://{tmp_path}")

    assert motion_paths == [motion]
    assert terrain_paths == [terrain]


def test_local_registry_requires_an_existing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        download_motion_and_terrain(f"file://{tmp_path / 'missing'}")


def test_artifact_uses_public_download_api(tmp_path: Path) -> None:
    motion = tmp_path / "walk_motion.npz"
    motion.touch()

    class Artifact:
        def __init__(self) -> None:
            self.download_calls = 0

        def download(self) -> str:
            self.download_calls += 1
            return str(tmp_path)

    artifact = Artifact()
    motion_paths, terrain_paths = download_motion_and_terrain(artifact=cast("wandb.Artifact", artifact))

    assert artifact.download_calls == 1
    assert motion_paths == [motion]
    assert terrain_paths is None


def test_registry_download_adds_latest_tag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    requested: list[str] = []

    class Artifact:
        def download(self) -> str:
            return str(tmp_path)

    class Api:
        def artifact(self, name: str) -> Artifact:
            requested.append(name)
            return Artifact()

    monkeypatch.setattr("holosoma.utils.wandb_registry.wandb.Api", Api)

    download_motion_and_terrain("entity/project/name")

    assert requested == ["entity/project/name:latest"]
