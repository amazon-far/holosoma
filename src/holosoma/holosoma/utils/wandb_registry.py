"""WandB artifact download utilities for motion/terrain data and checkpoints.

Pure WandB SDK calls with no
framework dependencies.
"""

from __future__ import annotations

from pathlib import Path

import wandb


def resolve_tag(registry_name: str, default_tag: str = "latest") -> str:
    """Ensure the registry name includes a tag. If not, append the default tag."""
    if ":" not in registry_name:
        registry_name = f"{registry_name}:{default_tag}"
    return registry_name


def download_motion_and_terrain(
    registry: str | None = None,
    artifact: wandb.Artifact | None = None,
) -> tuple[list[Path] | None, list[Path] | None]:
    """Download motion and terrain artifacts given a full registry name.

    If a registry name does not include a ``":"`` tag, ``":latest"`` is appended
    automatically.

    Returns ``(motion_paths, terrain_paths)`` or ``(None, None)`` if not provided
    or download failed.
    """

    def _download_from_registry(registry_name: str | None) -> Path | None:
        if registry_name and registry_name.startswith("file://"):
            local_path = Path(registry_name[len("file://") :]).expanduser().resolve()
            if not local_path.exists():
                print(f"[WARN] Local registry path does not exist: {local_path}")
                return None
            print(f"[INFO] Using local registry directory: {local_path}")
            return local_path
        if artifact is None and not registry_name:
            return None

        try:
            if artifact is not None:
                resolved_artifact = artifact
            else:
                assert registry_name is not None  # guaranteed by the early return above
                registry_name = resolve_tag(registry_name)
                resolved_artifact = wandb.Api().artifact(registry_name)
            return Path(resolved_artifact.download())
        except Exception as error:
            source = registry_name or getattr(artifact, "name", "provided artifact")
            print(f"[WARN] Failed to download artifact {source!r}: {error}")
            return None

    root = _download_from_registry(registry)

    motion: list[Path] | None = None
    terrain: list[Path] | None = None

    if root:
        motion_matches = sorted(
            (path for path in root.rglob("*") if path.is_file() and "motion" in path.name.lower()),
            key=lambda path: path.name,
        )
        terrain_matches = sorted(
            (path for path in root.rglob("*") if path.is_file() and "terrain" in path.name.lower()),
            key=lambda path: path.name,
        )
        motion = motion_matches or None
        terrain = terrain_matches or None

    if motion:
        print(f"[INFO] Motion file/artifact: {motion}")
    else:
        print("[WARN] No motion artifact downloaded.")
    if terrain:
        print(f"[INFO] Terrain file/artifact: {terrain}")
    else:
        print("[WARN] No terrain artifact downloaded.")

    return motion, terrain


def get_wandb_run_and_file(wandb_path: str) -> tuple[wandb.apis.public.Run, str, str]:
    """Resolve a wandb path to a Run object and model filename.

    Parameters
    ----------
    wandb_path : str
        Either ``"entity/project/run_id"`` or
        ``"entity/project/run_id/model_xxx.pt"``.

    Returns
    -------
    tuple
        ``(run, run_path, file_name)``
    """
    api = wandb.Api()
    # A run ID may contain dots; only a checkpoint suffix denotes an explicit file.
    last_segment = wandb_path.rsplit("/", 1)[-1]
    has_explicit_file = last_segment.endswith(".pt")

    if has_explicit_file:
        run_path = "/".join(wandb_path.split("/")[:-1])
        file_name: str | None = last_segment
    else:
        run_path = wandb_path
        file_name = None

    run = api.run(run_path)

    if file_name is None:
        # Collect .pt checkpoint files and pick the one with the largest index.
        # Workaround for a self-hosted W&B server bug: when fileCount is an exact
        # multiple of the page size (50), pagination returns ``{files: null}``
        # past the last record, which the wandb client dereferences as
        # ``TypeError: 'NoneType' object is not subscriptable``. Fall back to
        # probing ``model_<_step>.pt`` by exact name (run.file() works on the
        # affected runs even though run.files() doesn't).
        try:
            files = [f.name for f in run.files() if f.name.endswith(".pt")]
        except TypeError:
            files = []
            last_step = int((run.summary or {}).get("_step", 0))
            for n in range(last_step, max(last_step - 3, -1), -1):
                try:
                    if run.file(f"model_{n}.pt").size > 0:
                        files.append(f"model_{n}.pt")
                        break
                except Exception as error:
                    print(f"[WARN] Checkpoint probe for model_{n}.pt failed: {error}")
        if not files:
            raise FileNotFoundError(f"No .pt checkpoint files found in run {run_path}.")
        try:
            file_name = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))
        except Exception:
            # Fallback to lexicographic max if naming differs
            file_name = max(files)
    assert file_name is not None
    return run, run_path, file_name


def download_checkpoint_from_run(
    run: wandb.apis.public.Run,
    file_name: str,
    base_dir: str = "./logs/temp/checkpoints",
    replace: bool = True,
) -> Path:
    """Download the given file from the run into ``base_dir/<run_id>/``."""
    run_id: str = run.id
    download_dir = Path(base_dir) / run_id
    download_dir.mkdir(parents=True, exist_ok=True)

    wandb_file = run.file(str(file_name))
    wandb_file.download(str(download_dir), replace=replace)

    resume_path = download_dir / file_name
    if not resume_path.exists():
        raise FileNotFoundError(f"Downloaded checkpoint not found at {resume_path}")
    return resume_path


def download_checkpoint_motion_and_terrain(
    wandb_path: str,
    download_base: str = "./logs/temp/checkpoints",
) -> tuple[Path, list[Path] | None, list[Path] | None]:
    """Download checkpoint, motion and terrain artifacts from a WandB run.

    Returns ``(resume_checkpoint_path, motion_paths, terrain_paths)``.
    """
    run, _run_path, model_file = get_wandb_run_and_file(wandb_path)
    resume_path = download_checkpoint_from_run(run, model_file, base_dir=download_base)

    artifact = next((a for a in run.used_artifacts() if a.type == "terrains-motions"), None)

    if artifact is not None:
        motion, terrain = download_motion_and_terrain(registry=None, artifact=artifact)
    else:
        # No linked artifact — fall back to the registry_name stored in the run's config
        registry_name = (run.config or {}).get("training", {}).get("registry_name")
        if registry_name:
            registry_name = resolve_tag(registry_name)
            print(
                f"[INFO] No terrains-motions artifact linked; falling back to run config registry_name: {registry_name}"
            )
            motion, terrain = download_motion_and_terrain(registry=registry_name)
        else:
            motion, terrain = None, None

    print(f"[INFO] Loaded checkpoint: {resume_path}")
    if motion:
        print(f"[INFO] Motion file/artifact: {motion}")
    else:
        print("[WARN] No motion artifact found.")
    if terrain:
        print(f"[INFO] Terrain file/artifact: {terrain}")
    else:
        print("[WARN] No terrain artifact found.")

    return resume_path, motion, terrain


def download_checkpoint(
    wandb_path: str,
    download_base: str = "./logs/temp/checkpoints",
) -> Path:
    """Download just the checkpoint from a WandB run."""
    run, _, model_file = get_wandb_run_and_file(wandb_path)
    return download_checkpoint_from_run(run, model_file, base_dir=download_base)
