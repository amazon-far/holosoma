"""WandB artifact download utilities for motion, terrain, and checkpoints."""

from __future__ import annotations

import pathlib

import wandb
from loguru import logger


def resolve_tag(registry_name: str, default_tag: str = "latest") -> str:
    """Ensure the registry name includes a tag. If not, append the default tag."""
    if ":" not in registry_name:
        registry_name = f"{registry_name}:{default_tag}"
    return registry_name


def download_motion_and_terrain(
    registry: str | None = None,
    artifact: wandb.Artifact | None = None,
) -> tuple[list[pathlib.Path] | None, list[pathlib.Path] | None]:
    """Download motion and terrain artifacts given a full registry name.

    If a registry name does not include a ``":"`` tag, ``":latest"`` is appended
    automatically.

    Returns ``(motion_paths, terrain_paths)`` or ``(None, None)`` when no
    artifact was provided.
    """

    def _download_from_registry(registry_name: str | None) -> pathlib.Path | None:
        if artifact is not None:
            art = artifact
        elif not registry_name:
            return None
        elif registry_name.startswith("file://"):
            local_path = pathlib.Path(registry_name[len("file://") :]).expanduser().resolve()
            if not local_path.is_dir():
                raise FileNotFoundError(f"Local registry directory does not exist: {local_path}")
            logger.info(f"Using local registry directory: {local_path}")
            return local_path
        else:
            registry_name = resolve_tag(registry_name)
            try:
                api = wandb.Api()
                art = api.artifact(registry_name)
            except Exception as exc:
                raise RuntimeError(f"Failed to load artifact {registry_name!r}") from exc

        root = pathlib.Path(art.download())
        if not root.is_dir():
            raise FileNotFoundError(f"Downloaded artifact directory does not exist: {root}")
        return root

    root = _download_from_registry(registry)

    motion: list[pathlib.Path] | None = None
    terrain: list[pathlib.Path] | None = None

    if root:
        motion_matches = sorted(p for p in root.rglob("*") if p.is_file() and "motion" in p.name.lower())
        terrain_matches = sorted(p for p in root.rglob("*") if p.is_file() and "terrain" in p.name.lower())
        motion = motion_matches or None
        terrain = terrain_matches or None

    if motion:
        logger.info(f"Motion file/artifact: {motion}")
    else:
        logger.warning("No motion artifact found")
    if terrain:
        logger.info(f"Terrain file/artifact: {terrain}")
    else:
        logger.warning("No terrain artifact found")

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
    # Check if the last segment looks like a filename (has an extension)
    last_segment = wandb_path.rsplit("/", 1)[-1]
    has_explicit_file = "." in last_segment

    if has_explicit_file:
        run_path = "/".join(wandb_path.split("/")[:-1])
        file_name: str | None = last_segment
    else:
        run_path = wandb_path
        file_name = None

    run = api.run(run_path)

    if file_name is None:
        # Collect .pt checkpoint files and pick the one with the largest index
        files = [f.name for f in run.files() if f.name.endswith(".pt")]
        if not files:
            raise FileNotFoundError(
                f"No .pt checkpoint files found in run {run_path}. Available files: {[f.name for f in run.files()]}"
            )
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
) -> pathlib.Path:
    """Download the given file from the run into ``base_dir/<run_id>/``."""
    run_id = run.id
    download_dir = pathlib.Path(base_dir) / run_id
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
) -> tuple[pathlib.Path | None, list[pathlib.Path] | None, list[pathlib.Path] | None]:
    """Download checkpoint, motion and terrain artifacts from a WandB run.

    Returns ``(resume_checkpoint_path, motion_paths, terrain_paths)``.
    """
    run, _, model_file = get_wandb_run_and_file(wandb_path)
    resume_path = download_checkpoint_from_run(run, model_file, base_dir=download_base)

    artifact = next((a for a in run.used_artifacts() if a.type == "terrains-motions"), None)

    if artifact is not None:
        motion, terrain = download_motion_and_terrain(registry=None, artifact=artifact)
    else:
        # No linked artifact; fall back to the registry name stored in the run config.
        registry_name = (run.config or {}).get("training", {}).get("registry_name")
        if registry_name:
            registry_name = resolve_tag(registry_name)
            logger.info(
                f"No terrains-motions artifact linked; falling back to run config registry_name: {registry_name}"
            )
            motion, terrain = download_motion_and_terrain(registry=registry_name)
        else:
            motion, terrain = None, None

    logger.info(f"Loaded checkpoint: {resume_path}")
    if motion:
        logger.info(f"Motion file/artifact: {motion}")
    else:
        logger.warning("No motion artifact found")
    if terrain:
        logger.info(f"Terrain file/artifact: {terrain}")
    else:
        logger.warning("No terrain artifact found")

    return resume_path, motion, terrain


def download_checkpoint(
    wandb_path: str,
    download_base: str = "./logs/temp/checkpoints",
) -> pathlib.Path:
    """Download just the checkpoint from a WandB run."""
    run, _, model_file = get_wandb_run_and_file(wandb_path)
    return download_checkpoint_from_run(run, model_file, base_dir=download_base)
