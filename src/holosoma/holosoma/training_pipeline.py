"""Building blocks for running a curriculum phase as its own subprocess.

Each phase gets a clean optimizer state and fresh replay buffer, loading only
network weights from the previous phase's checkpoint. Instead of forwarding raw
CLI args to ``tyro.cli()`` (which can fail when complex nested types lack
concrete defaults), each phase is launched via a generated Python script that
constructs the ``ExperimentConfig`` programmatically and calls ``train()``
directly. The phase runner itself is :mod:`holosoma.training_pipeline_renew`;
recipes are run with :mod:`holosoma.train_recipe`.
"""

from __future__ import annotations

import dataclasses
import os
import signal
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from holosoma.config_values.experiment import DEFAULTS


def _kill_process_tree(pid: int) -> None:
    """Send SIGKILL to the entire process group rooted at *pid*."""
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


_SUBCOMMAND_DEFAULTS: dict[str, dict[str, Any]] = {}


def _get_subcommand_defaults() -> dict[str, dict[str, Any]]:
    """Lazily import and cache subcommand DEFAULTS dicts."""
    if _SUBCOMMAND_DEFAULTS:
        return _SUBCOMMAND_DEFAULTS

    import holosoma.config_values.action as _action
    import holosoma.config_values.algo as _algo
    import holosoma.config_values.command as _command
    import holosoma.config_values.curriculum as _curriculum
    import holosoma.config_values.logger as _logger
    import holosoma.config_values.observation as _observation
    import holosoma.config_values.randomization as _randomization
    import holosoma.config_values.reward as _reward
    import holosoma.config_values.robot as _robot
    import holosoma.config_values.simulator as _simulator
    import holosoma.config_values.terrain as _terrain
    import holosoma.config_values.termination as _termination

    _SUBCOMMAND_DEFAULTS.update({
        "algo": _algo.DEFAULTS,
        "simulator": _simulator.DEFAULTS,
        "terrain": _terrain.DEFAULTS,
        "observation": _observation.DEFAULTS,
        "action": _action.DEFAULTS,
        "reward": _reward.DEFAULTS,
        "termination": _termination.DEFAULTS,
        "randomization": _randomization.DEFAULTS,
        "command": _command.DEFAULTS,
        "curriculum": _curriculum.DEFAULTS,
        "robot": _robot.DEFAULTS,
        "logger": _logger.DEFAULTS,
    })
    return _SUBCOMMAND_DEFAULTS


def _coerce_value(value_str: str) -> Any:
    """Convert a CLI string value to its Python type."""
    if value_str == "True":
        return True
    if value_str == "False":
        return False
    if value_str == "None":
        return None
    try:
        return int(value_str)
    except ValueError:
        pass
    try:
        return float(value_str)
    except ValueError:
        pass
    return value_str


def _set_nested_value(obj: Any, dot_path: str, value: Any) -> Any:
    """Return a copy of *obj* with the nested field at *dot_path* set to *value*.

    Handles both frozen-dataclass fields (via ``dataclasses.replace``) and
    plain ``dict`` keys transparently.

    If a dict key along the path does not exist, the override is skipped with a
    warning (this allows phase schedules to reference reward terms that may have
    been removed by variant configs without crashing).
    """
    parts = dot_path.split(".", 1)
    key = parts[0]

    if len(parts) == 1:
        # Leaf — set the value
        if isinstance(obj, dict):
            new = dict(obj)
            new[key] = value
            return new
        return dataclasses.replace(obj, **{key: value})

    # Recurse into the child
    rest = parts[1]
    if isinstance(obj, dict):
        if key not in obj:
            logger.warning(
                f"_set_nested_value: key '{key}' not found in dict while "
                f"setting '{dot_path}' — skipping override (value={value!r})"
            )
            return obj
        child = obj[key]
        new = dict(obj)
        new[key] = _set_nested_value(child, rest, value)
        return new
    child = getattr(obj, key)
    return dataclasses.replace(
        obj, **{key: _set_nested_value(child, rest, value)}
    )


def _parse_base_args(base_args: list[str]) -> tuple[
    str, dict[str, str], dict[str, str], list[str]
]:
    """Parse pipeline base_args into structured components.

    Returns
    -------
    exp_key : str
        Experiment registry key (underscored), e.g. ``"g1_arm_tracking_gac_fast_sac"``.
    subcommand_overrides : dict[str, str]
        Subcommand selectors, e.g. ``{"simulator": "isaacsim", "logger": "wandb"}``.
    field_overrides : dict[str, str]
        Dot-path field overrides, e.g. ``{"training.headless": "True"}``.
    ablation_args : list[str]
        Raw ablation flag args (``--disable-obs``, ``--enable-fk-noise``, etc.).
    """
    subcommand_fields = set(_get_subcommand_defaults().keys())

    exp_key = ""
    subcommand_overrides: dict[str, str] = {}
    field_overrides: dict[str, str] = {}
    ablation_args: list[str] = []

    ablation_flags = {
        "--disable-obs", "--enable-fk-noise", "--fk-noise-sigma",
        "--enable-dynamics-rand",
    }
    # Ablation flags that consume a following value
    ablation_with_value = {"--disable-obs", "--fk-noise-sigma"}

    i = 0
    while i < len(base_args):
        arg = base_args[i]

        # Experiment selector: exp:name
        if arg.startswith("exp:"):
            exp_key = arg[len("exp:"):].replace("-", "_")
            i += 1
            continue

        # Subcommand selector: field:value (e.g. simulator:isaacsim)
        if ":" in arg and not arg.startswith("--"):
            field_name, selector = arg.split(":", 1)
            if field_name in subcommand_fields:
                subcommand_overrides[field_name] = selector
            i += 1
            continue

        # Ablation flags — collect as-is for forwarding
        if arg in ablation_flags:
            ablation_args.append(arg)
            if arg in ablation_with_value and i + 1 < len(base_args):
                i += 1
                ablation_args.append(base_args[i])
            i += 1
            continue

        # Field override: --dotted.path value
        if arg.startswith("--"):
            key = arg[2:]
            if i + 1 < len(base_args):
                field_overrides[key] = base_args[i + 1]
                i += 2
            else:
                i += 1
            continue

        i += 1

    return exp_key, subcommand_overrides, field_overrides, ablation_args


def _read_checkpoint_global_step(checkpoint_path: Path) -> int:
    """Read the global_step from a checkpoint file without loading full weights.

    Uses ``torch.load`` with ``weights_only=False`` but only extracts the
    scalar ``global_step`` key.  Falls back to 0 if the key is missing.
    """
    import torch

    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    return int(ckpt.get("global_step", 0))


def build_config_for_phase(
    base_args: list[str],
    phase: PhaseConfig,
    phase_idx: int = 0,
    checkpoint_path: Path | None = None,
    phase_log_dir: Path | None = None,
    wandb_run_id: str | None = None,
) -> Any:
    """Build an ExperimentConfig for a phase without using tyro.cli().

    Resolves the experiment default, applies subcommand selectors, field
    overrides, ablation flags, and phase-specific overrides programmatically.
    """
    from holosoma.ablation import (
        apply_ablation_flags,
        build_ablation_suffix,
        extract_ablation_flags,
    )

    exp_key, subcommand_overrides, field_overrides, ablation_args = (
        _parse_base_args(base_args)
    )

    # Start from the experiment default
    config = DEFAULTS[exp_key]

    # Apply subcommand selectors (e.g. simulator:isaacsim)
    sc_defaults = _get_subcommand_defaults()
    for field_name, selector in subcommand_overrides.items():
        registry = sc_defaults[field_name]
        if selector in registry:
            config = dataclasses.replace(config, **{field_name: registry[selector]})

    # Apply base field overrides (e.g. --training.headless True)
    for dot_path, raw_value in field_overrides.items():
        config = _set_nested_value(config, dot_path, _coerce_value(raw_value))

    # Apply ablation flags
    ablation_flags_obj, _ = extract_ablation_flags(ablation_args)
    if ablation_flags_obj.has_any():
        config = apply_ablation_flags(config, ablation_flags_obj)
        suffix = build_ablation_suffix(ablation_flags_obj)
        config = dataclasses.replace(
            config,
            training=dataclasses.replace(
                config.training,
                name=config.training.name + suffix,
            ),
        )

    # Apply iteration count.
    # When resuming from a checkpoint the agent's global_step is restored,
    # and the learn loop runs ``while global_step <= num_learning_iterations``.
    # We therefore need: num_learning_iterations = checkpoint_step + phase.num_iterations
    target_iterations = phase.num_iterations
    if checkpoint_path is not None:
        ckpt_step = _read_checkpoint_global_step(checkpoint_path)
        target_iterations = ckpt_step + phase.num_iterations
        logger.info(
            f"Checkpoint at step {ckpt_step} — setting "
            f"num_learning_iterations={target_iterations} "
            f"({ckpt_step} + {phase.num_iterations})"
        )
    config = _set_nested_value(
        config,
        "algo.config.num_learning_iterations",
        target_iterations,
    )

    # Apply checkpoint
    if checkpoint_path is not None:
        config = _set_nested_value(
            config, "training.checkpoint", str(checkpoint_path)
        )

    # Apply logger base_dir
    if phase_log_dir is not None:
        config = _set_nested_value(
            config, "logger.base_dir", str(phase_log_dir)
        )

    # Apply phase-specific overrides
    for dot_path, value in phase.overrides.items():
        config = _set_nested_value(config, dot_path, value)

    # Auto-propagate command.setup_terms.arm_command.params.* overrides
    # to reset_terms and step_terms so the user only needs to specify once.
    _CMD_PREFIX = "command.setup_terms.arm_command.params."
    for dot_path, value in phase.overrides.items():
        if dot_path.startswith(_CMD_PREFIX):
            param_suffix = dot_path[len(_CMD_PREFIX):]
            for stage in ("reset_terms", "step_terms"):
                mirror_path = f"command.{stage}.arm_command.params.{param_suffix}"
                if mirror_path not in phase.overrides:
                    config = _set_nested_value(config, mirror_path, value)

    # Append phase tag to logger.name so runs are distinguishable in wandb
    # Skip when using a shared wandb run ID — all phases share one run.
    if wandb_run_id is None:
        safe_phase_name = phase.name.replace(" ", "_").replace(":", "").replace("/", "_")
        phase_tag = f"_phase{phase_idx + 1}_{safe_phase_name}"
        current_name = getattr(config.logger, "name", None)
        if current_name is not None:
            config = _set_nested_value(
                config, "logger.name", current_name + phase_tag
            )
        else:
            config = _set_nested_value(
                config, "training.name", config.training.name + phase_tag
            )

    # Inject shared wandb run ID so all pipeline phases log to the same run.
    if wandb_run_id is not None:
        config = _set_nested_value(config, "logger.id", wandb_run_id)
        config = _set_nested_value(config, "logger.resume", "allow")
        logger.info(
            f"[Phase {phase_idx}] Injected wandb run_id={wandb_run_id}, "
            f"resume=allow (logger.id={getattr(config.logger, 'id', 'N/A')})"
        )

    return config


@dataclass
class PhaseConfig:
    """A single training phase in the curriculum."""

    name: str
    """Human-readable phase name (e.g., 'Phase A: Fixed Sampling')."""

    num_iterations: int
    """Number of training iterations for this phase."""

    overrides: dict[str, Any] = field(default_factory=dict)
    """CLI parameter overrides applied for this phase.

    Keys are dot-separated Tyro config paths (e.g.,
    ``"command.setup_terms.arm_command.params.use_fixed_sampling"``).
    """


def discover_latest_checkpoint(output_dir: Path) -> Path:
    """Find the most recent checkpoint in a phase's output directory.

    Scans for ``.pt`` files and returns the one with the highest sort key
    (lexicographic, which works for timestamped or iteration-numbered names).

    Parameters
    ----------
    output_dir : Path
        Directory to scan for checkpoint files.

    Returns
    -------
    Path
        Path to the most recent checkpoint file.

    Raises
    ------
    FileNotFoundError
        If no checkpoint files are found in the directory.
    """
    if not output_dir.is_dir():
        raise FileNotFoundError(
            f"Checkpoint directory does not exist: {output_dir}"
        )

    checkpoints = sorted(output_dir.glob("**/*.pt"))
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoint files (*.pt) found in: {output_dir}"
        )

    return checkpoints[-1]

