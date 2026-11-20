"""Ablation flag parsing, validation, and application.

Replaces the static ablation_configs.py approach with composable CLI flags
that work with any controller variant experiment config.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_types.observation import ObservationManagerCfg
from holosoma.config_values.arm_tracking.g1.randomization import (
    g1_arm_tracking_dynamics_randomization,
)


@dataclass(frozen=True)
class AblationFlags:
    """Parsed ablation CLI flags."""

    disable_obs: list[str] = field(default_factory=list)
    enable_fk_noise: bool = False
    fk_noise_sigma: float | None = None
    enable_dynamics_rand: bool = False

    def has_any(self) -> bool:
        """Return True if any ablation flag is active."""
        return bool(self.disable_obs) or self.enable_fk_noise or self.enable_dynamics_rand

    def validate(self) -> None:
        """Raise ValueError if flags are inconsistent."""
        if self.enable_fk_noise and self.fk_noise_sigma is None:
            raise ValueError("--enable-fk-noise requires --fk-noise-sigma <float>")
        if self.fk_noise_sigma is not None and not self.enable_fk_noise:
            raise ValueError("--fk-noise-sigma requires --enable-fk-noise")


def _disable_obs_term(
    obs_cfg: ObservationManagerCfg, term_name: str
) -> ObservationManagerCfg:
    """Disable a named observation term across all groups.

    Relocated from ablation_configs.py. Raises ValueError if term_name
    is not found in any group.
    """
    found = False
    new_groups = {}
    for group_name, group_cfg in obs_cfg.groups.items():
        if term_name in group_cfg.terms:
            found = True
            new_terms = {
                k: replace(v, enabled=False) if k == term_name else v
                for k, v in group_cfg.terms.items()
            }
            new_groups[group_name] = replace(group_cfg, terms=new_terms)
        else:
            new_groups[group_name] = group_cfg
    if not found:
        available = sorted(
            {k for g in obs_cfg.groups.values() for k in g.terms}
        )
        raise ValueError(
            f"Unknown observation term '{term_name}'. "
            f"Available terms: {available}"
        )
    return replace(obs_cfg, groups=new_groups)


def apply_ablation_flags(
    config: ExperimentConfig,
    flags: AblationFlags,
) -> ExperimentConfig:
    """Return a new ExperimentConfig with ablation modifications applied.

    Raises ValueError if:
    - A --disable-obs term name doesn't exist in any observation group
    - --enable-fk-noise is set without --fk-noise-sigma
    - --fk-noise-sigma is set without --enable-fk-noise
    """
    flags.validate()

    if not flags.has_any():
        return config

    cfg = config

    # Apply observation term disabling
    for term_name in flags.disable_obs:
        if cfg.observation is None:
            raise ValueError("No observation config to modify")
        cfg = replace(cfg, observation=_disable_obs_term(cfg.observation, term_name))

    # Apply FK noise settings as env-level config overrides
    if flags.enable_fk_noise:
        cfg = replace(
            cfg,
            fk_noise_enabled=True,
            fk_noise_sigma=flags.fk_noise_sigma,
        )

    # Apply dynamics randomization
    if flags.enable_dynamics_rand:
        cfg = replace(cfg, randomization=g1_arm_tracking_dynamics_randomization)

    return cfg

def build_ablation_suffix(flags: AblationFlags) -> str:
    """Build a descriptive suffix string from active ablation flags.

    Examples:
    - "_no-base-pos" for --disable-obs base_pos
    - "_no-base-pos_no-arm-command" for multiple disabled obs
    - "_fk-noise-0.005" for --enable-fk-noise --fk-noise-sigma 0.005
    - "_dynamics-rand" for --enable-dynamics-rand
    """
    if not flags.has_any():
        return ""

    parts: list[str] = []

    for term in flags.disable_obs:
        parts.append(f"no-{term.replace('_', '-')}")

    if flags.enable_fk_noise and flags.fk_noise_sigma is not None:
        parts.append(f"fk-noise-{flags.fk_noise_sigma}")

    if flags.enable_dynamics_rand:
        parts.append("dynamics-rand")

    return "_" + "_".join(parts) if parts else ""

def extract_ablation_flags(argv: list[str]) -> tuple[AblationFlags, list[str]]:
    """Extract ablation-specific flags from argv, returning flags and remaining args.

    Manually parses argv to pull out ablation flags before tyro sees them.
    This avoids conflicts with tyro's own CLI parsing.

    Args:
        argv: Command-line arguments (typically sys.argv[1:]).

    Returns:
        A tuple of (AblationFlags, remaining_argv) where remaining_argv
        contains all arguments not consumed by ablation flag parsing.
    """
    disable_obs: list[str] = []
    enable_fk_noise = False
    fk_noise_sigma: float | None = None
    enable_dynamics_rand = False
    remaining: list[str] = []

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--disable-obs":
            if i + 1 >= len(argv):
                raise ValueError("--disable-obs requires a term name argument")
            disable_obs.append(argv[i + 1])
            i += 2
        elif arg == "--enable-fk-noise":
            enable_fk_noise = True
            i += 1
        elif arg == "--fk-noise-sigma":
            if i + 1 >= len(argv):
                raise ValueError("--fk-noise-sigma requires a float argument")
            try:
                fk_noise_sigma = float(argv[i + 1])
            except ValueError as exc:
                raise ValueError(
                    f"--fk-noise-sigma requires a float argument, got '{argv[i + 1]}'"
                ) from exc
            i += 2
        elif arg == "--enable-dynamics-rand":
            enable_dynamics_rand = True
            i += 1
        else:
            remaining.append(arg)
            i += 1

    flags = AblationFlags(
        disable_obs=disable_obs,
        enable_fk_noise=enable_fk_noise,
        fk_noise_sigma=fk_noise_sigma,
        enable_dynamics_rand=enable_dynamics_rand,
    )
    return flags, remaining


