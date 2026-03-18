"""Default inference configurations for holosoma_inference."""

from __future__ import annotations

from dataclasses import replace
from importlib.metadata import entry_points

import tyro
from typing_extensions import Annotated

from holosoma_inference.config.config_types.inference import DualModePolicyConfig, InferenceConfig
from holosoma_inference.config.config_values import observation, robot, task

# Safety secondary for G1 configs - FastSAC locomotion.
# Users can select this explicitly via `secondary:g1-29dof-safety-loco`,
# or disable the secondary with `secondary:none`.
g1_safety_loco = InferenceConfig(
    robot=robot.g1_29dof,
    observation=observation.loco_g1_29dof,
    task=task.safety_locomotion_g1,
)

g1_29dof_loco = InferenceConfig(
    robot=robot.g1_29dof,
    observation=observation.loco_g1_29dof,
    task=task.locomotion,
)

t1_29dof_loco = InferenceConfig(
    robot=robot.t1_29dof,
    observation=observation.loco_t1_29dof,
    task=task.locomotion,
)

# fmt: off
_g1_29dof_wbt_robot = replace(
    robot.g1_29dof,
    stiff_startup_pos=(
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # left leg
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # right leg
        0.0, 0.0, 0.0,                          # waist
        0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,      # left arm
        0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,     # right arm
    ),
    stiff_startup_kp=(
        350.0, 200.0, 200.0, 300.0, 300.0, 150.0,
        350.0, 200.0, 200.0, 300.0, 300.0, 150.0,
        200.0, 200.0, 200.0,
        40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0,
        40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0,
    ),
    stiff_startup_kd=(
        5.0, 5.0, 5.0, 10.0, 5.0, 5.0,
        5.0, 5.0, 5.0, 10.0, 5.0, 5.0,
        5.0, 5.0, 5.0,
        3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
        3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
    ),
)

g1_29dof_wbt = InferenceConfig(
    robot=_g1_29dof_wbt_robot,
# fmt: on
    observation=observation.wbt,
    task=task.wbt,
)

# Core defaults - no extension imports at module load time
DEFAULTS: dict[str, DualModePolicyConfig] = {
    "g1-29dof-loco": DualModePolicyConfig(primary=g1_29dof_loco, secondary=g1_safety_loco),
    "t1-29dof-loco": DualModePolicyConfig(primary=t1_29dof_loco, secondary=None),
    "g1-29dof-wbt": DualModePolicyConfig(primary=g1_29dof_wbt, secondary=g1_safety_loco),
    "g1-29dof-safety-loco": DualModePolicyConfig(primary=g1_safety_loco, secondary=None),
}

# Track whether extensions have been loaded
_extensions_loaded = False


def _load_extensions() -> None:
    """Lazily load extension configs from entry points.

    This is deferred to avoid circular imports when extensions import
    from holosoma_inference.config at module load time.
    """
    global _extensions_loaded  # noqa: PLW0603
    if _extensions_loaded:
        return
    _extensions_loaded = True
    for ep in entry_points(group="holosoma.config.inference"):
        DEFAULTS[ep.name] = ep.load()


def _make_dual_mode_constructor():
    """Build subcommand constructor for the top-level config selector."""
    defaults = get_defaults()

    # Validate: each default secondary must be registered as a primary in DEFAULTS,
    # because we ask tyro to build the secondary subcommand choices from the same set.
    primaries = {id(v.primary) for v in defaults.values()}
    for name, cfg in defaults.items():
        if cfg.secondary is not None and id(cfg.secondary) not in primaries:
            raise ValueError(
                f"Default secondary for '{name}' is not a registered primary in DEFAULTS."
            )

    return tyro.extras.subcommand_type_from_defaults(
        {f"inference:{k}": v for k, v in defaults.items()}
    )


AnnotatedDualModePolicyConfig = Annotated[
    DualModePolicyConfig,
    tyro.conf.arg(constructor_factory=_make_dual_mode_constructor),
]


def get_defaults() -> dict[str, DualModePolicyConfig]:
    """Get all config defaults, including extensions."""
    _load_extensions()
    return DEFAULTS
