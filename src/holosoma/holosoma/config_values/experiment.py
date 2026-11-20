from typing import Annotated

import tyro

from holosoma.config_types.experiment import ExperimentConfig, HolosomaConfig
from holosoma.config_values import run, simulator, task
from holosoma.config_values.loco.g1.experiment import g1_29dof, g1_29dof_fast_sac
from holosoma.config_values.loco.t1.experiment import t1_29dof, t1_29dof_fast_sac
from holosoma.config_values.wbt.g1.experiment import (
    g1_29dof_wbt,
    g1_29dof_wbt_fast_sac,
    g1_29dof_wbt_fast_sac_w_object,
    g1_29dof_wbt_w_object,
)
from holosoma.utils.config_registry import ConfigRegistry

EXPERIMENT_REGISTRY = ConfigRegistry(ExperimentConfig, group="holosoma.config.experiment")

EXPERIMENT_REGISTRY.add("g1_29dof", g1_29dof)
EXPERIMENT_REGISTRY.add("g1_29dof_fast_sac", g1_29dof_fast_sac)
EXPERIMENT_REGISTRY.add("t1_29dof", t1_29dof)
EXPERIMENT_REGISTRY.add("t1_29dof_fast_sac", t1_29dof_fast_sac)
EXPERIMENT_REGISTRY.add("g1_29dof_wbt", g1_29dof_wbt)
EXPERIMENT_REGISTRY.add("g1_29dof_wbt_w_object", g1_29dof_wbt_w_object)
EXPERIMENT_REGISTRY.add("g1_29dof_wbt_fast_sac", g1_29dof_wbt_fast_sac)
EXPERIMENT_REGISTRY.add("g1_29dof_wbt_fast_sac_w_object", g1_29dof_wbt_fast_sac_w_object)

sim = HolosomaConfig(
    task=task.sim,
    env_class="holosoma.envs.simulation_task.SimulationTask",
    simulator=simulator.sim_mujoco,
    run=run.interactive,
)
"""The ``sim`` task's base config, not an ``exp:`` preset: one ``SimulationTask`` env with no policy,
bridge and gantry on, in the viewer at real time."""


def _as_holosoma_config(experiment: ExperimentConfig) -> HolosomaConfig:
    """``experiment`` with the default task, run and logger; a full config is kept as registered."""
    return experiment if isinstance(experiment, HolosomaConfig) else HolosomaConfig.from_experiment(experiment)


def get_annotated_experiment_config() -> type:
    """Return the ``exp:`` subcommand type: each experiment preset as a full ``HolosomaConfig``."""
    return Annotated[  # type: ignore[return-value]  # Annotated[...] alias; mypy models it as object
        HolosomaConfig,
        tyro.conf.arg(
            constructor=tyro.extras.subcommand_type_from_defaults(
                {f"exp:{k.replace('_', '-')}": _as_holosoma_config(v) for k, v in EXPERIMENT_REGISTRY.items()}
            )
        ),
    ]


from holosoma.utils.config_registry import (  # noqa: E402
    deprecated_defaults_alias as _deprecated_defaults_alias,
)

__getattr__ = _deprecated_defaults_alias(__name__, EXPERIMENT_REGISTRY)
