import tyro
from typing_extensions import Annotated

from holosoma.config_types.experiment import ExperimentConfig
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


def get_annotated_experiment_config() -> type:
    """Return the ``exp:`` subcommand type."""
    return Annotated[  # type: ignore[return-value]  # Annotated[...] alias; mypy models it as object
        ExperimentConfig,
        tyro.conf.arg(
            constructor=tyro.extras.subcommand_type_from_defaults(
                {f"exp:{k.replace('_', '-')}": v for k, v in EXPERIMENT_REGISTRY.items()}
            )
        ),
    ]


from holosoma.utils.config_registry import (  # noqa: E402
    deprecated_defaults_alias as _deprecated_defaults_alias,
)

__getattr__ = _deprecated_defaults_alias(__name__, EXPERIMENT_REGISTRY)
