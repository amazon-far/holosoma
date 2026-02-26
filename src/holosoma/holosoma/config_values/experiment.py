import tyro
from typing_extensions import Annotated

from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_values.loco.g1.experiment import g1_29dof, g1_29dof_fast_sac
from holosoma.config_values.loco.t1.experiment import t1_29dof, t1_29dof_fast_sac
from holosoma.config_values.wbt.g1.experiment import (
    g1_29dof_wbt,
    g1_29dof_wbt_fast_sac,
    g1_29dof_wbt_fast_sac_w_object,
    g1_29dof_wbt_future_motion,
    g1_29dof_wbt_future_motion_no_key_body,
    g1_29dof_wbt_w_object,
)
from holosoma.config_values.wbt.statue.experiment import (
    statue_v0_1_27dof_wbt_stiff,
    statue_v0_1_27dof_wbt_stiff_bootstrap,
    statue_v0_1_wbt_compliant,
    statue_v0_1_wbt_future_motion,
    statue_v0_1_wbt_stiff,
)
from holosoma.config_values.wbt.statue_v1.experiment import (
    statue_v1_rmd_wbt_future_motion,
)

DEFAULTS = {
    "g1_29dof": g1_29dof,
    "g1_29dof_fast_sac": g1_29dof_fast_sac,
    "t1_29dof": t1_29dof,
    "t1_29dof_fast_sac": t1_29dof_fast_sac,
    "g1_29dof_wbt": g1_29dof_wbt,
    "g1_29dof_wbt_w_object": g1_29dof_wbt_w_object,
    "g1_29dof_wbt_fast_sac": g1_29dof_wbt_fast_sac,
    "g1_29dof_wbt_fast_sac_w_object": g1_29dof_wbt_fast_sac_w_object,
    "g1_29dof_wbt_future_motion": g1_29dof_wbt_future_motion,
    "g1_29dof_wbt_future_motion_no_key_body": g1_29dof_wbt_future_motion_no_key_body,
    "statue_v0_1_wbt_stiff": statue_v0_1_wbt_stiff,
    "statue_v0_1_wbt_compliant": statue_v0_1_wbt_compliant,
    "statue_v0_1_wbt_future_motion": statue_v0_1_wbt_future_motion,
    "statue_v0_1_27dof_wbt_stiff": statue_v0_1_27dof_wbt_stiff,
    "statue_v0_1_27dof_wbt_stiff_bootstrap": statue_v0_1_27dof_wbt_stiff_bootstrap,
    "statue_v1_rmd_wbt_future_motion": statue_v1_rmd_wbt_future_motion,
}

AnnotatedExperimentConfig = Annotated[
    ExperimentConfig,
    tyro.conf.arg(
        constructor=tyro.extras.subcommand_type_from_defaults(
            {f"exp:{k.replace('_', '-')}": v for k, v in DEFAULTS.items()}
        )
    ),
]
