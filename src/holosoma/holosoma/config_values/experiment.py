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
    g1_29dof_wbt_future_motion_heightmap,
    g1_29dof_wbt_future_motion_heightmap_videomimic,
    g1_29dof_wbt_future_motion_heightmap_videomimic_foot_contact,
    g1_29dof_wbt_future_motion_no_key_body,
    g1_29dof_wbt_w_object,
)
from holosoma.config_values.wbt.g1.multi_motion_experiment import (
    g1_29dof_multi_motion,
    g1_29dof_multi_motion_future_motion,
    g1_29dof_multi_terrain,
    g1_29dof_multi_terrain_future_motion,
    g1_29dof_multi_terrain_future_motion_heightmap_videomimic,
    g1_29dof_phuma,
    g1_29dof_phuma_future_motion,
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
    "g1_29dof_wbt_future_motion_heightmap": g1_29dof_wbt_future_motion_heightmap,
    "g1_29dof_wbt_future_motion_heightmap_videomimic": g1_29dof_wbt_future_motion_heightmap_videomimic,
    "g1_29dof_wbt_future_motion_heightmap_videomimic_foot_contact": g1_29dof_wbt_future_motion_heightmap_videomimic_foot_contact,
    "g1_29dof_wbt_future_motion_no_key_body": g1_29dof_wbt_future_motion_no_key_body,
    "g1_29dof_multi_motion": g1_29dof_multi_motion,
    "g1_29dof_multi_motion_future_motion": g1_29dof_multi_motion_future_motion,
    "g1_29dof_multi_terrain": g1_29dof_multi_terrain,
    "g1_29dof_multi_terrain_future_motion": g1_29dof_multi_terrain_future_motion,
    "g1_29dof_multi_terrain_future_motion_heightmap_videomimic": g1_29dof_multi_terrain_future_motion_heightmap_videomimic,
    # Legacy aliases (kept for backward compatibility with existing train scripts)
    "g1_29dof_phuma": g1_29dof_phuma,
    "g1_29dof_phuma_future_motion": g1_29dof_phuma_future_motion,
}

AnnotatedExperimentConfig = Annotated[
    ExperimentConfig,
    tyro.conf.arg(
        constructor=tyro.extras.subcommand_type_from_defaults(
            {f"exp:{k.replace('_', '-')}": v for k, v in DEFAULTS.items()}
        )
    ),
]
