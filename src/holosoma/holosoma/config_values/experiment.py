"""Experiment registry: the configs selectable as ``exp:<name>`` on the CLI.

holosoma's own locomotion and whole-body-tracking experiments, plus the two G1
loco-arm tracking experiments of this release (GAC + RL residual, FastSAC,
symmetry):

* ``g1_loco_arm_tracking_posture_rawhist_fast_sac_sym`` -- v19_fixed, the
  released policy: locomotion + EE tracking + commanded trunk posture, with
  the raw a_{T-1} / a_{T-2} action history.
* ``g1_loco_arm_tracking_posture_fast_sac_sym`` -- v19, the same without that
  history fix, kept for reference.

Train them through the recipes (``python -m holosoma.train_recipe v19_fixed``),
which add the curriculum phases on top of these configs.
"""

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
from holosoma.config_values.loco_arm_tracking.g1.posture_experiment import (
    g1_loco_arm_tracking_posture_fast_sac_sym,
)
from holosoma.config_values.loco_arm_tracking.g1.raw_history_experiment import (
    g1_loco_arm_tracking_posture_rawhist_fast_sac_sym,
)

DEFAULTS = {
    # holosoma: locomotion (velocity tracking) and whole-body tracking.
    "g1_29dof": g1_29dof,
    "g1_29dof_fast_sac": g1_29dof_fast_sac,
    "t1_29dof": t1_29dof,
    "t1_29dof_fast_sac": t1_29dof_fast_sac,
    "g1_29dof_wbt": g1_29dof_wbt,
    "g1_29dof_wbt_w_object": g1_29dof_wbt_w_object,
    "g1_29dof_wbt_fast_sac": g1_29dof_wbt_fast_sac,
    "g1_29dof_wbt_fast_sac_w_object": g1_29dof_wbt_fast_sac_w_object,
    # v19: kept for reference; v19_fixed below is the released policy.
    "g1_loco_arm_tracking_posture_fast_sac_sym": g1_loco_arm_tracking_posture_fast_sac_sym,
    # v19_fixed: v19 with the raw a_{T-1}/a_{T-2} history.
    "g1_loco_arm_tracking_posture_rawhist_fast_sac_sym": g1_loco_arm_tracking_posture_rawhist_fast_sac_sym,
}

AnnotatedExperimentConfig = Annotated[
    ExperimentConfig,
    tyro.conf.arg(
        constructor=tyro.extras.subcommand_type_from_defaults(
            {f"exp:{k.replace('_', '-')}": v for k, v in DEFAULTS.items()}
        )
    ),
]
