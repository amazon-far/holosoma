"""Trunk-orientation (posture) command experiment configs for loco-arm-tracking.

v19's experiment. Adds commanded pelvis and torso orientation on top of the
v18 config, with the command expressed as the target gravity
direction in each trunk body's frame: upright is ``(0, 0, -1)`` and a forward
pitch of theta is ``(sin theta, 0, -cos theta)``.

This captures exactly the roll/pitch degrees of freedom and carries no yaw,
which matters because yaw is already owned by the heading command. It also
mirrors under the sagittal plane by negating y, so symmetry needs only the
trivial mirror functions.

By default the pitch ranges are zero, so the command is upright everywhere and
behaviour matches v18. ``tracking_orientation`` and
``tracking_torso_orientation`` use ``||g_xy - target_xy||^2``, which reduces
exactly to the previous ``||g_xy||^2`` when the target is upright, so the sigma
values keep their meaning and no rescaling is needed.

Configs without posture commands are unaffected: the observation change is a
transform applied only here, and the reward change falls back to plain uprightness when a
config does not command posture.
"""

from dataclasses import replace

from holosoma.config_types.command import CommandTermCfg
from holosoma.config_types.experiment import TrainingConfig

from holosoma.config_values.loco_arm_tracking.g1.observation import (
    with_posture_command,
)
from holosoma.config_values.loco_arm_tracking.g1.smoothness_variant import (
    g1_loco_arm_tracking_gac_acc_height_ground_heading_angj_fast_sac as _BASE,
)


# Ranges actually trained on. The command has to VARY or the policy learns to
# ignore the channel, so a per-episode pitch is sampled from these. Standing:
# pelvis 0-15 deg plus a 0-10 deg waist bend, i.e. torso 0-25 deg absolute,
# never behind the pelvis. Walking gets only a short lean, pelvis 0-5 plus
# waist 0-5 (torso 0-10), since a wide lean may not be walkable. The command
# slews at 30 deg/s between the two.
# The v19 recipe keeps A/B/C upright and widens the range
# through Phase DE.
_POSTURE_RANGES = {
    "pelvis_pitch_range_deg": (0.0, 15.0),
    "waist_pitch_range_deg": (0.0, 10.0),
    "walking_pelvis_pitch_range_deg": (0.0, 5.0),
    "walking_waist_pitch_range_deg": (0.0, 5.0),
    "posture_rate_deg_s": 30.0,
}


def _with_posture_params(command_cfg, params: dict):
    """Return a command config with posture params set on every arm_command stage."""
    out = command_cfg
    for stage in ("setup_terms", "reset_terms", "step_terms"):
        terms = getattr(out, stage, None)
        if not terms or "arm_command" not in terms:
            continue
        term = terms["arm_command"]
        new_term = CommandTermCfg(
            func=term.func, params={**term.params, **params}
        )
        out = replace(out, **{stage: {**terms, "arm_command": new_term}})
    return out


g1_loco_arm_tracking_posture_fast_sac = replace(
    _BASE,
    training=TrainingConfig(
        project="g1-arm-tracking",
        name="g1_loco_arm_tracking_posture_fast_sac",
    ),
    observation=with_posture_command(_BASE.observation),
    command=_with_posture_params(_BASE.command, _POSTURE_RANGES),
)


g1_loco_arm_tracking_posture_fast_sac_sym = replace(
    g1_loco_arm_tracking_posture_fast_sac,
    training=replace(
        g1_loco_arm_tracking_posture_fast_sac.training,
        name="g1_loco_arm_tracking_posture_fast_sac_sym",
    ),
    algo=replace(
        g1_loco_arm_tracking_posture_fast_sac.algo,
        config=replace(
            g1_loco_arm_tracking_posture_fast_sac.algo.config, use_symmetry=True
        ),
    ),
)



