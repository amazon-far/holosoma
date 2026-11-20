"""v19_fixed experiment: v19 with the action history it was meant to observe.

v19 observes ``last_action`` (a_{T-1}, arms clipped to +-1 and scaled) and
``prev_prev_action`` (raw a_{T-3}: it reads the action manager's buffer after
this step's push, a buffer added for the acceleration penalty, where it does
mean a_{k-2}). The two differ in units and are not adjacent in time, and the
deployed policy fed a_{T-2}, one step fresher than training. They are replaced
by the raw a_{T-1} / a_{T-2} pair (``last_raw_action`` / ``prev_raw_action``),
the quantities the rate and acceleration penalties act on. Actor and critic
change the same way; the actor input stays 204-D.

Nothing else changes: the GAC command is still observed as
``controller_arm_joint_command``. The v19 config is untouched.
"""

from dataclasses import replace

from holosoma.config_types.observation import ObsTermCfg

from holosoma.config_values.loco_arm_tracking.g1.posture_experiment import (
    g1_loco_arm_tracking_posture_fast_sac_sym as _V19,
)


_OBS = "holosoma.managers.observation.terms.loco_arm_tracking:"


_REMOVED = ("last_action", "prev_prev_action")


_ADDED = {
    name: ObsTermCfg(func=_OBS + name, params={}, scale=1.0, noise=0.0)
    for name in ("last_raw_action", "prev_raw_action")
}


def _with_raw_action_history(observation_cfg):
    """Swap v19's two action-history terms for the raw pair in the actor and critic groups."""
    groups = dict(observation_cfg.groups)
    for name in ("actor_obs", "critic_obs"):
        missing = [k for k in _REMOVED if k not in groups[name].terms]
        if missing:
            raise KeyError(f"{name} has no {missing}; v19_fixed expects the v19 observation layout")
        terms = {k: v for k, v in groups[name].terms.items() if k not in _REMOVED}
        groups[name] = replace(groups[name], terms={**terms, **_ADDED})
    return replace(observation_cfg, groups=groups)


g1_loco_arm_tracking_posture_rawhist_fast_sac_sym = replace(
    _V19,
    training=replace(_V19.training, name="g1_loco_arm_tracking_posture_rawhist_fast_sac_sym"),
    observation=_with_raw_action_history(_V19.observation),
)

