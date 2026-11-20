"""The recipe catalog: every run we train, each readable top to bottom.

``lm_renew()`` is the shared loco-manipulation curriculum. Each version below
reads as its parent plus the lines that differ, which is the whole point: to
see what v19 changed, read v19.

Two recipes are runnable: ``v19_fixed`` (the released policy) and ``v19``
(kept for reference). v16 -- the version the paper's experiments use -- and
v18 are the steps v19 is built from and are kept so the lineage reads top to
bottom; their own experiments are not part of this release, so they cannot be
trained from here.

Run one with::

    python -m holosoma.train_recipe v19_fixed --seed 87
    python -m holosoma.train_recipe v19_fixed --describe DE      # no training

Key names are defined once in :mod:`holosoma.recipes.keys`; that docstring is
the definition of each quantity, including its units and sign convention.
"""

from __future__ import annotations

from typing import Callable

from holosoma.agents.early_stop import EarlyStopCfg
from holosoma.config_types.observation import ObsTermCfg
from holosoma.recipes.recipe import Phase, Recipe


# Datasets. The balanced variants are what v16 onward actually train on.
_TRAJ_B = "data/trajectories/arm_smooth_phaseB.pt"


_TRAJ_DE = "data/trajectories/arm_mixed_phaseDE_taskspace_balanced.pt"


_TRAJ_TELEOP = "data/trajectories/arm_mixed_phaseF_mime_balanced.pt"


_COMBINED_EE_POS = (
    "holosoma.managers.reward.terms.loco_arm_tracking:"
    "tracking_ee_pos_combined_norm_phase_gated"
)


_STABLE = dict(metric="average_episode_length", threshold=990.0, direction="above")


def lm_renew(exp: str) -> Recipe:
    """Phases A -> B -> C -> DE -> F: stand, then arms, then walk, then both.

    A, B and C each learn one thing against a stable version of the others and
    graduate early once episodes survive, so the time they cost is bounded by
    how fast they stabilise, not by their iteration budget. DE combines walking
    with arm tracking and tightens the tracking kernels as it goes; F swaps in
    the combined L2sq+L1 position kernel for the final accuracy.
    """
    r = Recipe(exp, [])

    # -- A: stand still, hands at their default pose --------------------------
    a = r.add_phase(Phase("A", "Phase A: Standing + Fixed EE", 10_000,
                          early_stop=EarlyStopCfg(**_STABLE, patience=1000, min_iterations=5000)))
    a.set("command.arm.fixed_targets", True)
    a.set("command.arm.trajectory_mode", None)
    a.set("command.loco.stand_prob", 1.0)
    a.set("reward.tracking_ee_pos.sigma", 0.02)
    a.set("reward.penalty_arm_residual_l2.weight", -0.1)
    a.set("rand.dof_pos_scale_range", [0.9, 1.1])
    a.set("command.arm.height_min", 0.73)
    a.set("command.arm.height_max", 0.73)

    # -- B: stand still, arms follow a trajectory -----------------------------
    b = r.add_phase(Phase("B", "Phase B: Standing + Arm Trajectory", 10_000,
                          early_stop=EarlyStopCfg(**_STABLE, patience=2000, min_iterations=5000)))
    b.set("command.arm.fixed_targets", False)
    b.set("command.arm.trajectory_mode", "precomputed")
    b.set("command.arm.trajectory_file", _TRAJ_B)
    b.set("command.arm.other_joint_bounds_weight", 0.3)
    b.set("command.loco.stand_prob", 1.0)
    b.set("reward.tracking_ee_pos.sigma", 0.01)
    b.set("reward.tracking_ee_rot.sigma", 0.10)
    b.set("reward.penalty_arm_residual_l2.weight", -0.1)
    b.set("rand.dof_pos_scale_range", [0.85, 1.15])
    b.set("command.arm.height_min", 0.68)
    b.set("command.arm.height_max", 0.78)

    # -- C: slow walk, hands parked -------------------------------------------
    c = r.add_phase(Phase("C", "Phase C: Slow Walk + Fixed Arms", 20_000,
                          early_stop=EarlyStopCfg(**_STABLE, patience=2000, min_iterations=10_000)))
    c.set("command.arm.fixed_targets", True)
    c.set("command.arm.trajectory_mode", None)
    c.set("command.loco.stand_prob", 0.3)
    c.set("command.loco.vel_x", [-0.3, 0.3])
    c.set("command.loco.vel_y", [-0.3, 0.3])
    c.set("command.loco.yaw_rate", [-0.5, 0.5])
    c.set("reward.tracking_ee_pos.sigma", 0.02)
    c.set("reward.tracking_ee_rot.sigma", 0.10)
    c.set("reward.penalty_arm_residual_l2.weight", -0.1)
    c.set("rand.dof_pos_scale_range", [0.85, 1.15])
    c.set("command.arm.height_min", 0.68)
    c.set("command.arm.height_max", 0.78)

    # -- DE: walk AND track, tightening as it goes ----------------------------
    de = r.add_phase(Phase("DE", "Phase DE: Walk + Arm Tracking + Sigma Annealing", 40_000))
    de.set("command.arm.fixed_targets", False)
    de.set("command.arm.trajectory_mode", "precomputed")
    de.set("command.arm.trajectory_file", _TRAJ_DE)
    de.set("command.arm.other_joint_bounds_weight", 0.7)
    de.set("command.loco.stand_prob", 0.2)
    de.set("command.loco.turn_in_place_prob", 0.05)
    de.set("command.loco.vel_x", [-0.7, 0.7])
    de.set("command.loco.vel_y", [-0.7, 0.7])
    de.set("command.loco.yaw_rate", [-0.7, 0.7])
    de.set("reward.tracking_ee_pos.sigma", 0.005)
    de.set("reward.tracking_ee_pos.weight", 4.0)
    de.set("reward.tracking_ee_rot.sigma", 0.10)
    de.set("reward.tracking_ee_rot.weight", 2.5)
    de.set("reward.penalty_arm_residual_l2.weight", -0.1)
    de.set("reward.tracking_torso_orientation.sigma", 0.04)
    de.set("reward.penalty_pose_deviation.weight", -1.0)
    de.set("rand.dof_pos_scale_range", [0.7, 1.3])
    de.set("command.arm.height_min", 0.55)
    de.set("command.arm.height_max", 0.80)
    # Tracking tightens over the middle of the phase; the residual penalty
    # relaxes early so tracking may use more residual once the prior holds.
    de.ramp("reward.tracking_ee_pos.sigma", 0.005, 0.001, from_pct=0.2, until_pct=0.7)
    de.ramp("reward.tracking_ee_rot.sigma", 0.10, 0.02, from_pct=0.2, until_pct=0.7)
    de.ramp("reward.penalty_arm_residual_l2.weight", -0.1, -0.05, from_pct=0.0, until_pct=0.5)
    de.ramp("reward.tracking_lin_vel.sigma", 0.25, 0.10, from_pct=0.2, until_pct=0.8)
    de.ramp("reward.tracking_ang_vel.sigma", 0.25, 0.10, from_pct=0.2, until_pct=0.8)

    # -- F: final accuracy, combined L2sq+L1 position kernel ------------------
    f = r.add_phase(Phase("F", "Phase F: Fine-tune Combined L2sq+L1", 45_000))
    f.set("command.arm.fixed_targets", False)
    f.set("command.arm.trajectory_mode", "precomputed")
    f.set("command.arm.trajectory_file", _TRAJ_DE)
    f.set("command.arm.other_joint_bounds_weight", 0.7)
    f.set("command.loco.stand_prob", 0.2)
    f.set("command.loco.turn_in_place_prob", 0.10)
    f.set("command.loco.vel_x", [-0.7, 0.7])
    f.set("command.loco.vel_y", [-0.7, 0.7])
    f.set("command.loco.yaw_rate", [-0.7, 0.7])
    f.set("reward.tracking_ee_pos.func", _COMBINED_EE_POS)
    f.set("reward.tracking_ee_pos.weight", 4.0)
    f.set("reward.tracking_ee_pos.sq_sigma", 0.001)
    f.set("reward.tracking_ee_pos.l1_sigma", 0.01)
    f.set("reward.tracking_ee_pos.touchdown_gate_width", 0.20)
    f.set("reward.tracking_ee_rot.sigma", 0.02)
    f.set("reward.tracking_ee_rot.weight", 3.0)
    f.set("reward.tracking_torso_orientation.sigma", 0.04)
    f.set("reward.tracking_orientation.sigma", 0.04)
    f.set("reward.penalty_pose_deviation.weight", -1.0)
    f.set("rand.dof_pos_scale_range", [0.7, 1.3])
    f.set("command.arm.height_min", 0.55)
    f.set("command.arm.height_max", 0.80)
    f.ramp("reward.tracking_lin_vel.sigma", 0.10, 0.10, from_pct=0.1, until_pct=0.7)
    f.ramp("reward.tracking_ang_vel.sigma", 0.10, 0.10, from_pct=0.1, until_pct=0.7)
    return r


def with_teleop(r: Recipe, iterations: int, trajectory: str = _TRAJ_TELEOP) -> Recipe:
    """Append the teleop phase, built FROM Phase F rather than copied.

    Only the dataset and the velocity sigmas differ. The sigmas have to be
    stated because this phase carries no ramps: an unset sigma falls back to
    the config default of 0.25 and silently undoes what F annealed.
    """
    teleop = r.derive_phase("F", "TELEOP", "Phase Teleop: VLA Task-Space Mixed", iterations)
    teleop.set("command.arm.trajectory_file", trajectory)
    for k in ("reward.tracking_lin_vel.sigma", "reward.tracking_ang_vel.sigma"):
        ramp = next(rr for rr in r.phase("F").ramps if rr.key.name == k)
        teleop.set(k, ramp.end, by="inherited from F")
    return r


def with_gait_avg_penalties(r: Recipe, yaw: float, lateral: float) -> Recipe:
    """Swap the instantaneous leakage penalties for gait-cycle-averaged ones.

    Each averaged term replaces its instantaneous counterpart, which is zeroed
    so the two cannot double-count, and the critic gains the accumulator state
    those rewards are computed from.
    """
    r.set_all("reward.penalty_yaw_leakage.weight", 0.0)
    r.set_all("reward.penalty_yaw_leakage_gait_avg.weight", yaw)
    r.set_all("reward.penalty_lateral_velocity.weight", 0.0)
    r.set_all("reward.penalty_lateral_velocity_gait_avg.weight", lateral)
    r.set_all(
        "obs.critic.gait_avg_state",
        ObsTermCfg(
            func=(
                "holosoma.managers.observation.terms.loco_arm_tracking:"
                "gait_avg_velocity_state"
            ),
            params={},
            scale=1.0,
            noise=0.0,
        ),
    )
    return r


def with_touchdown_gate(r: Recipe, early: float, late: float) -> Recipe:
    """Set the touchdown gate: ``early`` for A/B/C, ``late`` for DE onward.

    See reward.tracking_ee_pos.touchdown_gate_scale for why the two halves of
    the curriculum want opposite values.
    """
    for pk in r.phase_keys:
        r.set(pk, "reward.tracking_ee_pos.touchdown_gate_scale",
              late if pk in ("DE", "F", "TELEOP") else early)
    return r


_GROUND_HEADING = "g1-loco-arm-tracking-gac-ground-heading-fast-sac-sym"


_POSTURE = "g1-loco-arm-tracking-posture-fast-sac-sym"


def v16() -> Recipe:
    """The baseline this line of runs is measured against.

    Symmetry loss on, gait-averaged leakage penalties, late smoothness
    penalties, and a touchdown gate that tightens early and relaxes from DE on.
    """
    r = lm_renew(_GROUND_HEADING)
    r.set_iterations({"DE": 50_000, "F": 60_000})
    with_teleop(r, iterations=60_000)

    # Late smoothness penalties: introduced at 40% in DE, full from F on. Their
    # job is to smooth arm motion that already tracks, not to suppress motion.
    r.set_per_phase("reward.penalty_arm_dof_vel.weight", {"DE": -0.02, "F": -0.05, "TELEOP": -0.05})
    r.set_per_phase("reward.penalty_ee_body_twist.weight", {"DE": -0.02, "F": -0.05, "TELEOP": -0.05})
    # Foot shaping: pre-touchdown velocity only once the gait is stable.
    r.set_per_phase("reward.penalty_foot_pretouchdown_velocity.weight", {"F": -2.0, "TELEOP": -2.0})
    r.set_all("reward.penalty_foot_contact_force.weight", -2.0)

    with_gait_avg_penalties(r, yaw=-10.0, lateral=-2.0)
    with_touchdown_gate(r, early=0.25, late=4.0)

    # Yaw tolerance stays at 0.25 rather than annealing to 0.10: tightening it
    # trades yaw tracking against the EE error the run is judged on.
    r.phase("DE").ramp("reward.tracking_ang_vel.sigma", 0.25, 0.25, from_pct=0.2, until_pct=0.8)
    r.phase("F").ramp("reward.tracking_ang_vel.sigma", 0.25, 0.25, from_pct=0.1, until_pct=0.7)
    r.phase("TELEOP").set("reward.tracking_ang_vel.sigma", 0.25, by="inherited from F")
    r.phase("TELEOP").set("reward.tracking_lin_vel.sigma", 0.25)

    r.set_all("action.gac.damping_ratio", 2.0)
    r.set_all("rand.torque_rfi.enabled", True)
    r.set_all("rand.torque_rfi.limit", 0.10)
    r.set_all("obs.actor.dof_vel_noise", 0.05)
    r.set_all("algo.symmetry_loss", True)
    r.set_all("algo.symmetry_actor_coef", 5.0)
    r.set_all("algo.symmetry_critic_coef", 0.5)
    return r


def v18() -> Recipe:
    """v16's gate split, with teleop inheriting F's linear velocity sigma.

    The only difference from v16: teleop takes lin_vel 0.10 from Phase F's
    ramp instead of being pinned back to 0.25.
    """
    r = v16()
    r.phase("TELEOP").set("reward.tracking_lin_vel.sigma", 0.10, by="inherited from F")
    return r


def v19() -> Recipe:
    """v18 plus a commanded trunk pitch, introduced only after the prior holds.

    A, B and C stay upright so standing, arm tracking and walking are learned
    first; DE widens the lean from nothing to the config's full range over its
    first half; F and teleop run the full range.
    """
    r = v18().with_exp(_POSTURE)
    r.set_per_phase(
        "command.arm.posture_range_scale",
        {"A": 0.0, "B": 0.0, "C": 0.0, "DE": 0.0, "F": 1.0, "TELEOP": 1.0},
    )
    r.phase("DE").ramp("command.arm.posture_range_scale", 0.0, 1.0, until_pct=0.5)
    return r


_POSTURE_RAWHIST = "g1-loco-arm-tracking-posture-rawhist-fast-sac-sym"


def v19_fixed() -> Recipe:
    """v19 with its two training bugs fixed, for the public release.

    No gravity compensation (v19 had none either). Same curriculum; only
    the experiment changes (see
    ``config_values/loco_arm_tracking/g1/raw_history_experiment.py``):
      * the action history is the raw a_{T-1} / a_{T-2} pair, matched in units
        and adjacent in time (v19 observed a_{T-1} scaled and a_{T-3} raw, and
        its deployment fed a_{T-2});
      * every actor update now holds originals and their mirrors. That fix is
        in the agent (fast_sac_agent.py), not in this recipe, so the v19
        recipe gets it too: v19 itself trained with one all-original and one
        all-mirrored actor update per iteration.
    """
    return v19().with_exp(_POSTURE_RAWHIST)


RECIPES: dict[str, Callable[[], Recipe]] = {
    "v19": v19,
    "v19_fixed": v19_fixed,
}


def get(name: str) -> Recipe:
    try:
        return RECIPES[name]()
    except KeyError:
        raise KeyError(
            f"unknown recipe {name!r}; available: {', '.join(RECIPES)}"
        ) from None

