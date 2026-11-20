"""The key registry: one precise definition per tunable, in one place.

Every value a recipe sets is addressed by a ``Key`` defined here. The key's
docstring IS the definition of the quantity: units, sign convention, the frame
or body it is measured on, and what the extremes mean. Recipes reference keys
and never restate their meaning, so there is exactly one place to read (and to
correct) what a number means.

Naming: ``<area>.<term>.<field>``, e.g. ``reward.tracking_ee_pos.sigma`` or
``command.arm.pelvis_pitch_range``. The dotted config path each key expands to
is an implementation detail carried in ``path``.

Command-term keys use the ``setup_terms`` stage only. The pipeline mirrors
``command.setup_terms.arm_command.params.*`` onto the reset and step stages, so
setting it once is correct and sufficient.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["Key", "KEYS", "key", "REWARD_PARAM", "REWARD_WEIGHT", "COMMAND_ATTR"]


# Kinds decide how a key can be scheduled (ramped within a phase).
REWARD_PARAM = "reward_param"    # reward term params[...]: rampable
REWARD_WEIGHT = "reward_weight"  # reward term weight: rampable
COMMAND_ATTR = "command_attr"    # attribute on a live command term: rampable
PLAIN = "plain"                  # anything else: set per phase, not rampable


@dataclass(frozen=True)
class Key:
    """A single tunable, its config path, its units and its meaning."""

    name: str
    """Short dotted name recipes use, e.g. ``reward.tracking_ee_pos.sigma``."""

    path: str
    """Dotted path into ExperimentConfig that this key sets."""

    doc: str
    """What the quantity IS. Units, sign convention, frame, and what the
    extremes mean. This is the definition of record."""

    unit: str = "dimensionless"
    """Physical unit, e.g. 'm', 'deg', 'm/s', 'iterations', or 'dimensionless'."""

    kind: str = PLAIN
    """One of REWARD_PARAM / REWARD_WEIGHT / COMMAND_ATTR / PLAIN."""

    term: str | None = None
    """Reward or command term name, for keys that can be scheduled."""

    field: str | None = None
    """Params key or command attribute name, for keys that can be scheduled."""

    valid: tuple[Any, Any] | None = None
    """Inclusive (min, max) for numeric keys, or None when unbounded."""

    def check(self, value: Any) -> None:
        """Raise if ``value`` is outside this key's documented valid range."""
        if self.valid is None or not isinstance(value, (int, float)):
            return
        lo, hi = self.valid
        if (lo is not None and value < lo) or (hi is not None and value > hi):
            raise ValueError(
                f"{self.name} = {value} is outside its valid range {self.valid} "
                f"({self.unit}). {self.doc.splitlines()[0]}"
            )


_ARM = "command.setup_terms.arm_command.params."
_LOCO = "command.setup_terms.locomotion_command.params."
_RW = "reward.terms."


def _reward_param(name, term, param, doc, unit="dimensionless", valid=None):
    return Key(name, f"{_RW}{term}.params.{param}", doc, unit, REWARD_PARAM, term, param, valid)


def _reward_weight(name, term, doc, unit="dimensionless", valid=None):
    return Key(name, f"{_RW}{term}.weight", doc, unit, REWARD_WEIGHT, term, "weight", valid)


def _arm(name, param, doc, unit="dimensionless", valid=None, kind=PLAIN):
    return Key(name, f"{_ARM}{param}", doc, unit, kind, "arm_command", param, valid)


def _loco(name, param, doc, unit="dimensionless", valid=None):
    return Key(name, f"{_LOCO}{param}", doc, unit, PLAIN, "locomotion_command", param, valid)


_ALL: list[Key] = [
    # ---------------------------------------------------------------- arm command
    _arm(
        "command.arm.fixed_targets", "use_fixed_sampling",
        """Hold the EE targets at their default pose for the whole episode.

        True means no arm motion is commanded: the targets are the robot's
        default hand poses, reprojected for the commanded base height. Used by
        the phases that are learning to stand or to walk, so arm tracking does
        not compete with the task being learned.""",
        unit="bool",
    ),
    _arm(
        "command.arm.trajectory_mode", "trajectory_mode",
        """How EE targets are produced: None or 'precomputed'.

        None samples targets internally (and must be paired with
        command.arm.fixed_targets). 'precomputed' plays a dataset trajectory
        from command.arm.trajectory_file, one per episode, advancing one frame
        per control step.""",
        unit="enum",
    ),
    _arm(
        "command.arm.trajectory_file", "trajectory_file",
        """Path to the .pt trajectory dataset, relative to the repo root.

        Positions in these datasets are stored in the PELVIS frame of an
        upright robot; with reference_frame 'ground_heading' the z component is
        reprojected by the commanded base height at playback.""",
        unit="path",
    ),
    _arm(
        "command.arm.other_joint_bounds_weight", "other_joint_bounds_weight",
        """Weight on the non-arm joint bounds used when sampling arm targets.

        Higher values keep sampled targets further inside the reachable set of
        the rest of the body. Raised from 0.3 to 0.7 once walking joins arm
        tracking.""",
        valid=(0.0, 1.0),
    ),
    _arm(
        "command.arm.height_min", "height_range_min",
        """Lower end of the commanded PELVIS height, sampled per episode.

        Measured from the ground to the pelvis origin. Nominal standing height
        is 0.73. The v18 policy saturates near 0.54, so values below that are
        commanded but not achieved.""",
        unit="m", valid=(0.3, 1.0),
    ),
    _arm(
        "command.arm.height_max", "height_range_max",
        """Upper end of the commanded PELVIS height, sampled per episode.

        See command.arm.height_min. Equal min and max pins the height.""",
        unit="m", valid=(0.3, 1.0),
    ),
    _arm(
        "command.arm.pelvis_pitch_range", "pelvis_pitch_range_deg",
        """(min, max) forward pitch of the PELVIS, sampled once per episode.

        Positive is a forward (nose-down) lean. The command is carried as the
        target gravity direction in the pelvis frame: upright is (0, 0, -1) and
        a pitch of t is (sin t, 0, -cos t). That carries roll and pitch but no
        yaw, which is deliberate: yaw is owned by the heading command.

        This range applies while the velocity command is exactly zero. Walking
        episodes use command.arm.walking_pelvis_pitch_range instead, and the
        commanded pitch slews between the two at command.arm.posture_rate.

        The sampled value is multiplied by command.arm.posture_range_scale.""",
        unit="deg", valid=(0.0, 45.0),
    ),
    _arm(
        "command.arm.waist_pitch_range", "waist_pitch_range_deg",
        """(min, max) WAIST bend, sampled per episode ON TOP OF the pelvis pitch.

        The torso pitch command becomes pelvis + waist, so the torso can never
        end up behind the pelvis (a backward waist bend), which independent
        sampling of command.arm.torso_pitch_range allows. When this key is set
        it supersedes command.arm.torso_pitch_range.""",
        unit="deg", valid=(0.0, 45.0),
    ),
    _arm(
        "command.arm.torso_pitch_range", "torso_pitch_range_deg",
        """(min, max) forward pitch of the TORSO, ABSOLUTE, not relative.

        Measured at torso_link, i.e. pelvis pitch PLUS waist bend. A 30 deg
        pelvis with a 15 deg waist bend is 45 here, NOT 15. Same gravity-
        direction representation as command.arm.pelvis_pitch_range.

        Prefer command.arm.waist_pitch_range, which expresses the same thing as
        an offset and cannot produce a backward waist bend.""",
        unit="deg", valid=(0.0, 60.0),
    ),
    _arm(
        "command.arm.walking_pelvis_pitch_range", "walking_pelvis_pitch_range_deg",
        """(min, max) pelvis pitch used while the velocity command is NON-zero.

        A wide lean is not walkable, so walking gets its own, smaller range.
        Set together with command.arm.walking_waist_pitch_range; leaving both
        unset means the standing range applies regardless of velocity.

        Turning in place counts as walking, since its yaw command is non-zero.""",
        unit="deg", valid=(0.0, 20.0),
    ),
    _arm(
        "command.arm.walking_waist_pitch_range", "walking_waist_pitch_range_deg",
        """(min, max) waist bend while walking, on top of the walking pelvis pitch.

        See command.arm.waist_pitch_range for the offset convention and
        command.arm.walking_pelvis_pitch_range for when it applies.""",
        unit="deg", valid=(0.0, 20.0),
    ),
    _arm(
        "command.arm.posture_rate", "posture_rate_deg_s",
        """Slew rate limit on the commanded trunk pitch.

        The command moves toward its goal by at most this much per second, so a
        stand/walk switch or an episode reset never steps the command. Applies
        to the pelvis and torso channels alike.""",
        unit="deg/s", valid=(1.0, 360.0),
    ),
    _arm(
        "command.arm.posture_range_scale", "posture_range_scale",
        """Curriculum scale on the per-episode trunk pitch.

        Multiplies the pitch sampled at each episode reset, so the range always
        widens outward from upright: 0.0 commands upright whatever the
        configured ranges are, 1.0 uses them in full.

        Changing it affects only episodes that reset afterwards; it never
        changes the goal an in-flight episode is already slewing toward. This
        is the key the DE phase ramps to widen the lean gradually.""",
        valid=(0.0, 1.0), kind=COMMAND_ATTR,
    ),
    # --------------------------------------------------------- locomotion command
    _loco(
        "command.loco.stand_prob", "stand_prob",
        """Probability that a resampled velocity command is exactly zero.

        Velocity commands resample every 10 s of episode time as well as at
        reset, so this is the fraction of TIME spent standing, not the fraction
        of episodes. 1.0 means the robot never walks.""",
        valid=(0.0, 1.0),
    ),
    _loco(
        "command.loco.turn_in_place_prob", "turn_in_place_prob",
        """Probability that a resampled command keeps yaw but zeroes linear velocity.

        Drawn among the commands that did not already become a standing command,
        so the two probabilities do not compete.""",
        valid=(0.0, 1.0),
    ),
    _loco(
        "command.loco.vel_x", "command_ranges.lin_vel_x",
        """[min, max] commanded forward velocity, in the heading frame.

        Positive is forward. Sampled uniformly at each resample.""",
        unit="m/s",
    ),
    _loco(
        "command.loco.vel_y", "command_ranges.lin_vel_y",
        """[min, max] commanded lateral velocity, in the heading frame.

        Positive is to the robot's left.""",
        unit="m/s",
    ),
    _loco(
        "command.loco.yaw_rate", "command_ranges.ang_vel_yaw",
        """[min, max] commanded yaw rate. Positive turns left.""",
        unit="rad/s",
    ),
    # ------------------------------------------------------------- EE tracking
    _reward_param(
        "reward.tracking_ee_pos.sigma", "tracking_ee_pos", "tracking_sigma",
        """Kernel width for EE position tracking: exp(-e^2 / sigma).

        e is the position error in metres, so sigma has units of m^2 and
        SMALLER is tighter. Only used by the plain L2 kernel; the combined
        L2sq+L1 kernel used from Phase F on reads sq_sigma and l1_sigma
        instead.""",
        unit="m^2", valid=(1e-5, 1.0),
    ),
    _reward_param(
        "reward.tracking_ee_pos.sq_sigma", "tracking_ee_pos", "sq_sigma",
        """Squared-error kernel width of the combined L2sq+L1 EE position reward.

        Dominates when the hand is already close to its target.""",
        unit="m^2", valid=(1e-5, 1.0),
    ),
    _reward_param(
        "reward.tracking_ee_pos.l1_sigma", "tracking_ee_pos", "l1_sigma",
        """Absolute-error kernel width of the combined L2sq+L1 EE position reward.

        Dominates far from the target, where it keeps a usable gradient that a
        squared kernel alone would have flattened.""",
        unit="m", valid=(1e-4, 1.0),
    ),
    _reward_param(
        "reward.tracking_ee_pos.touchdown_gate_scale", "tracking_ee_pos",
        "touchdown_gate_sigma_scale",
        """Multiplier on the EE position sigma at peak foot touchdown.

        sigma_eff = sigma * lerp(1.0, scale, gate), where gate peaks at
        footstrike. Below 1 TIGHTENS the kernel there, above 1 RELAXES it, and
        1.0 disables the gate.

        Measured effect (our ablation, same seed): 0.25 helps Phase DE, whose
        plain exp(-e^2/sigma) kernel still returns a useful value at a typical
        touchdown error. It HURTS Phase F and teleop, where the scale
        multiplies both sigmas of the combined kernel, the reward saturates to
        about 1e-4 around footstrike, and those steps stop carrying gradient;
        ee_pos_error ended about twice as bad through teleop. Hence 0.25 early,
        4.0 from DE on.""",
        valid=(0.01, 100.0),
    ),
    _reward_param(
        "reward.tracking_ee_pos.touchdown_gate_width", "tracking_ee_pos",
        "touchdown_gate_phase_width",
        """Width of the touchdown gate, as a fraction of the gait cycle.

        The gate is centred on footstrike, so 0.20 covers roughly a fifth of
        the cycle around it.""",
        valid=(0.0, 1.0),
    ),
    Key(
        "reward.tracking_ee_pos.func", f"{_RW}tracking_ee_pos.func",
        """Which EE position reward function is active.

        The plain L2 form reads reward.tracking_ee_pos.sigma; the combined
        L2sq+L1 form reads sq_sigma and l1_sigma. Switching the function
        without switching the sigmas silently leaves the wrong ones in use.""",
        "callable-path", PLAIN, "tracking_ee_pos", "func",
    ),
    _reward_weight(
        "reward.tracking_ee_pos.weight", "tracking_ee_pos",
        """Weight on the EE position tracking reward.""",
    ),
    _reward_param(
        "reward.tracking_ee_rot.sigma", "tracking_ee_rot", "tracking_sigma",
        """Kernel width for EE orientation tracking.

        The error is a rotation-matrix geodesic distance, so this is in
        rad^2-like units; smaller is tighter.""",
        unit="rad^2", valid=(1e-4, 10.0),
    ),
    _reward_weight(
        "reward.tracking_ee_rot.weight", "tracking_ee_rot",
        """Weight on the EE orientation tracking reward.""",
    ),
    # ------------------------------------------------------------ trunk / posture
    _reward_param(
        "reward.tracking_orientation.sigma", "tracking_orientation", "tracking_sigma",
        """Kernel width for PELVIS orientation tracking.

        The error is ||g_xy - target_xy||^2, where g is the gravity direction
        in the pelvis frame. With an upright target this reduces exactly to
        ||g_xy||^2, the plain uprightness penalty, so the sigma keeps its
        meaning when a posture command is introduced.""",
        valid=(1e-4, 10.0),
    ),
    _reward_param(
        "reward.tracking_torso_orientation.sigma", "tracking_torso_orientation",
        "tracking_sigma",
        """Kernel width for TORSO orientation tracking, measured at torso_link.

        Same error form as reward.tracking_orientation.sigma.""",
        valid=(1e-4, 10.0),
    ),
    # ------------------------------------------------------------ velocity tracking
    _reward_param(
        "reward.tracking_lin_vel.sigma", "tracking_lin_vel", "tracking_sigma",
        """Kernel width for linear velocity tracking; smaller is tighter.

        Phases without a scheduler do not inherit this automatically: leaving
        it unset falls back to the config default of 0.25 and silently loosens
        a tolerance an earlier phase had annealed.""",
        unit="(m/s)^2", valid=(1e-3, 10.0),
    ),
    _reward_param(
        "reward.tracking_ang_vel.sigma", "tracking_ang_vel", "tracking_sigma",
        """Kernel width for yaw rate tracking; smaller is tighter.

        See reward.tracking_lin_vel.sigma on inheritance.""",
        unit="(rad/s)^2", valid=(1e-3, 10.0),
    ),
    # ------------------------------------------------------------------ penalties
    _reward_weight(
        "reward.penalty_arm_residual_l2.weight", "penalty_arm_residual_l2",
        """Weight on the squared GAC arm residual (negative = penalty).

        Suppresses how far the policy's residual pushes the arm away from the
        controller's own solution. Relaxed from -0.1 to -0.05 during DE so
        tracking can use more residual once the prior is in place.""",
        valid=(None, 0.0),
    ),
    _reward_weight(
        "reward.penalty_arm_dof_vel.weight", "penalty_arm_dof_vel",
        """Weight on squared arm joint velocity (negative = penalty).

        A smoothness term, introduced late: applying it before arm tracking
        works suppresses the motion instead of smoothing it.""",
        valid=(None, 0.0),
    ),
    _reward_weight(
        "reward.penalty_ee_body_twist.weight", "penalty_ee_body_twist",
        """Weight on end-effector body twist (negative = penalty).

        Penalises hand angular velocity, i.e. flicking the wrist to chase
        orientation targets. Introduced late, like penalty_arm_dof_vel.""",
        valid=(None, 0.0),
    ),
    _reward_weight(
        "reward.penalty_pose_deviation.weight", "penalty_pose_deviation",
        """Weight on weighted joint deviation from the default pose.

        The per-joint weights live in the term's pose_weights param, not here.
        Waist pitch is lowered once a torso orientation command regularises it,
        and ankle pitch can be lowered for deep leans because ankle
        dorsiflexion is what bounds them.""",
        valid=(None, 0.0),
    ),
    _reward_weight(
        "reward.penalty_foot_contact_force.weight", "penalty_foot_contact_force",
        """Weight on excessive vertical foot contact force (negative = penalty).

        Discourages stamping.""",
        valid=(None, 0.0),
    ),
    _reward_weight(
        "reward.penalty_foot_pretouchdown_velocity.weight",
        "penalty_foot_pretouchdown_velocity",
        """Weight on foot velocity just before touchdown (negative = penalty).

        Shapes the foot to arrive slowly rather than be caught at impact.
        Enabled from Phase F on, once the gait itself is stable.""",
        valid=(None, 0.0),
    ),
    _reward_weight(
        "reward.penalty_yaw_leakage.weight", "penalty_straight_walk_yaw_leakage",
        """Weight on instantaneous yaw drift while walking straight.

        Superseded by its gait-averaged counterpart when that is enabled: the
        instantaneous weight is then zeroed, because the two would double-count
        the same error.""",
        valid=(None, 0.0),
    ),
    _reward_weight(
        "reward.penalty_yaw_leakage_gait_avg.weight",
        "penalty_straight_walk_yaw_leakage_gait_avg",
        """Weight on yaw drift averaged over a gait cycle.

        Averaging over the cycle stops the penalty from fighting the yaw
        excursion that is part of a normal step. Enabling it adds the
        gait_avg_velocity_state accumulator to the CRITIC observation, which is
        why the critic is 5 dims wider than the config alone suggests.""",
        valid=(None, 0.0),
    ),
    _reward_weight(
        "reward.penalty_lateral_velocity.weight", "penalty_lateral_velocity",
        """Weight on instantaneous lateral velocity while walking straight.

        Zeroed when the gait-averaged variant is enabled.""",
        valid=(None, 0.0),
    ),
    _reward_weight(
        "reward.penalty_lateral_velocity_gait_avg.weight",
        "penalty_lateral_velocity_gait_avg",
        """Weight on lateral velocity averaged over a gait cycle.

        See reward.penalty_yaw_leakage_gait_avg.weight for the critic-obs
        side effect.""",
        valid=(None, 0.0),
    ),
    # ------------------------------------------------------------- randomization
    Key(
        "rand.dof_pos_scale_range",
        "randomization.reset_terms.randomize_dof_state.params.joint_pos_scale_range",
        """[min, max] multiplier on default joint positions at episode reset.

        Widened as the curriculum progresses so later phases start from more
        varied postures.""",
    ),
    Key(
        "rand.torque_rfi.enabled",
        "randomization.setup_terms.setup_torque_rfi.params.enabled",
        """Enable random force injection: per-step random torque on every joint.

        A sim-to-real term standing in for unmodelled actuator dynamics.""",
        "bool",
    ),
    Key(
        "rand.torque_rfi.limit",
        "randomization.setup_terms.setup_torque_rfi.params.rfi_lim",
        """Random torque amplitude, as a fraction of each joint's torque limit.""",
        valid=(0.0, 1.0),
    ),
    # -------------------------------------------------------------- observation
    Key(
        "obs.actor.dof_vel_noise",
        "observation.groups.actor_obs.terms.dof_vel.noise",
        """Gaussian noise added to the ACTOR's joint velocity observation.

        Sized against real encoder-difference noise so the policy does not
        learn to trust a signal the hardware does not provide this cleanly.""",
        valid=(0.0, 1.0),
    ),
    Key(
        "obs.critic.gait_avg_state",
        "observation.groups.critic_obs.terms.gait_avg_velocity_state",
        """Critic-only observation of the gait-average accumulator (5 dims).

        Required by the gait-averaged leakage penalties: without it the critic
        cannot see the accumulator state those rewards depend on, and the value
        function has to guess. Present only when those penalties are on, which
        makes the critic 5 dims wider than the base config.""",
        "obs-term",
    ),
    # ------------------------------------------------------------------ logging
    Key(
        "logging.video", "logger.video.enabled",
        """Record and upload rollout videos during training.

        Operational, not part of what a recipe trains: it renders a camera in
        the simulator every logger.video.interval episodes, which in headless
        IsaacSim is expensive enough to slow training noticeably. The launcher
        turns it off by default; recipes leave it alone.""",
        "bool",
    ),
    # --------------------------------------------------------------- action / algo
    Key(
        "action.gac.damping_ratio",
        "action.terms.loco_arm_residual.params.controller_kwargs.damping_ratio",
        """Damping ratio of the GAC arm impedance controller.

        1.0 is critically damped; above 1 is overdamped, trading tracking
        bandwidth for less overshoot at footstrike.""",
        valid=(0.1, 10.0),
    ),
    Key(
        "training.num_envs", "training.num_envs",
        """Parallel simulated environments.

        Sets sample throughput and diversity, and it is the multiplier on
        replay storage: the buffer holds num_envs x algo.replay_buffer_size
        single frames. Lowering it is the widest memory lever, because it
        shrinks the simulator and the buffer together.""",
        unit="envs", valid=(1, 65536),
    ),
    Key(
        "algo.batch_size", "algo.config.batch_size",
        """Transitions per gradient update.

        Memory cost is this times algo.updates_per_step, because a step samples
        every update's batch at once, and doubled again by symmetry
        augmentation.""",
        valid=(64, 65536),
    ),
    Key(
        "algo.sample_chunk_updates", "algo.config.sample_chunk_updates",
        """Draw replay samples this many updates at a time; 0 = all at once.

        A memory setting. Each learning step draws algo.batch_size for every
        one of its updates, doubled by symmetry. With symmetry the agent draws
        one update at a time regardless, so every update holds originals and
        their mirrors (a multi-update draw would give updates originals only or
        mirrors only).""",
        valid=(0, 64),
    ),
    Key(
        "algo.replay_buffer_size", "algo.config.buffer_size",
        """Replay capacity PER ENVIRONMENT; total transitions is this x num_envs.""",
        unit="steps", valid=(64, 100000),
    ),
    Key(
        "algo.symmetry_loss", "algo.config.use_symmetry_loss",
        """Add the symmetry loss (mirrored-state consistency) to actor and critic.

        Distinct from algo.config.use_symmetry, which mirrors and augments the
        replay data itself.""",
        "bool",
    ),
    Key(
        "algo.symmetry_actor_coef", "algo.config.symmetry_actor_coef",
        """Coefficient on the actor's symmetry loss.""",
        valid=(0.0, 100.0),
    ),
    Key(
        "algo.symmetry_critic_coef", "algo.config.symmetry_critic_coef",
        """Coefficient on the critic's symmetry loss.""",
        valid=(0.0, 100.0),
    ),
]

KEYS: dict[str, Key] = {k.name: k for k in _ALL}


def key(name: str) -> Key:
    """Look up a key by its short name, with a helpful error when it is unknown."""
    try:
        return KEYS[name]
    except KeyError:
        from difflib import get_close_matches

        near = get_close_matches(name, KEYS, n=3)
        raise KeyError(
            f"unknown key {name!r}"
            + (f"; did you mean {', '.join(near)}?" if near else "")
        ) from None
