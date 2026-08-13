"""Shared cross-backend domain-randomization (DR) checks.

The DR terms themselves are backend-agnostic: every term is called identically on every
backend (the per-backend dispatch lives inside the term). Only *reading back* the mutated
physics-model field differs — MuJoCo classic exposes float64 numpy on ``backend.model``,
MuJoCo Warp a per-world ``warp_model_bridge`` view, IsaacGym/IsaacSim per-actor property
structs. So these helpers take a small ``ModelReader`` that abstracts the four read-backs and
run the FULL matrix of DR terms against ONE already-built real environment.

This is deliberately "one sim, many asserts": building a full env (real action_manager +
randomization_manager, no shims) is the expensive step, so each backend's thin test file builds
the env once and calls ``run_robot_dr`` + ``run_object_dr`` here. Manager/action-state and
actor-state read-backs (PD/RFI scales, push velocity, object pose) are backend-agnostic and read
directly off the live env, so they live here too.

All ranges below are deliberately exaggerated and disjoint from the defaults so a no-op or a
wrong-backend write is unambiguous.
"""

from __future__ import annotations

import dataclasses
from contextlib import contextmanager
from typing import Protocol, cast

import torch

from holosoma.managers.randomization.exceptions import RandomizerNotSupportedError
from holosoma.managers.randomization.terms import locomotion as L
from holosoma.managers.randomization.terms import objects as O
from holosoma.simulator.shared.object_registry import ObjectType
from holosoma.utils.sampler import STAGE_RESET, TermSampler
from holosoma.utils.simulator_config import SimulatorType


def _sampler(env, term: str = "dr_matrix_test") -> TermSampler:
    """A bound TermSampler for calling DR terms directly in tests.

    These checks assert physical effects/bands, not seed reproducibility; TermSampler requires a seed
    (no global-RNG fallback), so fall back to a fixed test seed when the env has none.
    """
    base_seed = getattr(env, "dr_base_seed", None) or 0
    episode = getattr(env, "dr_episode_count", None)
    return TermSampler.bind(base_seed, term, STAGE_RESET, episode)


@contextmanager
def build_full_env(simulator_cfg, *, num_envs: int, device: str, scene_cfg=None, seed: int = 0):
    """Yield a real, fully-managed locomotion env for DR testing — NO shims.

    Swaps the ``simulator`` (and optionally ``scene``) of the ``g1_29dof`` experiment preset and
    constructs the env through the production path (``get_class(env_class)(get_tyro_env_config)``
    under ``training_context``), so it has a real action_manager (with the ``joint_control`` term)
    and randomization_manager (push/actuator states) wired exactly as training does. The
    ``g1_29dof`` randomization preset already runs the setup DR terms during construction; this
    yields the built env (NO ``reset_all``) so the per-term checks can re-run each term with known
    ranges and read the result straight off the model.

    ``reset_all`` is deliberately NOT called: it steps physics, which triggers the locomotion
    terrain height-scan (warp ray-cast) — unrelated to DR and version-fragile — while the DR
    checks need only the constructed env + managers, not a physics step.

    A context manager (not a plain return) so ``training_context`` stays open for the whole test —
    its ``__exit__`` calls ``close_simulation_app``, which would tear down the IsaacSim app early.

    Parameters
    ----------
    simulator_cfg : SimulatorConfig
        One of ``config_values.simulator.{mujoco, mjwarp, isaacgym, isaacsim}``.
    num_envs : int
        1 for the MuJoCo ClassicBackend (single-env only); >1 for the GPU backends.
    device : str
        ``"cpu"`` for classic, ``"cuda:0"`` for the GPU backends.
    scene_cfg : SceneConfig | None
        Scene preset; defaults to ``free_and_static`` so object DR has a free body to target.
    """
    # Imported lazily: importing config_values / train_agent pulls in the simulator stack, which
    # the per-backend launcher (isaacgym/isaacsim) must initialize before torch is imported.
    from holosoma.config_types.env import get_tyro_env_config
    from holosoma.config_values import experiment
    from holosoma.train_agent import training_context
    from holosoma.utils.common import seeding
    from holosoma.utils.helpers import get_class

    # Default object scene is a test preset (free + static bodies), injected here by object (no tyro).
    from tests.simulators._scene_presets import free_and_static

    seeding(seed)
    base = experiment.g1_29dof
    cfg = dataclasses.replace(
        base,
        simulator=simulator_cfg,
        scene=free_and_static if scene_cfg is None else scene_cfg,
        training=dataclasses.replace(
            base.training, num_envs=num_envs, headless=True, seed=seed, torch_deterministic=False
        ),
    )
    with training_context(cfg):
        env = get_class(cfg.env_class)(get_tyro_env_config(cfg), device=device)
        yield env


class ModelReader(Protocol):
    """Backend-specific read-back of physics-model fields for a single env (env 0).

    Implementations return the *current* live value the term should have mutated. For
    multi-env backends, read env 0 (the terms are applied to all env_ids passed in).

    Capability flags tell the matrix which fields a backend can honestly read back from the
    LIVE solver state (vs a field the backend's API does not reflect or does not model). When a
    flag is False the corresponding band assertion is SKIPPED (not faked-green) and the success
    banner does not claim that field for the backend.
    """

    # Whether ``body_com`` returns the live integrated CoM after a CoM write (see _IsaacGymReader).
    COM_READBACK_TRUSTED: bool = True
    # Whether the backend models restitution and exposes it (only IsaacSim, see _IsaacSimReader).
    RESTITUTION_READBACK_TRUSTED: bool = False

    def body_mass(self, body_name: str) -> float: ...
    def geom_friction_for_body(self, body_name: str) -> float: ...
    def body_inertia(self, body_name: str) -> tuple[float, float, float]: ...
    def body_com(self, body_name: str) -> tuple[float, float, float]: ...
    # Optional, IsaacSim-only (gated by RESTITUTION_READBACK_TRUSTED):
    def restitution_for_body(self, body_name: str) -> float: ...
    def dynamic_friction_for_body(self, body_name: str) -> float: ...


# --- exaggerated, disjoint-from-default ranges so any change is unambiguous -------------------
ROBOT_BASE_MASS_ADD = (2.0, 3.0)
ROBOT_FRICTION = (0.5, 0.7)
# Distinct band for the distribution-DR friction re-draw: run_robot_dr already sets friction in
# ROBOT_FRICTION, and the keyed sampler draws the SAME underlying uniform for an identical (term,
# stage, band), so a uniform and a gaussian over the SAME band would land ~1e-3 apart and the
# "changed" assertion would flake. A disjoint band makes the change unambiguous regardless of
# sampler correlation (the point of that check is "gaussian friction lands in band and took effect").
GAUSS_FRICTION = (0.85, 0.95)
# Distinct band again for the IsaacSim robot-MATERIAL term (separate from randomize_friction_startup,
# which already set GAUSS_FRICTION) so its "landed in band" check is unambiguous.
ROBOT_MATERIAL_FRICTION = (0.30, 0.40)
ROBOT_LINK_MASS_SCALE = (2.0, 3.0)
ROBOT_BASE_COM_X = (0.05, 0.06)
PD_SCALE = (2.0, 3.0)
RFI_SCALE = (2.0, 3.0)
ACTION_DELAY = 2
DOF_POS_BIAS = (0.5, 0.6)
DOF_VEL = (0.1, 0.2)
OBJ_MASS_ADD = (5.0, 6.0)
OBJ_FRICTION = (0.2, 0.3)
# Restitution band disjoint from friction so a mis-wired slot is unambiguous; IsaacSim only
# (the only backend that models restitution — IsaacGym/MuJoCo ignore it, gated below).
OBJ_RESTITUTION = (0.7, 0.8)
OBJ_INERTIA_IXX_SCALE = (2.0, 3.0)


def _env_ids(env) -> torch.Tensor:
    return torch.arange(env.num_envs, device=env.device, dtype=torch.long)


def run_robot_dr(env, reader: ModelReader) -> list[str]:
    """Run every robot DR term against ``env`` and assert each took effect.

    Covers: link/base mass, friction, base CoM (physics-model fields via ``reader``); PD-gain,
    RFI-limit, action-delay, DOF-state randomizers (manager/action state, read directly).

    Returns the list of check labels that were SKIPPED because the backend cannot honestly read
    the live field back (e.g. ``["base CoM"]`` on IsaacGym). The caller uses this to scope the
    success banner so it never claims a field it did not actually verify.
    """
    skipped: list[str] = []
    idx = _env_ids(env)
    torso = env.robot_config.torso_name
    link_names = list(env.robot_config.randomize_link_body_names or [])
    assert link_names, "robot config declares no randomize_link_body_names"

    # --- base mass (additive offset); isolate from link scaling --------------------------------
    before = reader.body_mass(torso)
    L.randomize_mass_startup(
        env,
        idx,
        sampler=_sampler(env),
        enable_link_mass=False,
        enable_base_mass=True,
        added_mass_range=ROBOT_BASE_MASS_ADD,
    )
    after = reader.body_mass(torso)
    assert ROBOT_BASE_MASS_ADD[0] - 1e-2 <= after - before <= ROBOT_BASE_MASS_ADD[1] + 1e-2, (
        f"robot base mass offset out of band: {before} -> {after}"
    )

    # --- link mass (scale); exaggerated factor so the change is unambiguous ---------------------
    link0 = link_names[0]
    before = reader.body_mass(link0)
    L.randomize_mass_startup(
        env,
        idx,
        sampler=_sampler(env),
        enable_link_mass=True,
        link_mass_range=ROBOT_LINK_MASS_SCALE,
        enable_base_mass=False,
    )
    after = reader.body_mass(link0)
    assert ROBOT_LINK_MASS_SCALE[0] - 1e-3 <= after / before <= ROBOT_LINK_MASS_SCALE[1] + 1e-3, (
        f"robot link mass scale out of band: {after / before}"
    )

    # --- friction (abs) -------------------------------------------------------------------------
    before = reader.geom_friction_for_body(torso)
    L.randomize_friction_startup(env, idx, sampler=_sampler(env), friction_range=ROBOT_FRICTION)
    after = reader.geom_friction_for_body(torso)
    assert ROBOT_FRICTION[0] - 1e-3 <= after <= ROBOT_FRICTION[1] + 1e-3, f"robot friction out of band: {after}"
    assert abs(after - before) > 1e-3, "robot friction unchanged"

    # --- base CoM (add to body_ipos) ------------------------------------------------------------
    # The setup preset already applied a base-CoM bias, and the backends differ in how a second
    # call composes: MuJoCo ADDs to body_ipos (accumulate), IsaacGym overwrites from the original.
    # A zero-range call first neutralizes any prior bias to a clean baseline so the before/after
    # delta equals exactly this call's offset on either semantics.
    #
    # The band assertion reads the LIVE CoM via reader.body_com and is gated by the reader's
    # COM_READBACK_TRUSTED flag. On IsaacGym that flag is False: its get_actor_rigid_body_properties
    # does not reliably reflect a CoM write back (undocumented/version-fragile) and the state tensor
    # carries no body-frame CoM, so there is NO trustworthy live read-back in this harness. We still
    # RUN the term (so a hard write-path failure raises) but SKIP the band assertion and record it,
    # rather than re-adding the term's own recorded bias (which would be a green tautology).
    L.randomize_base_com_startup(
        env, idx, sampler=_sampler(env), base_com_range={"x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 0.0]}
    )
    com_trusted = getattr(reader, "COM_READBACK_TRUSTED", True)
    com_before: tuple[float, float, float] | None = reader.body_com(torso) if com_trusted else None
    L.randomize_base_com_startup(
        env, idx, sampler=_sampler(env), base_com_range={"x": list(ROBOT_BASE_COM_X), "y": [0.0, 0.0], "z": [0.0, 0.0]}
    )
    if com_before is not None:
        com_after = reader.body_com(torso)
        assert ROBOT_BASE_COM_X[0] - 1e-3 <= com_after[0] - com_before[0] <= ROBOT_BASE_COM_X[1] + 1e-3, (
            f"robot base CoM x offset out of band: {com_after[0] - com_before[0]}"
        )
        assert abs(com_after[1] - com_before[1]) < 1e-3 and abs(com_after[2] - com_before[2]) < 1e-3, (
            "CoM y/z wrongly shifted"
        )
    else:
        skipped.append("base CoM (no trustworthy live read-back on this backend)")

    # --- PD gains (actuator-state randomizer; read via the joint action term) -------------------
    jt = env.action_manager.get_term("joint_control")
    kp_before, _ = jt.get_pd_scale_tensors()
    kp_before = kp_before.clone()
    L.randomize_pd_gains(env, idx, sampler=_sampler(env), kp_range=PD_SCALE, kd_range=PD_SCALE, enabled=True)
    kp_after, _ = jt.get_pd_scale_tensors()
    assert torch.all(kp_after >= PD_SCALE[0] - 1e-3) and torch.all(kp_after <= PD_SCALE[1] + 1e-3), (
        f"PD kp scale out of band: {kp_after.min()}..{kp_after.max()}"
    )
    assert (kp_after - kp_before).abs().max() > 1e-3, "PD kp scale unchanged"

    # --- RFI limits -----------------------------------------------------------------------------
    L.randomize_rfi_limits(env, idx, sampler=_sampler(env), rfi_lim_range=RFI_SCALE, enabled=True)
    rfi_after = jt.get_rfi_scale_tensor()
    assert torch.all(rfi_after >= RFI_SCALE[0] - 1e-3) and torch.all(rfi_after <= RFI_SCALE[1] + 1e-3), (
        f"RFI scale out of band: {rfi_after.min()}..{rfi_after.max()}"
    )

    # --- action delay (integer index in [lo, hi]) -----------------------------------------------
    env._ctrl_delay_step_range = [ACTION_DELAY, ACTION_DELAY]
    env._randomize_ctrl_delay = True
    L.randomize_action_delay(
        env, idx, sampler=_sampler(env), ctrl_delay_step_range=[ACTION_DELAY, ACTION_DELAY], enabled=True
    )
    assert int(env.action_delay_idx[0]) == ACTION_DELAY, f"action delay idx not set: {env.action_delay_idx[0]}"

    # --- DOF state (pos bias + velocity; scale held at 1.0 so the bias band is exact) -----------
    L.randomize_dof_state(
        env,
        idx,
        sampler=_sampler(env),
        joint_pos_scale_range=[1.0, 1.0],
        joint_pos_bias_range=list(DOF_POS_BIAS),
        joint_vel_range=list(DOF_VEL),
        randomize_dof_pos_bias=True,
    )
    dpos = env.simulator.dof_pos[0] - env.default_dof_pos[0]
    assert float(dpos.min()) >= DOF_POS_BIAS[0] - 1e-3 and float(dpos.max()) <= DOF_POS_BIAS[1] + 1e-3, (
        f"DOF pos bias out of band: {float(dpos.min())}..{float(dpos.max())}"
    )
    dvel = env.simulator.dof_vel[0]
    assert float(dvel.min()) >= DOF_VEL[0] - 1e-3 and float(dvel.max()) <= DOF_VEL[1] + 1e-3, (
        f"DOF vel out of band: {float(dvel.min())}..{float(dvel.max())}"
    )

    # --- DOF-pos bias setup (default pose shifted within band; distinct from randomize_dof_state) ---
    base = env.default_dof_pos_base[0].clone()
    L.setup_dof_pos_bias(env, sampler=_sampler(env), dof_pos_bias_range=list(DOF_POS_BIAS), enabled=True)
    bias = env.default_dof_pos[0] - base
    assert float(bias.min()) >= DOF_POS_BIAS[0] - 1e-3 and float(bias.max()) <= DOF_POS_BIAS[1] + 1e-3, (
        f"setup_dof_pos_bias out of band: {float(bias.min())}..{float(bias.max())}"
    )

    # --- torque-RFI setup (configures the joint action term; read the flag/limit back) ----------
    L.setup_torque_rfi(env, sampler=_sampler(env), enabled=True, rfi_lim=0.123)
    assert jt._randomize_torque_rfi is True, "setup_torque_rfi did not enable torque RFI on the action term"
    assert abs(jt._rfi_lim - 0.123) < 1e-6, f"setup_torque_rfi did not set rfi_lim: {jt._rfi_lim}"

    return skipped


def run_push_dr(env) -> None:
    """Run the push DR terms and assert a push imparts root velocity (backend-agnostic read).

    Covers BOTH the reset-stage ``randomize_push_schedule`` (+ a direct ``_push_robots``) AND the
    step-stage ``apply_pushes`` (the schedule -> due-envs -> push flow that runs every step in
    training), so the registered push term — not just the manager helper — is exercised.
    """
    idx = _env_ids(env)
    L.randomize_push_schedule(
        env, idx, sampler=_sampler(env), push_interval_s=[1.0, 1.0], max_push_vel=[5.0, 5.0], enabled=True
    )
    vel_before = env.simulator.robot_root_states[:, 7:9].clone()
    env._push_robots(idx)
    vel_after = env.simulator.robot_root_states[:, 7:9]
    assert (vel_after - vel_before).abs().max() > 1e-3, "push did not change root velocity"

    # apply_pushes (step term): drive the real schedule -> due -> push flow. apply_pushes' own
    # configure() sets a [1s,1s] interval (=> interval_steps = round(1/dt)); set every env's counter to
    # exactly that so due_envs() fires this step, making the push deterministic (not schedule-flaky).
    state = env.randomization_manager.get_state("push_randomizer_state")
    state.configure(enabled=True, push_interval_s=[1.0, 1.0], max_push_vel=[5.0, 5.0])
    state.push_robot_counter[idx] = (state.push_interval_s[idx] / env.dt).to(torch.int)
    vel_before = env.simulator.robot_root_states[:, 7:9].clone()
    L.apply_pushes(env, sampler=_sampler(env), enabled=True, push_interval_s=[1.0, 1.0], max_push_vel=[5.0, 5.0])
    vel_after = env.simulator.robot_root_states[:, 7:9]
    assert (vel_after - vel_before).abs().max() > 1e-3, "apply_pushes did not change root velocity"


def run_object_dr(env, reader: ModelReader) -> None:
    """Run every object DR term against the first free body and assert each took effect.

    Covers object mass, material (friction), inertia (physics-model fields via ``reader``) and
    pose jitter (actor state, read directly). Asserts the scene actually has a free object.
    """
    idx = _env_ids(env)
    free = env.simulator.object_registry.get_names_by_type(ObjectType.INDIVIDUAL)
    assert free, "scene has no free (INDIVIDUAL) objects to randomize"
    # Pass the backend-neutral registry object name (e.g. "free0"); each backend's reader resolves
    # it to its own handle (MuJoCo body "free0_baseLink", gym actor, isaaclab RigidObject) by
    # prefix-match. Resolving via a MuJoCo-only helper here would import mujoco on the Isaac envs.
    name = free[0]

    # --- object mass (add) ----------------------------------------------------------------------
    before = reader.body_mass(name)
    O.randomize_object_rigid_body_mass_startup(
        env, idx, sampler=_sampler(env), mass_distribution_params=OBJ_MASS_ADD, object_names=[name]
    )
    after = reader.body_mass(name)
    assert OBJ_MASS_ADD[0] - 1e-2 <= after - before <= OBJ_MASS_ADD[1] + 1e-2, (
        f"object mass offset out of band: {before} -> {after}"
    )

    # --- object material (friction abs; + restitution on IsaacSim only) -------------------------
    # Restitution is modelled only by IsaacSim (PhysX material slot 2); IsaacGym and MuJoCo ignore
    # restitution_range entirely (they log a warning and set sliding friction only). So we pass a
    # DISJOINT restitution band and assert it only when the reader declares restitution trusted —
    # never where the backend would ignore the write (which would make the assert vacuous/false).
    restitution_trusted = getattr(reader, "RESTITUTION_READBACK_TRUSTED", False)
    restitution_range = OBJ_RESTITUTION if restitution_trusted else (0.0, 0.0)
    before = reader.geom_friction_for_body(name)
    rest_before = reader.restitution_for_body(name) if restitution_trusted else None
    dyn_before = reader.dynamic_friction_for_body(name) if restitution_trusted else None
    # Per-backend material config: each backend gets the channels it honors. friction band OBJ_FRICTION
    # on all; restitution only where the reader trusts the read-back (IsaacSim) — elsewhere [0,0] / absent.
    O.randomize_object_rigid_body_material_startup(
        env,
        idx,
        sampler=_sampler(env),
        object_names=[name],
        material={
            "isaacgym": {"friction": OBJ_FRICTION, "restitution": restitution_range},
            "isaacsim": {
                "static_friction": OBJ_FRICTION,
                "dynamic_friction": OBJ_FRICTION,
                "restitution": restitution_range,
            },
            "mujoco": {"sliding_friction": OBJ_FRICTION},
        },
    )
    after = reader.geom_friction_for_body(name)
    assert OBJ_FRICTION[0] - 1e-3 <= after <= OBJ_FRICTION[1] + 1e-3, f"object friction out of band: {after}"
    assert abs(after - before) > 1e-3, "object friction unchanged"
    if restitution_trusted and rest_before is not None and dyn_before is not None:
        # Live PhysX restitution (material slot 2) must land in the disjoint band AND have changed.
        rest_after = reader.restitution_for_body(name)
        assert OBJ_RESTITUTION[0] - 1e-3 <= rest_after <= OBJ_RESTITUTION[1] + 1e-3, (
            f"object restitution out of band: {rest_after}"
        )
        assert abs(rest_after - rest_before) > 1e-3, "object restitution unchanged"
        # Dynamic friction (slot 1) was driven by the same disjoint OBJ_FRICTION band; verify live.
        dyn_after = reader.dynamic_friction_for_body(name)
        assert OBJ_FRICTION[0] - 1e-3 <= dyn_after <= OBJ_FRICTION[1] + 1e-3, (
            f"object dynamic friction out of band: {dyn_after}"
        )
        assert abs(dyn_after - dyn_before) > 1e-3, "object dynamic friction unchanged"

    # --- object inertia (scale Ixx only; Iyy/Izz + off-diagonals identity) ----------------------
    inertia_before = reader.body_inertia(name)
    O.randomize_object_rigid_body_inertia_startup(
        env,
        idx,
        sampler=_sampler(env),
        object_names=[name],
        # Only Ixx is randomized; the other five components default to the identity scale [1.0, 1.0].
        inertia_distribution_params_dict={"Ixx": OBJ_INERTIA_IXX_SCALE},
    )
    inertia_after = reader.body_inertia(name)
    # reader.body_inertia returns the live diagonal (Ixx, Iyy, Izz) = [0], [1], [2].
    assert OBJ_INERTIA_IXX_SCALE[0] - 1e-3 <= inertia_after[0] / inertia_before[0] <= OBJ_INERTIA_IXX_SCALE[1] + 1e-3, (
        f"object Ixx scale out of band: {inertia_after[0] / inertia_before[0]}"
    )
    # Iyy/Izz were held at identity (1.0,1.0), so their live ratios must stay ~1.0 — a guard that
    # the term scaled ONLY Ixx and did not bleed into the other principal moments (mirrors the
    # CoM y/z guard above). Tolerance matches the Ixx band slack.
    iyy_ratio = inertia_after[1] / inertia_before[1]
    izz_ratio = inertia_after[2] / inertia_before[2]
    assert abs(iyy_ratio - 1.0) < 1e-3 and abs(izz_ratio - 1.0) < 1e-3, (
        f"object Iyy/Izz wrongly scaled: {iyy_ratio}, {izz_ratio}"
    )

    # --- object pose jitter (actor state; XY only, read directly) -------------------------------
    st_before = env.simulator.get_actor_states([name], idx).clone()
    O.jitter_object_pose_on_reset(env, idx, sampler=_sampler(env), xy_range=0.1, yaw_range=0.0, object_names=[name])
    st_after = env.simulator.get_actor_states([name], idx)
    assert (st_after[:, :2] - st_before[:, :2]).abs().max() > 1e-4, "object pose was not jittered"


# --- distribution coverage: bands chosen so a gaussian truncates inside and a log_uniform stays
#     positive; the signed range is invalid for log_uniform (must raise on every backend). --------
GAUSS_BASE_MASS_ADD = {"kind": "gaussian", "low": 2.0, "high": 3.0}
LOGU_LINK_MASS_SCALE = {"kind": "log_uniform", "low": 2.0, "high": 3.0}  # strictly positive -> valid
SIGNED_LOGU_MASS_ADD = {"kind": "log_uniform", "low": -1.0, "high": 3.0}  # signed -> invalid (must raise)
GAUSS_OBJ_MASS_ADD = {"kind": "gaussian", "low": 5.0, "high": 6.0}
EXPLICIT_GAUSS_BASE_MASS = {"kind": "gaussian", "low": 2.0, "high": 3.0, "mean": 2.5, "std": 0.1}


def _assert_damping_term_unsupported(env, idx, term, object_name: str) -> None:
    try:
        term(env, idx, sampler=_sampler(env), damping_range=(0.0, 0.05), object_names=[object_name])
    except RandomizerNotSupportedError:
        pass
    else:
        raise AssertionError(f"IsaacGym {term.__name__} should raise RandomizerNotSupportedError")


def run_distribution_dr(env, reader: ModelReader) -> list[str]:
    """Assert a non-uniform distribution leaf lands in the same band/marginal on every backend.

    Covers, on the robot torso/link masses (always readable) and — where the scene has a free body —
    object mass: (1) gaussian lands in-band on EVERY backend, including IsaacSim; (2) explicit (mean,
    std) gaussian lands in-band; (3) log_uniform on a positive scale range works; (4) log_uniform on a
    SIGNED range RAISES (a clean ValueError, not a silent NaN) on every backend; (5) material/friction
    RAISES on any non-uniform distribution.

    Returns the list of check labels SKIPPED on this backend (e.g. object checks on a robot-only
    scene) so the caller's banner never claims an unverified field.
    """
    skipped: list[str] = []
    idx = _env_ids(env)
    torso = env.robot_config.torso_name

    # (1) gaussian base mass (add): truncated to the band on every backend incl. IsaacSim.
    before = reader.body_mass(torso)
    L.randomize_mass_startup(
        env,
        idx,
        sampler=_sampler(env),
        enable_link_mass=False,
        enable_base_mass=True,
        added_mass_range=GAUSS_BASE_MASS_ADD,
    )
    after = reader.body_mass(torso)
    lo, hi = cast("float", GAUSS_BASE_MASS_ADD["low"]), cast("float", GAUSS_BASE_MASS_ADD["high"])
    assert lo - 1e-2 <= after - before <= hi + 1e-2, f"gaussian base mass offset out of band: {before} -> {after}"

    # (2) explicit (mean, std) gaussian: also truncated to the band.
    before = reader.body_mass(torso)
    L.randomize_mass_startup(
        env,
        idx,
        sampler=_sampler(env),
        enable_link_mass=False,
        enable_base_mass=True,
        added_mass_range=EXPLICIT_GAUSS_BASE_MASS,
    )
    after = reader.body_mass(torso)
    assert 2.0 - 1e-2 <= after - before <= 3.0 + 1e-2, (
        f"explicit-(mean,std) gaussian base mass out of band: {before} -> {after}"
    )

    # (3) log_uniform link mass (scale, strictly positive): lands in the positive band.
    link_names = list(env.robot_config.randomize_link_body_names or [])
    assert link_names, "robot config declares no randomize_link_body_names"
    link0 = link_names[0]
    before = reader.body_mass(link0)
    L.randomize_mass_startup(
        env,
        idx,
        sampler=_sampler(env),
        enable_link_mass=True,
        link_mass_range=LOGU_LINK_MASS_SCALE,
        enable_base_mass=False,
    )
    after = reader.body_mass(link0)
    lo, hi = cast("float", LOGU_LINK_MASS_SCALE["low"]), cast("float", LOGU_LINK_MASS_SCALE["high"])
    assert lo - 1e-3 <= after / before <= hi + 1e-3, f"log_uniform link mass scale out of band: {after / before}"

    # (4) log_uniform on a SIGNED range must RAISE (clean ValueError) on every backend — no NaN.
    try:
        L.randomize_mass_startup(
            env,
            idx,
            sampler=_sampler(env),
            enable_link_mass=False,
            enable_base_mass=True,
            added_mass_range=SIGNED_LOGU_MASS_ADD,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("log_uniform on a signed range should raise ValueError, but did not")

    # (5) gaussian FRICTION lands in-band on every backend (continuous on IsaacGym/MuJoCo;
    #     64-bucket quantile staircase on IsaacSim — both marginal-matched to the same band). Uses
    #     GAUSS_FRICTION (disjoint from the ROBOT_FRICTION band run_robot_dr already applied) so the
    #     "took effect" check is unambiguous even though the keyed sampler reuses the same uniform.
    before = reader.geom_friction_for_body(torso)
    gauss_friction = {"kind": "gaussian", "low": GAUSS_FRICTION[0], "high": GAUSS_FRICTION[1]}
    L.randomize_friction_startup(env, idx, sampler=_sampler(env), friction_range=gauss_friction)
    after = reader.geom_friction_for_body(torso)
    assert GAUSS_FRICTION[0] - 1e-3 <= after <= GAUSS_FRICTION[1] + 1e-3, f"gaussian friction out of band: {after}"
    assert abs(after - before) > 1e-3, "gaussian friction unchanged"

    # log_uniform friction on a SIGNED range must still RAISE (positivity guard) on every backend.
    try:
        L.randomize_friction_startup(
            env, idx, sampler=_sampler(env), friction_range={"kind": "log_uniform", "low": -0.5, "high": 0.5}
        )
    except ValueError:
        pass
    else:
        raise AssertionError("log_uniform friction on a signed range should raise ValueError, but did not")

    # A typo'd distribution kind must still raise (fail-fast) at spec construction.
    try:
        L.randomize_friction_startup(
            env, idx, sampler=_sampler(env), friction_range={"kind": "gaussain", "low": 0.5, "high": 0.7}
        )
    except ValueError:
        pass
    else:
        raise AssertionError("unknown distribution on friction should raise ValueError, but did not")

    # Object-mass gaussian (only if the scene has a free body) — proves the object path honors it too.
    free = env.simulator.object_registry.get_names_by_type(ObjectType.INDIVIDUAL)
    if free:
        name = free[0]
        before = reader.body_mass(name)
        O.randomize_object_rigid_body_mass_startup(
            env, idx, sampler=_sampler(env), mass_distribution_params=GAUSS_OBJ_MASS_ADD, object_names=[name]
        )
        after = reader.body_mass(name)
        lo, hi = cast("float", GAUSS_OBJ_MASS_ADD["low"]), cast("float", GAUSS_OBJ_MASS_ADD["high"])
        assert lo - 1e-2 <= after - before <= hi + 1e-2, f"gaussian object mass offset out of band: {before} -> {after}"
    else:
        skipped.append("object-mass gaussian (robot-only scene)")

    # IsaacGym has no runtime body-damping path; assert BOTH damping terms raise rather than silently
    # no-op (covers linear AND angular — only linear is exercised behaviorally elsewhere).
    if free and env.simulator.get_simulator_type() == SimulatorType.ISAACGYM:
        for term in (O.randomize_object_linear_damping_startup, O.randomize_object_angular_damping_startup):
            _assert_damping_term_unsupported(env, idx, term, free[0])

    # Robot material term (IsaacSim-only; distinct from randomize_friction_startup). Friction lands in
    # band per the reader; restitution only asserted where the reader trusts it (gated above).
    if env.simulator.get_simulator_type() == SimulatorType.ISAACSIM:
        before = reader.geom_friction_for_body(torso)
        L.randomize_robot_rigid_body_material_startup(
            env,
            idx,
            sampler=_sampler(env),
            static_friction_range=ROBOT_MATERIAL_FRICTION,
            dynamic_friction_range=ROBOT_MATERIAL_FRICTION,
            restitution_range=(0.0, 0.0),
        )
        after = reader.geom_friction_for_body(torso)
        assert ROBOT_MATERIAL_FRICTION[0] - 1e-3 <= after <= ROBOT_MATERIAL_FRICTION[1] + 1e-3, (
            f"robot material friction out of band: {after}"
        )
        assert abs(after - before) > 1e-3, "robot material friction unchanged"

    return skipped
