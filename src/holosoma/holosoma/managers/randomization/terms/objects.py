"""Randomization terms for scene objects (free bodies).

These act on registered free bodies (``ObjectType.INDIVIDUAL``) spawned by the scene-asset
config. They default to "all registered free bodies", so a preset stays object-count-agnostic
and no-ops cleanly on a robot-only scene.

- ``randomize_object_rigid_body_mass_startup`` / ``..._material_startup`` /
  ``..._inertia_startup`` — startup physics DR on free bodies, cross-backend (IsaacSim /
  IsaacGym / MuJoCo), each branch addressing the object's own bodies/geoms (mirrors the robot
  mass/friction DR in ``locomotion.py``).
- ``jitter_object_pose_on_reset`` — opt-in reset-time pose jitter on free bodies.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
from loguru import logger

from holosoma.config_types.randomization import InertiaScale, MaterialRandomizationConfig
from holosoma.managers.randomization.exceptions import RandomizerNotSupportedError
from holosoma.managers.randomization.terms._shared import (
    _MATERIAL_NUM_BUCKETS,
    _ensure_env_ids_tensor,
)
from holosoma.simulator import mujoco_required_field
from holosoma.simulator.shared.field_decorators import MUJOCO_FIELD_ATTR
from holosoma.simulator.shared.object_registry import ObjectType
from holosoma.utils.rotations import quat_from_angle_axis, quat_mul
from holosoma.utils.sampler import DistributionLike, DistributionSpec, TermSampler, quantiles
from holosoma.utils.simulator_config import SimulatorType


def _resolve_object_names(env: Any, object_names: Sequence[str] | None) -> list[str]:
    """Free-body (INDIVIDUAL) actor names a per-object randomizer should target.

    Defaults to every registered free body so a preset stays object-count-agnostic
    (robot-only scenes resolve to an empty list -> the term no-ops). An explicit
    ``object_names`` narrows it to specific bodies. Static (SCENE) bodies are never
    included — they have no manipulable physics to randomize here.
    """
    if object_names is not None:
        return list(object_names)
    return env.simulator.object_registry.get_names_by_type(ObjectType.INDIVIDUAL)


def _mujoco_object_root_body(simulator: Any, name: str) -> str:
    """The compiled-model (prefixed) root-body name of registered object ``name``.

    Resolves from whichever scene-manager map holds the object: a STANDALONE rigid object is
    recorded in ``rigid_object_root_bodies`` (``{name}_{root}``), while a body that came from a
    1->N scene FILE is recorded in ``scene_file_bodies`` (the body is ``{file}_{body}``, which is
    the actor name itself). Raises KeyError if ``name`` is neither.
    """
    sm = simulator.scene_manager
    if name in sm.rigid_object_root_bodies:
        return sm.rigid_object_root_bodies[name]
    if name in getattr(sm, "scene_file_bodies", {}):
        return sm.scene_file_bodies[name][0]  # (prefixed_root_body, is_static)
    raise KeyError(f"Object '{name}' is not a registered standalone or scene-file body.")


def _mujoco_object_bodies(simulator: Any, name: str) -> list[tuple[int, str]]:
    """``(body_id, body_name)`` for every compiled-model body owned by object ``name``.

    Reads the CPU ``mujoco.MjModel`` (``simulator.backend.model``), which both MuJoCo
    backends expose, so it feeds the WarpBackend and ClassicBackend DR paths alike. Collects
    the object's root body plus its descendant subtree via ``body_parentid`` — correct for both
    a standalone object (single root, possibly with nested child bodies) and a scene-file body
    (its own root in the shared file tree, NOT its file-siblings). Resolving the subtree
    structurally (not by a ``{name}_`` name prefix) is what keeps scene-file siblings separate.
    """
    import mujoco

    model = simulator.backend.model
    root_body = _mujoco_object_root_body(simulator, name)
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_body)
    if root_id == -1:
        raise ValueError(f"Root body '{root_body}' for object '{name}' not found in compiled model.")

    # Subtree = root + every body whose parent chain reaches root (walk parentid up to root/world).
    subtree = {root_id}
    for bid in range(model.nbody):
        anc = bid
        while anc not in (0, root_id) and anc != model.body_parentid[anc]:
            anc = int(model.body_parentid[anc])
            if anc == root_id:
                subtree.add(bid)
                break
    return [(bid, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)) for bid in sorted(subtree)]


def _mujoco_object_body_names(simulator: Any, name: str) -> list[str]:
    """Prefixed compiled-model body names owned by object ``name``."""
    return [bname for _bid, bname in _mujoco_object_bodies(simulator, name)]


def _mujoco_object_body_ids(simulator: Any, name: str) -> list[int]:
    """Body IDs owned by object ``name`` in the compiled CPU model.

    URDF-imported descendant bodies are often unnamed (``mj_id2name`` returns ``None``), so the
    mass/inertia randomizers resolve bodies by ID — mirroring ``_mujoco_object_geom_ids`` — and
    pass them as pre-resolved ``entity_ids`` to ``randomize_field``, bypassing the by-name
    ``resolve_entity_ids`` path that would choke on a ``None`` name.
    """
    return [bid for bid, _bname in _mujoco_object_bodies(simulator, name)]


def _mujoco_object_geom_ids(simulator: Any, name: str) -> list[int]:
    """Geom IDs owned by object ``name``'s bodies in the compiled CPU model.

    URDF-imported geoms are usually unnamed, so resolve by body ownership (``geom_bodyid``)
    rather than by name — the MuJoCo randomizer accepts pre-resolved geom IDs directly.
    """
    model = simulator.backend.model
    body_ids = {bid for bid, _bname in _mujoco_object_bodies(simulator, name)}
    return [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in body_ids]


def _coerce_material_cfg(material: MaterialRandomizationConfig | dict) -> MaterialRandomizationConfig:
    """Accept either a :class:`MaterialRandomizationConfig` or an equivalent nested dict.

    Configs constructed in Python pass the dataclass; a config loaded from a plain dict (e.g. a
    serialized preset) passes the nested-dict form. Both resolve to the typed config so the term
    body only ever sees backend sub-blocks.
    """
    if isinstance(material, MaterialRandomizationConfig):
        return material
    return MaterialRandomizationConfig(**material)


def _draw_material_channel(
    sampler: TermSampler,
    spec: DistributionLike,
    env_ids: torch.Tensor,
    obj_ids: torch.Tensor,
    stream: int,
    num_buckets: int | None,
) -> torch.Tensor:
    """Draw a per-(env, object) material value, continuous or quantized onto ``num_buckets``.

    ``None``: one continuous keyed draw per (env, object) on stream ``stream`` -> [n_env, n_obj].
    An int: fill ``num_buckets`` buckets with the spec's quantile values (keyed permute so the
    staircase is reproducible-but-shuffled per seed), then pick a bucket per (env, object) via the
    keyed selection — same quantization the IsaacSim material writer / robot friction term use, so a
    bucketed object-material config reads consistently across backends.
    """
    if num_buckets is None:
        return sampler.draw(spec, env_ids=env_ids, coords=(stream, obj_ids))
    column = quantiles(DistributionSpec.parse(spec), num_buckets, "cpu")[sampler.permute(num_buckets, (stream,))]
    bucket_ids = sampler.draw_int(0, num_buckets - 1, env_ids=env_ids, coords=(stream, obj_ids))
    return column[bucket_ids]


@mujoco_required_field("geom_friction")
def randomize_object_rigid_body_material_startup(
    env,
    env_ids: Sequence[int] | torch.Tensor | None = None,
    *,
    sampler: TermSampler,
    material: MaterialRandomizationConfig | dict,
    object_names: Sequence[str] | None = None,
    num_buckets: int | None = _MATERIAL_NUM_BUCKETS,
    enabled: bool = True,
    **_,
) -> None:
    """Randomize free-body rigid-body material properties, with PER-BACKEND channel configs.

    ``material`` is a :class:`MaterialRandomizationConfig` (or an equivalent nested dict) whose
    backend sub-blocks name ONLY the channels that backend honors — no silent cross-backend drop:
    - **isaacgym** ``{friction, restitution}`` — single sliding friction + native restitution.
    - **isaacsim** ``{static_friction, dynamic_friction, restitution}`` — via a quantile bucket table
      (PhysX caps unique materials per scene).
    - **mujoco**   ``{sliding_friction, solref, solimp}`` — ``geom_friction`` tangential coefficient
      plus the per-component ``geom_solref``/``geom_solimp`` vectors through which MuJoCo expresses
      contact stiffness/restitution (no scalar ``[0,1]`` restitution channel; ``solref``/``solimp``
      are dicts of {component index -> range}).
    A backend whose sub-block is absent randomizes nothing on it; an absent channel within a present
    block is left at the spawned value. Each channel is a config range value — a ``[lo, hi]`` pair (uniform)
    or a ``{kind, low, high, mean, std}`` spec dict. Targets every registered free body by default;
    pass ``object_names`` to narrow it. No-ops on a robot-only scene.

    ``num_buckets`` controls VALUE QUANTIZATION uniformly across backends (mirrors the robot friction
    term). An int (default 64) quantizes every channel's draw onto that many stratified quantile
    values — required on IsaacSim to respect PhysX's per-scene material cap, and applied identically on
    IsaacGym/MuJoCo (which have no cap) so a bucketed config reads consistently everywhere. ``None``
    draws continuously; use it when the per-object material count is safely under the cap. Granularity
    is unchanged (per-object on IsaacGym, per-shape on IsaacSim, per-geom on MuJoCo) — objects keep
    their within-env material variety.
    """
    if not enabled:
        return
    material = _coerce_material_cfg(material)

    idx = _ensure_env_ids_tensor(env, env_ids)
    if idx.numel() == 0:
        return

    names = _resolve_object_names(env, object_names)
    if not names:  # robot-only scene -> nothing to randomize, not an error
        return

    simulator = env.simulator
    sim_type = simulator.get_simulator_type()

    if sim_type == SimulatorType.ISAACGYM:
        ig_cfg = material.isaacgym
        if ig_cfg is None:
            return
        gym = simulator.gym
        idx_cpu = idx.to(device="cpu", dtype=torch.long)
        # One keyed value per (env, object) per requested channel (distinct int stream coord),
        # written to the object's shapes — per-object like the mass/inertia branches, so objects
        # within one env get independent materials (matching MuJoCo/IsaacSim's within-env variety).
        # A None channel is left at the spawned value. friction -> single sliding coeff (stream 0);
        # restitution -> native per-shape restitution (stream 1). num_buckets quantizes each draw the
        # same way as the IsaacSim writer. -> [E, n_names].
        obj_ids = torch.arange(len(names))[None, :]
        fric = (
            _draw_material_channel(sampler, ig_cfg.friction, idx_cpu, obj_ids, 0, num_buckets)
            if ig_cfg.friction is not None
            else None
        )
        rest = (
            _draw_material_channel(sampler, ig_cfg.restitution, idx_cpu, obj_ids, 1, num_buckets)
            if ig_cfg.restitution is not None
            else None
        )
        if fric is None and rest is None:
            return
        for offset, env_id in enumerate(idx_cpu.tolist()):
            env_ptr = simulator.envs[env_id]
            for name_off, name in enumerate(names):
                actor = simulator.object_handles[name][env_id]
                shape_props = gym.get_actor_rigid_shape_properties(env_ptr, actor)
                for prop in shape_props:
                    if fric is not None:
                        prop.friction = float(fric[offset, name_off])
                    if rest is not None:
                        prop.restitution = float(rest[offset, name_off])
                gym.set_actor_rigid_shape_properties(env_ptr, actor, shape_props)
        return

    if sim_type == SimulatorType.ISAACSIM:
        isim_cfg = material.isaacsim
        if isim_cfg is None:
            return
        try:
            from isaaclab.managers import SceneEntityCfg
        except ImportError as exc:  # pragma: no cover - defensive
            raise RuntimeError("IsaacSim material randomization requires isaaclab.") from exc
        from holosoma.simulator.isaacsim.events import randomize_rigid_body_material

        env_ids_cpu = idx.to(device="cpu", dtype=torch.long)
        for name in names:
            asset_cfg = SceneEntityCfg(name, body_names=".*")
            asset_cfg.resolve(simulator.scene)
            randomize_rigid_body_material(
                simulator,
                env_ids_cpu,
                asset_cfg,
                static_friction=isim_cfg.static_friction,
                dynamic_friction=isim_cfg.dynamic_friction,
                restitution=isim_cfg.restitution,
                num_buckets=num_buckets,
                sampler=sampler,
            )
        return

    if sim_type == SimulatorType.MUJOCO:
        mj_cfg = material.mujoco
        if mj_cfg is None or (mj_cfg.sliding_friction is None and mj_cfg.solref is None and mj_cfg.solimp is None):
            return
        # No material cap here, but num_buckets still applies for cross-backend value consistency
        # (None = continuous per-geom, the true marginal). Each present channel writes its own geom
        # field; a distinct axis_base keeps the sampler streams independent.
        from holosoma.simulator.mujoco.backends.randomization import randomize_field

        # (geom field, ranges-by-axis, axis_base) for each requested channel. friction -> geom_friction
        # axis 0 (sliding); solref/solimp -> their per-component dicts as given.
        writes = []
        if mj_cfg.sliding_friction is not None:
            writes.append(("geom_friction", {0: mj_cfg.sliding_friction}, 0))
        if mj_cfg.solref is not None:
            # axis_base offset so streams don't collide
            writes.append(("geom_solref", {int(k): v for k, v in mj_cfg.solref.items()}, 10))
        if mj_cfg.solimp is not None:
            writes.append(("geom_solimp", {int(k): v for k, v in mj_cfg.solimp.items()}, 20))
        for name in names:
            geom_ids = _mujoco_object_geom_ids(simulator, name)
            if not geom_ids:
                logger.warning(f"Object '{name}' has no geoms; material randomization skipped.")
                continue
            geom_ids_t = torch.tensor(geom_ids, device=simulator.sim_device, dtype=torch.long)
            for field, ranges, axis_base in writes:
                randomize_field(
                    simulator,
                    field=field,
                    ranges=ranges,
                    sampler=sampler,
                    env_ids=idx,
                    entity_ids=geom_ids_t,
                    operation="abs",
                    axis_base=axis_base,
                    num_buckets=num_buckets,
                )
        return

    raise RandomizerNotSupportedError(
        f"randomize_object_rigid_body_material_startup does not support {type(simulator).__name__}."
    )


@mujoco_required_field("body_mass")
def randomize_object_rigid_body_mass_startup(
    env,
    env_ids: Sequence[int] | torch.Tensor | None = None,
    *,
    sampler: TermSampler,
    mass_distribution_params: DistributionLike,
    object_names: Sequence[str] | None = None,
    enabled: bool = True,
    recompute_inertia: bool = True,
    **_,
) -> None:
    """Randomize free-body rigid-body mass (additive offset), one offset per object.

    Cross-backend (IsaacSim / IsaacGym / MuJoCo Warp GPU + Classic CPU); the offset is sampled
    from ``mass_distribution_params`` and ADDED, identically on every backend. Scene objects are
    single rigid bodies (no articulated-object support), so "per (env, object)" and "per body" are
    the same draw. Targets every registered free body by default; pass ``object_names`` to narrow
    it. No-ops on a robot-only scene. MuJoCo Classic CPU is single-env.

    ``recompute_inertia`` (default True): when True, each body's inertia is rescaled by the SAME
    factor its mass changed by (``m_after / m_before``), identically on all backends (explicit
    multiplicative ratio — NOT a from-geometry recompute — so it commutes with a separate inertia
    DR term, order-independent). When False, mass changes and inertia is left untouched.

    ``mass_distribution_params`` is a config range value — a ``[lo, hi]`` pair (uniform) or a
    ``{kind, low, high, mean, std}`` spec dict (gaussian / log_uniform) — honored identically on ALL
    backends (IsaacGym per-actor loop, MuJoCo ``randomize_field``, project-owned IsaacSim
    ``randomize_rigid_body_mass``), all via the shared keyed
    :meth:`holosoma.utils.sampler.TermSampler.draw`. gaussian is a truncated normal on ``[lo, hi]``;
    log_uniform requires positive bounds.
    """
    if not enabled:
        return

    idx = _ensure_env_ids_tensor(env, env_ids)
    if idx.numel() == 0:
        return

    names = _resolve_object_names(env, object_names)
    if not names:  # robot-only scene -> nothing to randomize, not an error
        return

    simulator = env.simulator
    sim_type = simulator.get_simulator_type()

    if sim_type == SimulatorType.ISAACGYM:
        gym = simulator.gym
        # One keyed offset per (env, object), added to all of that object's bodies — keyed by the
        # object's index among ``names`` (stable) so the draw is reproducible per (term, env, episode).
        # Object indices as a [1, n_names] coord -> [n_env, n_names].
        obj_ids = torch.arange(len(names))[None, :]
        offsets = sampler.draw(mass_distribution_params, env_ids=idx, coords=(obj_ids,), device="cpu")
        for env_off, env_id in enumerate(idx.tolist()):
            env_ptr = simulator.envs[env_id]
            for name_off, name in enumerate(names):
                actor = simulator.object_handles[name][env_id]
                body_props = gym.get_actor_rigid_body_properties(env_ptr, actor)
                delta = float(offsets[env_off, name_off])  # add operation, matches the other backends
                for prop in body_props:
                    m_before = prop.mass
                    # Clamp like the IsaacSim helper (min_mass=1e-6): a signed band on a light body
                    # must not produce a non-positive mass (which would also flip the inertia sign).
                    prop.mass = max(m_before + delta, 1e-6)
                    if recompute_inertia and m_before > 0.0:
                        # Explicit per-body mass-ratio scale (NOT recomputeInertia=True, which would
                        # recompute from geometry — a different result that also clobbers a prior
                        # inertia-DR scale). Multiplicative => commutes with the inertia term.
                        r = prop.mass / m_before
                        inertia = prop.inertia  # gymapi.Mat33, symmetric
                        inertia.x.x *= r
                        inertia.y.y *= r
                        inertia.z.z *= r
                        inertia.x.y *= r
                        inertia.y.x *= r
                        inertia.y.z *= r
                        inertia.z.y *= r
                        inertia.x.z *= r
                        inertia.z.x *= r
                # recomputeInertia=False: we set inertia explicitly above (or leave it untouched).
                gym.set_actor_rigid_body_properties(env_ptr, actor, body_props, recomputeInertia=False)
        return

    if sim_type == SimulatorType.ISAACSIM:
        from isaaclab.managers import SceneEntityCfg

        from holosoma.simulator.isaacsim.events import randomize_rigid_body_mass

        env_ids_cpu = idx.to(device="cpu", dtype=torch.long)

        for name in names:
            asset_cfg = SceneEntityCfg(name, body_names=".*")
            asset_cfg.resolve(simulator.scene)
            randomize_rigid_body_mass(
                simulator,
                env_ids_cpu,
                asset_cfg,
                mass_distribution_params,
                operation="add",
                sampler=sampler,
                recompute_inertia=recompute_inertia,
            )
        return

    if sim_type == SimulatorType.MUJOCO:
        from holosoma.simulator.mujoco.backends.randomization import (
            _field_view,
            randomize_field,
            scale_inertia_by_mass_ratio,
        )

        for name in names:
            body_ids = _mujoco_object_body_ids(simulator, name)
            body_ids_t = torch.tensor(body_ids, device=simulator.sim_device, dtype=torch.long)
            # Snapshot mass BEFORE so the inertia rescale uses the exact ratio this write produced
            # (matches IsaacGym/IsaacSim's explicit m_after/m_before scale).
            mass_before = _field_view(simulator, "body_mass")[:, body_ids_t].clone() if recompute_inertia else None
            randomize_field(
                simulator,
                field=getattr(randomize_object_rigid_body_mass_startup, MUJOCO_FIELD_ATTR),
                ranges=mass_distribution_params,
                sampler=sampler,
                env_ids=idx,
                # Resolve by ID (URDF descendant bodies are often unnamed) — mirrors the geom path.
                entity_ids=body_ids_t,
                operation="add",
                # One offset per object, shared by all its bodies (matches the IsaacGym branch,
                # which adds one delta to every body of the actor).
                shared_across_entities=True,
            )
            # Clamp like the Isaac paths (min_mass=1e-6): a signed band on a light body must not
            # leave a non-positive mass in the model.
            mass_view = _field_view(simulator, "body_mass")
            mass_view[:, body_ids_t] = mass_view[:, body_ids_t].clamp(min=1e-6)
            if recompute_inertia:
                scale_inertia_by_mass_ratio(simulator, body_ids_t, mass_before)
        return

    raise RandomizerNotSupportedError(
        f"randomize_object_rigid_body_mass_startup does not support {type(simulator).__name__}."
    )


_INERTIA_ORDER = ["Ixx", "Iyy", "Izz", "Ixy", "Iyz", "Ixz"]

_IDENTITY_SCALE: DistributionLike = (1.0, 1.0)


def _coerce_inertia_params(
    params: InertiaScale | dict[str, DistributionLike],
) -> dict[str, DistributionLike]:
    """Normalize the inertia-scale config to a full ``{Ixx..Ixz}`` dict of leaves.

    Accepts an :class:`InertiaScale` or an equivalent (possibly partial) dict — a serialized config
    arrives as a dict. Absent components default to the identity scale ``[1.0, 1.0]`` (no-op), so a
    caller names only the components it randomizes.
    """
    if isinstance(params, InertiaScale):
        return {k: getattr(params, k) for k in _INERTIA_ORDER}
    unknown = set(params) - set(_INERTIA_ORDER)
    if unknown:
        raise ValueError(f"unknown inertia component(s) {sorted(unknown)}; expected {_INERTIA_ORDER}.")
    return {k: params.get(k, _IDENTITY_SCALE) for k in _INERTIA_ORDER}


def _is_identity_scale(leaf: DistributionLike) -> bool:
    """Whether a config range value is the identity scale [1.0, 1.0] (no-op off-diagonal)."""
    try:
        spec = DistributionSpec.parse(leaf)
    except (ValueError, TypeError):
        return False
    return spec.low == 1.0 and spec.high == 1.0 and spec.mean is None and spec.std is None


def _warn_off_diagonal_inertia_ignored(backend: str, inertia_params: dict[str, DistributionLike]) -> None:
    """One-time warning when a backend that can only scale the diagonal moments is asked to
    scale an off-diagonal (Ixy/Iyz/Ixz) by a non-identity factor. ``[1.0, 1.0]`` is the
    identity scale (no-op), matching how the shipped config leaves off-diagonals untouched."""
    offending = [k for k in ("Ixy", "Iyz", "Ixz") if not _is_identity_scale(inertia_params.get(k, (1.0, 1.0)))]
    if offending:
        logger.warning(
            f"{backend} object inertia DR scales only the diagonal principal moments "
            f"(Ixx/Iyy/Izz); off-diagonal scale(s) {offending} are ignored."
        )


@mujoco_required_field("body_inertia")
def randomize_object_rigid_body_inertia_startup(
    env,
    env_ids: Sequence[int] | torch.Tensor | None = None,
    *,
    sampler: TermSampler,
    inertia_distribution_params_dict: InertiaScale | dict[str, DistributionLike],
    object_names: Sequence[str] | None = None,
    enabled: bool = True,
    **_,
) -> None:
    """Randomize free-body rigid-body inertia (scale the inertia tensor).

    Cross-backend. ``inertia_distribution_params_dict`` is an :class:`InertiaScale` (or an equivalent,
    possibly partial, dict) of per-component SCALE ranges keyed [Ixx, Iyy, Izz, Ixy, Iyz, Ixz]; each
    value is a ``[lo, hi]`` pair (uniform) or a spec dict, and an unnamed component defaults to the
    identity scale [1.0, 1.0]:
    - **IsaacSim / IsaacGym** scale the full symmetric 3x3 (all 6 components).
    - **MuJoCo** (Warp GPU or Classic CPU) stores inertia as the 3 DIAGONAL principal moments
      in a principal frame (``body_inertia`` is [nbody, 3]; off-diagonals are 0 by construction),
      so it scales only Ixx/Iyy/Izz and logs a one-time warning if a non-identity off-diagonal
      range is requested. The shipped config leaves off-diagonals at [1.0, 1.0] (identity), so
      MuJoCo reproduces it faithfully.

    Order-independent w.r.t. the mass DR term: both scale inertia MULTIPLICATIVELY (this term by the
    configured factors; mass DR by its mass ratio when ``recompute_inertia=True``), and the IsaacGym
    write uses ``recomputeInertia=False``, so neither clobbers the other regardless of which runs
    first. Targets every registered free body by default; pass ``object_names`` to narrow. No-ops on a
    robot-only scene. MuJoCo classic-CPU is single-env (one sample); MuJoCo classic via the shared
    ``randomize_field`` path.

    Each scale range is honored identically on ALL backends — IsaacGym per-component loop, MuJoCo
    ``randomize_field``, and the project-owned IsaacSim ``randomize_rigid_body_inertia`` — all via the
    shared keyed :meth:`holosoma.utils.sampler.TermSampler.draw`. A gaussian spec is a truncated normal
    on each ``[lo, hi]``; log_uniform requires positive bounds (satisfied by identity-and-up scales).
    """
    if not enabled:
        return

    idx = _ensure_env_ids_tensor(env, env_ids)
    if idx.numel() == 0:
        return

    names = _resolve_object_names(env, object_names)
    if not names:  # robot-only scene -> nothing to randomize, not an error
        return

    inertia_distribution_params_dict = _coerce_inertia_params(inertia_distribution_params_dict)
    simulator = env.simulator
    sim_type = simulator.get_simulator_type()

    if sim_type == SimulatorType.ISAACGYM:
        # Full 6-component scale via the per-actor RigidBodyProperties Mat33 inertia tensor, with
        # recomputeInertia=False (we set the tensor explicitly). The mass DR term also uses
        # recomputeInertia=False + a multiplicative mass-ratio scale, so the two inertia writes
        # commute — no clobber, no ordering requirement.
        gym = simulator.gym
        # Pre-draw one scale per (env, object) for each of the 6 components, keyed by object index
        # (stable, a [1, n_names] coord) and a distinct int stream coord per component, then apply to
        # every body of that object. -> each entry [n_env, n_names].
        obj_ids = torch.arange(len(names))[None, :]
        comp_draws = {
            k: sampler.draw(inertia_distribution_params_dict[k], env_ids=idx, coords=(axis, obj_ids), device="cpu")
            for axis, k in enumerate(_INERTIA_ORDER)
        }
        for env_off, env_id in enumerate(idx.tolist()):
            env_ptr = simulator.envs[env_id]
            for name_off, name in enumerate(names):
                actor = simulator.object_handles[name][env_id]
                body_props = gym.get_actor_rigid_body_properties(env_ptr, actor)
                fxx, fyy, fzz = (float(comp_draws[k][env_off, name_off]) for k in ("Ixx", "Iyy", "Izz"))
                fxy, fyz, fxz = (float(comp_draws[k][env_off, name_off]) for k in ("Ixy", "Iyz", "Ixz"))
                for prop in body_props:
                    inertia = prop.inertia  # gymapi.Mat33, symmetric
                    inertia.x.x *= fxx
                    inertia.y.y *= fyy
                    inertia.z.z *= fzz
                    inertia.x.y *= fxy
                    inertia.y.x *= fxy
                    inertia.y.z *= fyz
                    inertia.z.y *= fyz
                    inertia.x.z *= fxz
                    inertia.z.x *= fxz
                gym.set_actor_rigid_body_properties(env_ptr, actor, body_props, recomputeInertia=False)
        return

    if sim_type == SimulatorType.ISAACSIM:
        from isaaclab.managers import SceneEntityCfg

        from holosoma.simulator.isaacsim.events import randomize_rigid_body_inertia

        env_ids_cpu = idx.to(device="cpu", dtype=torch.long)
        # One spec per component in [Ixx, Iyy, Izz, Ixy, Iyz, Ixz] order.
        specs = [DistributionSpec.parse(inertia_distribution_params_dict[key]) for key in _INERTIA_ORDER]
        for name in names:
            asset_cfg = SceneEntityCfg(name, body_names=".*")
            asset_cfg.resolve(simulator.scene)
            randomize_rigid_body_inertia(simulator, env_ids_cpu, asset_cfg, specs, operation="scale", sampler=sampler)
        return

    if sim_type == SimulatorType.MUJOCO:
        # MuJoCo body_inertia is [nbody, 3] diagonal principal moments — scale Ixx/Iyy/Izz only.
        from holosoma.simulator.mujoco.backends.randomization import randomize_field

        _warn_off_diagonal_inertia_ignored("MuJoCo", inertia_distribution_params_dict)
        ranges = {axis: inertia_distribution_params_dict[k] for axis, k in enumerate(("Ixx", "Iyy", "Izz"))}
        for name in names:
            body_ids = _mujoco_object_body_ids(simulator, name)
            randomize_field(
                simulator,
                field=getattr(randomize_object_rigid_body_inertia_startup, MUJOCO_FIELD_ATTR),
                ranges=ranges,  # axis 0/1/2 = Ixx/Iyy/Izz principal moments
                sampler=sampler,
                env_ids=idx,
                # Resolve by ID (URDF descendant bodies are often unnamed) — mirrors the geom path.
                entity_ids=torch.tensor(body_ids, device=simulator.sim_device, dtype=torch.long),
                operation="scale",
                # One scale per object, shared by all its bodies (matches the IsaacGym branch).
                shared_across_entities=True,
            )
        return

    raise RandomizerNotSupportedError(
        f"randomize_object_rigid_body_inertia_startup does not support {type(simulator).__name__}."
    )


# Freejoint DOF layout: a free body's 6 velocity DOFs are [vx, vy, vz, wx, wy, wz], so the
# linear-damping DOFs are the first 3 and the angular-damping DOFs the last 3, counted from the
# freejoint's qvel base (object_addrs[name]["qvel_addr"]).
_FREEJOINT_LINEAR_DOFS = (0, 1, 2)
_FREEJOINT_ANGULAR_DOFS = (3, 4, 5)

# Body-damping randomization. Damping is expressed differently per engine but means the same
# physically (a drag that bleeds linear/angular velocity), and is randomized as a single linear +
# single angular scalar per body on every backend (PhysX's model), kept aligned:
# - MuJoCo: ``dof_damping`` on the freejoint, one shared value across its 3 linear / 3 angular DOFs.
# - IsaacSim: the USD ``PhysxRigidBodyAPI.linearDamping``/``angularDamping`` scalar, writable at
#   runtime (the live PhysX sim picks up the edit on the next step).
# - IsaacGym: NO runtime path. Source-verified the gym exposes damping only on ``AssetOptions`` at
#   asset import; neither ``RigidBodyProperties`` (mass/inertia/com only) nor ``RigidShapeProperties``
#   (friction/restitution/compliance/offsets only) carries damping, and the only runtime damping
#   setter, ``set_actor_dof_properties``, is for articulation joints, not free-body drag. So
#   IsaacGym raises (set damping via PhysXPhysicsConfig at spawn instead).
_DAMPING_ISAACGYM_MSG = (
    "Runtime body-damping randomization is unsupported on IsaacGym. "
    "Set damping via PhysXPhysicsConfig at spawn instead."
)


def _isaacsim_randomize_body_damping(
    simulator, names, idx, *, axis: str, leaf: DistributionLike, sampler: TermSampler
) -> None:
    """Set free-body linear/angular damping per (object, env) on IsaacSim via the live USD schema.

    Writes ``PhysxRigidBodyAPI.{axis}Damping`` on each object's per-env rigid-body prim to a keyed
    sample (one value per (env, object), keyed by the object index). The running PhysX sim reflects
    the edit on the next step. ``axis`` is "linear" or "angular". Unlike the mass/com/inertia paths,
    this writes the USD attr directly, drawing through the bound ``sampler`` AND reproducibly.

    The stage is traversed ONCE into a list of API-bearing prims (rather than re-walking the full
    stage for every (name, env) pair); the per-(name, env) work then scans only that list.
    """
    import omni.usd
    from pxr import PhysxSchema

    stage = omni.usd.get_context().get_stage()
    env_id_list = idx.to(device="cpu", dtype=torch.long).tolist()
    # One keyed value per (env, object), keyed by object index (stable across the run); object indices
    # as a [1, n_names] coord -> [n_env, n_names].
    obj_ids = torch.arange(len(names))[None, :]
    damping = sampler.draw(leaf, env_ids=idx, coords=(obj_ids,), device="cpu")

    # One pass over the stage: collect (path_str, prim) for every PhysxRigidBodyAPI prim.
    rigid_prims = [
        (str(prim.GetPath()), prim) for prim in stage.Traverse() if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI)
    ]

    for name_off, name in enumerate(names):
        for env_off, e in enumerate(env_id_list):
            prefix = f"/World/envs/env_{e}/{name}"
            value = float(damping[env_off, name_off])
            for path_str, prim in rigid_prims:
                # Boundary-anchored match: the prim is this body or a child of it. A bare
                # startswith would over-match siblings (env_1 -> env_10, box -> box2).
                if path_str != prefix and not path_str.startswith(prefix + "/"):
                    continue
                api = PhysxSchema.PhysxRigidBodyAPI(prim)
                attr = api.GetLinearDampingAttr() if axis == "linear" else api.GetAngularDampingAttr()
                attr.Set(value)


def _randomize_object_freejoint_damping(
    env,
    env_ids,
    sampler: TermSampler,
    *,
    dofs: tuple[int, ...],
    axis: str,
    damping_range: DistributionLike,
    object_names: Sequence[str] | None,
    field_attr: str,
) -> None:
    """Shared core for linear/angular body-damping DR (MuJoCo + IsaacSim; IsaacGym unsupported).

    ``axis`` is "linear" or "angular"; ``dofs`` are the freejoint-relative DOF offsets MuJoCo
    writes (linear 0:3 / angular 3:6). One value is sampled per env and applied to the whole axis
    group — matching PhysX's single linear/angular damping scalar, so the concept is identical on
    every supported backend (MuJoCo shares it across the 3 freejoint DOFs; IsaacSim sets the one
    PhysxRigidBodyAPI scalar).

    ``damping_range`` is a config range value ([lo, hi] pair or spec dict), honored on BOTH MuJoCo (via
    ``randomize_field``) and IsaacSim (the live-USD writer). IsaacGym has no runtime damping path and
    raises regardless.
    """
    idx = _ensure_env_ids_tensor(env, env_ids)
    if idx.numel() == 0:
        return
    names = _resolve_object_names(env, object_names)
    if not names:  # robot-only scene -> nothing to randomize, not an error
        return

    simulator = env.simulator
    sim_type = simulator.get_simulator_type()

    if sim_type == SimulatorType.ISAACGYM:
        raise RandomizerNotSupportedError(_DAMPING_ISAACGYM_MSG)

    if sim_type == SimulatorType.ISAACSIM:
        _isaacsim_randomize_body_damping(simulator, names, idx, axis=axis, leaf=damping_range, sampler=sampler)
        return

    if sim_type == SimulatorType.MUJOCO:
        from holosoma.simulator.mujoco.backends.randomization import randomize_field

        for name in names:
            qvel_base = simulator.object_addrs[name]["qvel_addr"]  # freejoint's 6-DOF velocity base
            dof_ids = torch.tensor([qvel_base + d for d in dofs], device=simulator.sim_device, dtype=torch.long)
            randomize_field(
                simulator,
                field=field_attr,
                ranges=damping_range,
                sampler=sampler,
                env_ids=idx,
                entity_ids=dof_ids,
                operation="abs",
                shared_across_entities=True,
            )
        return

    raise RandomizerNotSupportedError(
        f"{field_attr} damping randomization does not support {type(simulator).__name__}."
    )


@mujoco_required_field("dof_damping")
def randomize_object_linear_damping_startup(
    env,
    env_ids: Sequence[int] | torch.Tensor | None = None,
    *,
    sampler: TermSampler,
    damping_range: DistributionLike,
    object_names: Sequence[str] | None = None,
    enabled: bool = True,
    **_,
) -> None:
    """Randomize free-body LINEAR (translational) damping — drag on linear velocity.

    MuJoCo (Classic + Warp, via freejoint ``dof_damping``) and IsaacSim (via the live USD
    ``PhysxRigidBodyAPI.linearDamping``); IsaacGym is unsupported and raises (it accepts body
    damping only at asset import — see the error). One value per env is applied as the body's
    single linear-damping scalar (on MuJoCo, shared across the 3 translational freejoint DOFs), so
    the concept matches PhysX on every backend. Targets every registered free body by default. On
    the WarpBackend keep ``damping_range`` <= ~0.05 (freejoint dof_damping diverges above that at
    the default dt); the ClassicBackend and IsaacSim are stable at any value.

    ``damping_range`` is a config range value — a ``[lo, hi]`` pair (uniform) or a spec dict — honored on
    both MuJoCo and IsaacSim; gaussian truncates to ``[lo, hi]`` and log_uniform requires positive
    bounds.
    """
    if not enabled:
        return
    _randomize_object_freejoint_damping(
        env,
        env_ids,
        sampler,
        dofs=_FREEJOINT_LINEAR_DOFS,
        axis="linear",
        damping_range=damping_range,
        object_names=object_names,
        field_attr=getattr(randomize_object_linear_damping_startup, MUJOCO_FIELD_ATTR),
    )


@mujoco_required_field("dof_damping")
def randomize_object_angular_damping_startup(
    env,
    env_ids: Sequence[int] | torch.Tensor | None = None,
    *,
    sampler: TermSampler,
    damping_range: DistributionLike,
    object_names: Sequence[str] | None = None,
    enabled: bool = True,
    **_,
) -> None:
    """Randomize free-body ANGULAR (rotational) damping — drag on angular velocity.

    MuJoCo (Classic + Warp) and IsaacSim (via the live USD ``PhysxRigidBodyAPI.angularDamping``);
    IsaacGym unsupported (see :func:`randomize_object_linear_damping_startup`). One value per env is
    applied as the body's single angular-damping scalar (on MuJoCo, shared across the 3 rotational
    freejoint DOFs), matching PhysX on every backend. WarpBackend stable band <= ~0.05.

    ``damping_range`` is a config range value — a ``[lo, hi]`` pair (uniform) or a spec dict — honored on
    both MuJoCo and IsaacSim; gaussian truncates to ``[lo, hi]`` and log_uniform requires positive
    bounds.
    """
    if not enabled:
        return
    _randomize_object_freejoint_damping(
        env,
        env_ids,
        sampler,
        dofs=_FREEJOINT_ANGULAR_DOFS,
        axis="angular",
        damping_range=damping_range,
        object_names=object_names,
        field_attr=getattr(randomize_object_angular_damping_startup, MUJOCO_FIELD_ATTR),
    )


def _jitter_spec(range_cfg: float | DistributionLike) -> DistributionSpec | None:
    """Build a :class:`DistributionSpec` for a pose-jitter component, or ``None`` if it's a no-op.

    A scalar ``r`` is the symmetric uniform band ``[-r, r]``; ``0.0`` (or ``[0, 0]``) is a no-op
    -> ``None``. A ``[lo, hi]`` pair (uniform) or spec dict is taken as-is. A non-uniform jitter
    (e.g. gaussian) must be given as a spec dict, not a bare float.
    """
    if isinstance(range_cfg, (int, float)):
        r = float(range_cfg)
        if r == 0.0:
            return None
        return DistributionSpec(kind="uniform", low=-r, high=r)
    spec = DistributionSpec.parse(range_cfg)
    # Treat an explicit zero-width band as a no-op, regardless of kind: a [c, c] band is a point
    # mass at c, and only c == 0 leaves the baseline pose unchanged. (An explicit (mean, std)
    # gaussian has low/high == None, so the `== 0.0` test is False and it is never a no-op here.)
    if spec.low == 0.0 and spec.high == 0.0:
        return None
    return spec


def jitter_object_pose_on_reset(
    env,
    env_ids: Sequence[int] | torch.Tensor | None = None,
    *,
    sampler: TermSampler,
    xy_range: float | DistributionLike = 0.0,
    yaw_range: float | DistributionLike = 0.0,
    object_names: Sequence[str] | None = None,
    enabled: bool = True,
    **_,
) -> None:
    """Jitter free-body XY position and yaw around the baseline reset pose, per env.

    The random yaw is COMPOSED onto the initial orientation (which may be non-identity).
    Only the pose is jittered; each body keeps its configured initial velocity
    (``RigidObjectConfig.linear_velocity``/``angular_velocity``, zero by default), matching
    the baseline reset (``BaseTask._reset_objects_callback``) this term overwrites. Static
    (SCENE) bodies are never jittered. No-ops on a robot-only scene or when both ranges are
    zero.

    Parameters
    ----------
    xy_range : float | [lo, hi] | spec dict
        XY offset (metres). A FLOAT ``r`` is the symmetric uniform half-width ``[-r, r]``; a
        ``[lo, hi]`` pair (uniform) or ``{kind, low, high, mean, std}`` spec dict is taken directly.
    yaw_range : float | [lo, hi] | spec dict
        Yaw delta (radians); same forms as ``xy_range``.
    object_names : Sequence[str] | None
        Free bodies to jitter. ``None`` (default) jitters every registered free body.
    """
    if not enabled:
        return

    xy_spec = _jitter_spec(xy_range)
    yaw_spec = _jitter_spec(yaw_range)
    if xy_spec is None and yaw_spec is None:
        return

    idx = _ensure_env_ids_tensor(env, env_ids)
    if idx.numel() == 0:
        return

    # Only INDIVIDUAL (free) bodies are jitterable; never write a static body's pose. An explicitly
    # requested static name is a config mistake — warn instead of silently skipping it.
    static = set(env.simulator.object_registry.get_names_by_type(ObjectType.SCENE))
    resolved = _resolve_object_names(env, object_names)
    dropped_static = [n for n in resolved if n in static]
    if dropped_static and object_names is not None:
        logger.warning(f"pose jitter skips static (SCENE) bodies: {dropped_static}")
    names = [n for n in resolved if n not in static]
    if not names:
        return

    # Baseline reset pose + velocity (env origins already applied), object-major / env-minor:
    # pose [len(names)*len(env_ids), 7] = [x,y,z,qx,qy,qz,qw]; velocity [..., 6] = [vx,vy,vz,wx,wy,wz].
    base = env.simulator.get_actor_initial_poses(names, idx)
    base_vel = env.simulator.get_actor_initial_velocities(names, idx)
    n_rows = base.shape[0]

    states = torch.zeros(n_rows, 13, device=env.device)
    states[:, :7] = base
    states[:, 7:] = base_vel  # keep the configured initial velocity; jitter only the pose

    # Draws are keyed per (env, object) and laid out object-major to match the row order
    # [name0_env0, name0_env1, ..., name1_env0, ...]: draw [n_env, n_names] then transpose -> flatten.
    n_names = len(names)
    obj_ids = torch.arange(n_names)[None, :]  # [1, n_names] coord -> trailing object dimension

    def _draw_object_major(spec: DistributionSpec, axis: int) -> torch.Tensor:
        # one value per (env, object) -> object-major column vector of length n_rows. ``axis`` is the
        # int STREAM coord (x/y/yaw kept independent); obj ids land on the trailing dimension.
        per_env_obj = sampler.draw(spec, env_ids=idx, coords=(axis, obj_ids), device=env.device)  # [n_env, n_names]
        return per_env_obj.transpose(0, 1).reshape(-1)  # [n_names, n_env] -> n_rows

    if xy_spec is not None:
        states[:, 0] += _draw_object_major(xy_spec, axis=0)
        states[:, 1] += _draw_object_major(xy_spec, axis=1)

    if yaw_spec is not None:
        yaw = _draw_object_major(yaw_spec, axis=2)
        z_axis = torch.zeros(n_rows, 3, device=env.device)
        z_axis[:, 2] = 1.0
        yaw_quat = quat_from_angle_axis(yaw, z_axis, w_last=True)  # xyzw
        # Compose the yaw delta onto the baseline orientation (world-frame yaw): q = dq * q0.
        states[:, 3:7] = quat_mul(yaw_quat, base[:, 3:7].clone(), w_last=True)

    env.simulator.set_actor_states(names, idx, states)
