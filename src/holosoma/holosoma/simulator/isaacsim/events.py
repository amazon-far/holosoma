"""Project-owned IsaacSim physics randomization writers (mass / CoM / inertia / material).

These deliberately do NOT route through IsaacLab's ``mdp.randomize_rigid_body_*`` events. IsaacLab's
samplers read a gaussian's two parameters as ``(mean, std)``, which conflicts with the project's
``[lo, hi]`` BOUNDS convention and silently produces a wrong distribution. By owning the write here
and sampling through the shared keyed :meth:`holosoma.utils.sampler.TermSampler.draw`, a config means exactly
the same thing on IsaacSim as on MuJoCo and IsaacGym — including gaussian and log_uniform.

The mass/CoM/inertia writers take a list of per-component :class:`DistributionSpec` (validated at the
term boundary) and write through ``asset.root_physx_view``; CoM/inertia reset to the asset's captured
defaults (``asset.data.default_*``) before applying, so repeated calls compose from the original.

The MATERIAL writer (:func:`randomize_rigid_body_material`) is different in kind because PhysX caps
the number of unique physics materials per scene (~64k). It has two orthogonal knobs. ``num_buckets``
controls VALUE QUANTIZATION: an int fills ``num_buckets`` materials with the distribution's QUANTILE
values (:func:`~holosoma.utils.sampler.quantiles`) and picks a bucket (matching IsaacLab's mechanism),
bounding unique materials at ``num_buckets`` — an ``n``-atom staircase matching the continuous
backends APPROXIMATELY (raise to tighten); ``None`` draws continuously. ``per_env`` controls
GRANULARITY: ``False`` (default) draws per shape (IsaacLab-native, for object/robot material terms),
``True`` draws one value per env broadcast across its shapes — matching the IsaacGym per-env friction
path and capping unique materials at ~``num_envs`` when continuous.
"""

from __future__ import annotations

from typing import Literal, Sequence

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

from holosoma.utils.sampler import DistributionSpec, DistributionLike, TermSampler, quantiles


def _resolve_body_ids(asset: RigidObject | Articulation, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Resolve the body indices targeted by ``asset_cfg`` (CPU long tensor)."""
    if asset_cfg.body_ids == slice(None):
        if asset_cfg.body_names is not None:
            body_ids, _ = asset.find_bodies(asset_cfg.body_names)
            return torch.tensor(body_ids, dtype=torch.int, device="cpu")
        return torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    return torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")


def _sample_columns(
    sampler: TermSampler,
    specs: Sequence[DistributionSpec],
    env_ids: torch.Tensor,
    body_ids: torch.Tensor,
    device: str | torch.device,
) -> torch.Tensor:
    """Keyed per-component draw -> ``[n_env, n_body, len(specs)]``.

    Each component (x/y/z for CoM, the 6 inertia components) gets its own int stream coord ``k`` so
    they stay on independent streams; ``body_ids`` are the stable per-body keys (passed as a ``[1,
    n_body]`` coord so they form the trailing dimension). Routes through the bound
    :class:`TermSampler`, so a per-component config (different bounds, or even a different
    ``kind``/explicit mean,std per component) is honored AND reproducible per (term, env, episode).
    """
    cols = [
        sampler.draw(spec, env_ids=env_ids, coords=(k, body_ids[None, :]), device=device)  # [n_env, n_body]
        for k, spec in enumerate(specs)
    ]
    return torch.stack(cols, dim=-1)  # [n_env, n_body, n_spec]


def randomize_rigid_body_mass(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    mass: DistributionLike,
    operation: Literal["add", "abs", "scale"],
    sampler: TermSampler,
    axis: int = 0,
    min_mass: float = 1e-6,
    recompute_inertia: bool = True,
):
    """Randomize rigid-body mass, drawing through the bound keyed sampler.

    ``mass`` is a config range value ([lo, hi] pair / spec dict / DistributionSpec). Composes the
    perturbation onto the CURRENT live mass
    (add/scale; see the comment below), clamps to ``min_mass``, and — when ``recompute_inertia`` —
    rescales each body's inertia by the SAME factor its mass changed by THIS call (``m_after /
    m_before``). Unlike IsaacLab's event it draws from the project ``TermSampler`` — so gaussian /
    log_uniform mean the same thing here as on MuJoCo/IsaacGym and the draw is reproducible per
    (term, env, episode).

    ``axis`` is the sampler STREAM coord (a plain int) so independent mass draws stay decorrelated — a
    link-mass SCALE and a base-mass ADD must use different stream coords, or a body in both lists (e.g.
    the torso) would get perfectly-correlated scale and add. (IsaacGym/MuJoCo decorrelate these the
    same way; this matches them.)

    ``recompute_inertia`` uses an explicit MULTIPLICATIVE ratio (not PhysX recompute-from-geometry),
    so it commutes with a separate inertia-shape DR term — order between the two does not matter,
    unlike a from-geometry recompute which would clobber a prior inertia scale.
    """
    spec = DistributionSpec.parse(mass)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    body_ids = _resolve_body_ids(asset, asset_cfg)

    # Compose onto the CURRENT live mass (add/scale), matching IsaacGym (``mass += offset`` /
    # ``mass *= scale``) and MuJoCo (``current + random`` / ``current * random``). DR mass terms are
    # setup-stage (run once), so onto-current == onto-default at startup; composing onto current is
    # what keeps "the same config means the same thing" across backends (a reset-to-default here made
    # a second call return ``draw - prior_draw`` instead of ``draw`` — a cross-backend divergence).
    masses = asset.root_physx_view.get_masses()
    mass_before = masses[env_ids[:, None], body_ids].clone()

    # ``axis`` int coord = stream; body_ids as a [1, n_body] coord -> [n_env, n_body] to match the index.
    bias = sampler.draw(spec, env_ids=env_ids, coords=(axis, body_ids[None, :]), device=masses.device)

    if operation == "add":
        masses[env_ids[:, None], body_ids] += bias
    elif operation == "scale":
        masses[env_ids[:, None], body_ids] *= bias
    elif operation == "abs":
        masses[env_ids[:, None], body_ids] = bias
    else:
        raise ValueError(f"Unknown operation: '{operation}'. Use 'add', 'abs' or 'scale'.")

    masses[env_ids[:, None], body_ids] = masses[env_ids[:, None], body_ids].clamp(min=min_mass)
    asset.root_physx_view.set_masses(masses, env_ids)

    if recompute_inertia:
        # Scale inertia by the per-body mass ratio THIS call produced (commutes with inertia DR).
        ratios = masses[env_ids[:, None], body_ids] / mass_before.clamp(min=min_mass)
        inertias = asset.root_physx_view.get_inertias()
        if isinstance(asset, Articulation):
            inertias[env_ids[:, None], body_ids] *= ratios[..., None]
        else:
            inertias[env_ids] *= ratios
        asset.root_physx_view.set_inertias(inertias, env_ids)


def randomize_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    specs: Sequence[DistributionSpec],
    operation: Literal["add", "abs", "scale"],
    sampler: TermSampler,
):
    """Randomize body center-of-mass by adding/scaling/setting a per-axis (x,y,z) sample.

    ``specs`` is one :class:`DistributionSpec` per axis. CoM is reset to ``env.default_coms`` before
    applying, so repeated calls compose from the original CoM. Draws through the bound keyed sampler,
    so gaussian (truncated to the band) and log_uniform behave identically to other backends and the
    draw is reproducible per (term, env, episode). Targets a single torso body (the per-axis sample is
    shared across any resolved bodies).
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    body_ids = _resolve_body_ids(asset, asset_cfg)
    # One (x,y,z) sample is drawn and applied to every resolved body; CoM DR is meant for a single
    # body (the torso). Guard so a multi-body asset_cfg fails loudly instead of silently sharing one
    # draw across bodies.
    assert body_ids.numel() == 1, (
        f"randomize_body_com expects a single body, got {body_ids.numel()} "
        f"(asset_cfg.body_names={asset_cfg.body_names!r}); CoM DR targets one body (the torso)."
    )

    coms = asset.root_physx_view.get_coms()
    coms[env_ids[:, None], body_ids] = env.default_coms[env_ids[:, None], body_ids].clone()

    # One (x,y,z) sample per env (first resolved body's row); CoM DR targets the torso, a single body.
    bias = _sample_columns(sampler, specs, env_ids, body_ids[:1], coms.device)[:, 0, :]
    env.base_com_bias[env_ids, :] = bias.to(env.base_com_bias.device)

    if operation == "add":
        coms[env_ids[:, None], body_ids, :3] += env.base_com_bias[env_ids[:, None], :]
    elif operation == "abs":
        coms[env_ids[:, None], body_ids, :3] = env.base_com_bias[env_ids[:, None], :]
    elif operation == "scale":
        coms[env_ids[:, None], body_ids, :3] *= env.base_com_bias[env_ids[:, None], :]
    else:
        raise ValueError(f"Unknown operation: '{operation}'. Use 'add', 'abs' or 'scale'.")

    asset.root_physx_view.set_coms(coms, env_ids)


def randomize_rigid_body_inertia(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    specs: Sequence[DistributionSpec],
    operation: Literal["add", "scale", "abs"],
    sampler: TermSampler,
):
    """Randomize the rigid-body inertia tensor by scaling/adding/setting per-component samples.

    ``specs`` holds 6 :class:`DistributionSpec` in the order [Ixx, Iyy, Izz, Ixy, Iyz, Ixz]. The 3x3
    inertia is symmetric; off-diagonals are mirrored. Draws through the bound keyed sampler, so a
    gaussian or log_uniform scale config means the same thing as on MuJoCo/IsaacGym and is
    reproducible per (term, env, episode).

    The inertia matrix is stored (PhysX) in the order Ixx, Iyx, Izx, Ixy, Iyy, Izy, Ixz, Iyz, Izz.
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    body_ids = _resolve_body_ids(asset, asset_cfg)

    inertias_original = asset.root_physx_view.get_inertias()  # (num_envs, 9) or (num_envs, num_bodies, 9)
    if inertias_original.ndim == 2:
        inertias = inertias_original.unsqueeze(1).clone()  # (num_envs, 1, 9)
    else:
        inertias = inertias_original.clone()  # (num_envs, num_bodies, 9)

    # Per-component sample, ordered [Ixx, Iyy, Izz, Ixy, Iyz, Ixz] -> (n_env, n_body, 6).
    inertia_random = _sample_columns(sampler, specs, env_ids, body_ids, inertias.device)

    # Size the bias to the env SUBSET (inertia_random has len(env_ids) rows), not the full view, so the
    # fill loop and the indexed apply below agree on dim 0 even when env_ids is a strict subset.
    inertias_bias = torch.ones(
        (inertia_random.shape[0], inertia_random.shape[1], 9),
        dtype=inertias.dtype,
        device=inertias.device,
    )
    diagonal_elements = [(0, 0), (1, 4), (2, 8)]  # Ixx, Iyy, Izz -> matrix slots 0,4,8
    off_diagonal_elements = [(3, 1, 3), (4, 7, 5), (5, 6, 2)]  # Ixy, Iyz, Ixz -> symmetric slot pairs

    for param_idx, matrix_idx in diagonal_elements:
        inertias_bias[:, :, matrix_idx] = inertia_random[:, :, param_idx]
    for param_idx, primary_idx, symmetric_idx in off_diagonal_elements:
        inertias_bias[:, :, primary_idx] = inertia_random[:, :, param_idx]
        inertias_bias[:, :, symmetric_idx] = inertias_bias[:, :, primary_idx]

    if operation == "add":
        inertias[env_ids[:, None], body_ids] += inertias_bias
    elif operation == "scale":
        inertias[env_ids[:, None], body_ids] *= inertias_bias
    elif operation == "abs":
        inertias[env_ids[:, None], body_ids] = inertias_bias
    else:
        raise ValueError(f"Unknown operation: '{operation}'. Use 'add', 'abs' or 'scale'.")

    if inertias_original.ndim == 2:
        inertias = inertias.squeeze(1)

    asset.root_physx_view.set_inertias(inertias, env_ids)


def randomize_rigid_body_material(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    static_friction: DistributionLike | None,
    dynamic_friction: DistributionLike | None,
    restitution: DistributionLike | None,
    num_buckets: int | None,
    sampler: TermSampler,
    make_consistent: bool = True,
    per_env: bool = False,
):
    """Randomize friction/restitution, bucketed or continuous, per shape or per env.

    Each channel is a config range value ([lo, hi] / spec dict / DistributionSpec), or ``None`` to leave
    that channel at its spawned value.

    ``num_buckets`` controls VALUE QUANTIZATION (how the PhysX per-scene material cap ~64k is respected):
    - an int (the default path): replicates IsaacLab's mechanism (build ``num_buckets`` materials once,
      assign a bucket), so total unique materials stay bounded at ``num_buckets`` regardless of
      env/shape count. The ONE change from IsaacLab: fill each channel with the QUANTILE values of its
      :class:`DistributionSpec` (via :func:`~holosoma.utils.sampler.quantiles`) instead of
      ``sample_uniform``, so a gaussian / log_uniform config matches the continuous backends' marginal
      APPROXIMATELY as an ``n``-atom staircase (raise ``num_buckets`` to tighten it).
    - ``None``: draw a CONTINUOUS value per channel (a true marginal).

    ``per_env`` controls GRANULARITY (independent of quantization):
    - ``False`` (default): draw independently PER SHAPE — the IsaacLab-native behavior, kept for the
      object/robot material terms.
    - ``True``: draw ONE value per env and broadcast it across that env's shapes, matching the IsaacGym
      per-env friction path (and MuJoCo's ``shared_across_entities``). The friction term uses this so
      one robot has a single contact-friction coefficient on every backend. Per-env also caps unique
      materials at ~``num_envs`` when continuous, keeping it under the PhysX limit.

    A channel spec of ``None`` is NOT randomized — that column keeps each shape's spawned value
    (so a config can randomize, e.g., friction only and leave restitution as authored). Returns
    early if all three are ``None``.

    ``make_consistent`` clamps dynamic friction <= static friction (PhysX expects this),
    matching IsaacLab's optional flag; enabled by default here since friction DR always wants it.

    Reproducibility: every draw (bucket fill shuffle, bucket pick, or continuous value) goes through
    the keyed ``sampler`` on distinct axes, so a seeded run produces the same realized material.
    """

    def _spec(leaf: DistributionLike | None) -> DistributionSpec | None:
        return None if leaf is None else DistributionSpec.parse(leaf)

    static_friction_spec = _spec(static_friction)
    dynamic_friction_spec = _spec(dynamic_friction)
    restitution_spec = _spec(restitution)
    channel_specs = (static_friction_spec, dynamic_friction_spec, restitution_spec)
    if all(spec is None for spec in channel_specs):
        return

    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    materials = asset.root_physx_view.get_material_properties()
    total_num_shapes = asset.root_physx_view.max_shapes
    samples = materials[env_ids].clone()  # (n_env, n_shapes, 3); start from current, overwrite chosen cols

    # ``per_env`` picks the shape coordinate: absent -> one draw per env, broadcast across shapes;
    # present (a [1, n_shapes] coord) -> an independent per-(env, shape) draw. The trailing shape
    # broadcast/assignment is identical either way.
    shape_ids = torch.arange(total_num_shapes)
    shape_coord: tuple = () if per_env else (shape_ids[None, :],)

    for k, spec in enumerate(channel_specs):
        if spec is None:
            continue
        if num_buckets is None:
            # Continuous: channel stream coords 0/1/2 keep the three columns decorrelated.
            drawn = sampler.draw(spec, env_ids=env_ids, coords=(k, *shape_coord))  # [n_env] or [n_env, n_shapes]
        else:
            # Bucketed: fill a per-channel bucket column with the spec's quantile values, shuffle it
            # with a KEYED permutation (distinct stream coord per channel so columns are not
            # rank-aligned — matches IsaacLab's independent per-column draw, reproducible per seed),
            # then pick a bucket via the keyed selection (stream coord 3, distinct from 0/1/2). Total
            # unique materials stay bounded at num_buckets.
            column = quantiles(spec, num_buckets, "cpu")[sampler.permute(num_buckets, (k,))]
            bucket_ids = sampler.draw_int(0, num_buckets - 1, env_ids=env_ids, coords=(3, *shape_coord))
            drawn = column[bucket_ids]
        samples[..., k] = drawn[:, None] if per_env else drawn  # broadcast per-env value over shapes

    if make_consistent and (static_friction_spec is not None or dynamic_friction_spec is not None):
        samples[..., 1] = torch.min(samples[..., 0], samples[..., 1])  # dynamic <= static (PhysX)

    materials[env_ids] = samples.to(materials.dtype)
    asset.root_physx_view.set_material_properties(materials, env_ids)
