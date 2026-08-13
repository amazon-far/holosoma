"""MuJoCo domain randomization utilities (both backends).

Adapted from mjlab (Apache 2.0): https://github.com/mujocolab/mjlab/blob/main/src/mjlab/sim/randomization.py
See THIRD_PARTY_LICENSES for full license text.

Provides ``randomize_field`` for randomizing model fields, which works on BOTH MuJoCo
backends: the WarpBackend (GPU, vectorized across the per-world-expanded model via the
WarpBridge) and the single-env ClassicBackend (CPU, writing the one ``mujoco.MjModel`` field
in place). ``expand_model_fields`` and its Warp kernel are WarpBackend-only — they tile a
single-world model across environments and are a no-op for the single-env ClassicBackend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal, cast

import mujoco
import torch
import warp as wp

from holosoma.utils.sampler import DistributionLike, DistributionSpec, TermSampler, quantiles

if TYPE_CHECKING:
    import mujoco_warp as mjwarp


@wp.kernel(module="unique")
def repeat_array_kernel(
    src: wp.array(dtype=Any),  # type: ignore[valid-type]
    nelems_per_world: int,
    dst: wp.array(dtype=Any),  # type: ignore[valid-type]
):
    """Warp kernel to repeat array elements across worlds."""
    tid = wp.tid()
    src_idx = tid % nelems_per_world
    dst[tid] = src[src_idx]  # type: ignore[index]


def expand_model_fields(
    model: mjwarp.Model,
    nworld: int,
    fields_to_expand: list[str],
) -> None:
    """Expand model fields to support per-environment randomization.

    Tiles single-world model fields across all environments, enabling
    per-environment physics parameter randomization. This must be called
    BEFORE the randomization manager is initialized.

    Parameters
    ----------
    model : mjwarp.Model
        MuJoCo Warp model to expand
    nworld : int
        Number of parallel environments
    fields_to_expand : list[str]
        List of field names to expand (e.g., ['body_mass', 'geom_friction'])
    """
    if nworld == 1:
        return

    # Initialize registry to track which fields have been expanded
    if not hasattr(model, "_expanded_fields"):
        model._expanded_fields = set()  # type: ignore[attr-defined]

    def tile(x: wp.array) -> wp.array:
        """Tile a Warp array across environments."""
        # Create new array with same shape but first dim multiplied by nworld.
        new_shape = list(x.shape)
        new_shape[0] = nworld
        wp_array = cast(
            "Callable[..., Any]", {1: wp.array, 2: wp.array2d, 3: wp.array3d, 4: wp.array4d}[len(new_shape)]
        )
        dst = wp_array(shape=new_shape, dtype=x.dtype, device=x.device)

        src_flat = x.flatten()
        dst_flat = dst.flatten()

        # Launch kernel to repeat data, one thread per destination element.
        n_elems_per_world = dst_flat.shape[0] // nworld
        wp.launch(
            repeat_array_kernel,
            dim=dst_flat.shape[0],
            inputs=[src_flat, n_elems_per_world],
            outputs=[dst_flat],
            device=x.device,
        )
        return dst

    for field in model.__dataclass_fields__:
        # Skip fields already expanded: re-tiling REALLOCATES the array (setattr below), which
        # would invalidate a captured CUDA step-graph that still points at the old array. Making
        # this idempotent lets the runtime static-move path call it safely (it's a no-op once the
        # field was expanded at setup); only the first expansion per field allocates.
        if field in fields_to_expand and field not in model._expanded_fields:  # type: ignore[attr-defined]
            array = getattr(model, field)
            setattr(model, field, tile(array))
            # Register this field as expanded for validation
            model._expanded_fields.add(field)  # type: ignore[attr-defined]


def resolve_entity_ids(mj_model: mujoco.MjModel, names: list[str], entity_type: str) -> list[int]:
    """Resolve entity names to MuJoCo indices.

    Parameters
    ----------
    mj_model : mujoco.MjModel
        The CPU MuJoCo model
    names : List[str]
        List of entity names to resolve
    entity_type : str
        The type of entity ("body", "geom", "joint", "site", "actuator", etc.)

    Returns
    -------
    List[int]
        List of MuJoCo indices corresponding to the entity names

    Raises
    ------
    ValueError
        If entity type is unknown or entity name is not found
    """
    # Map string type to MuJoCo enum
    type_map = {
        "body": mujoco.mjtObj.mjOBJ_BODY,
        "geom": mujoco.mjtObj.mjOBJ_GEOM,
        "joint": mujoco.mjtObj.mjOBJ_JOINT,
        "site": mujoco.mjtObj.mjOBJ_SITE,
        "actuator": mujoco.mjtObj.mjOBJ_ACTUATOR,
        "camera": mujoco.mjtObj.mjOBJ_CAMERA,
        "sensor": mujoco.mjtObj.mjOBJ_SENSOR,
        "light": mujoco.mjtObj.mjOBJ_LIGHT,
        "mesh": mujoco.mjtObj.mjOBJ_MESH,
        "texture": mujoco.mjtObj.mjOBJ_TEXTURE,
        "material": mujoco.mjtObj.mjOBJ_MATERIAL,
    }

    if entity_type.lower() not in type_map:
        raise ValueError(f"Unknown entity type: '{entity_type}'. Supported: {list(type_map.keys())}")

    obj_type = type_map[entity_type.lower()]
    indices = []

    for name in names:
        idx = mujoco.mj_name2id(mj_model, obj_type, name)
        if idx == -1:
            # Try prefixed name
            idx_prefixed = mujoco.mj_name2id(mj_model, obj_type, "robot_" + name)
            if idx_prefixed == -1:
                raise ValueError(f"Entity '{name}' of type '{entity_type}' not found in model.")
            idx = idx_prefixed
        indices.append(idx)

    return indices


def _field_view(simulator: Any, field: str) -> torch.Tensor:
    """A writable ``[num_worlds, ...]`` torch view of MuJoCo model ``field`` on either backend.

    WarpBackend: the per-world bridge view (already ``[num_envs, ...]``). ClassicBackend: the single
    CPU ``MjModel`` field wrapped as a torch view with a length-1 world axis prepended, so callers
    index it identically. ``from_numpy`` aliases the live model on Classic, so writes land on the
    model ``mj_step`` reads.
    """
    if hasattr(simulator.backend, "warp_model_bridge"):
        return getattr(simulator.backend.warp_model_bridge, field)
    return torch.from_numpy(getattr(simulator.backend.model, field)).unsqueeze(0)


def scale_inertia_by_mass_ratio(
    simulator: Any, body_ids: torch.Tensor, mass_before: torch.Tensor, *, min_mass: float = 1e-6
) -> None:
    """Scale ``body_inertia`` rows for ``body_ids`` by each body's CURRENT/``mass_before`` ratio.

    The MuJoCo half of the cross-backend ``recompute_inertia`` contract: after a mass write, multiply
    each body's diagonal inertia (``body_inertia`` is ``[nbody, 3]`` principal moments) by the factor
    its mass changed by, so a uniform-density mass change scales inertia consistently with the Isaac
    backends. Multiplicative => commutes with a separate inertia-shape DR term (order-independent).

    ``mass_before`` is the ``[num_worlds, len(body_ids)]`` mass snapshot taken BEFORE the mass write.
    """
    mass_now = _field_view(simulator, "body_mass")[:, body_ids]  # [num_worlds, n_body]
    ratio = (mass_now / mass_before.clamp(min=min_mass)).to(torch.float64)  # body_inertia is float64-backed
    inertia = _field_view(simulator, "body_inertia")  # [num_worlds, nbody, 3]
    inertia[:, body_ids, :] *= ratio.to(inertia.dtype).unsqueeze(-1)


def randomize_field(
    simulator: Any,
    field: str,
    ranges: DistributionLike | dict[int, DistributionLike],
    sampler: TermSampler,
    env_ids: torch.Tensor | None = None,
    entity_ids: torch.Tensor | None = None,
    entity_names: list[str] | None = None,
    entity_type: str | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    shared_across_entities: bool = False,
    axis_base: int = 0,
    num_buckets: int | None = None,
) -> None:
    """Unified model randomization for the MuJoCo backends (WarpBackend GPU + ClassicBackend CPU).

    Randomizes physics parameters in the MuJoCo model for specified environments and entities.
    On the WarpBackend this is vectorized across the per-world-expanded model via the
    WarpBridge; on the single-env ClassicBackend it writes the one ``mujoco.MjModel`` field
    directly (one sample — there is only one environment). The indexing/sampling logic is
    shared: the classic field is wrapped as a ``[1, ...]`` view so the same code path applies.

    Parameters
    ----------
    simulator : Any
        The simulator instance (WarpBackend exposes backend.warp_model_bridge; ClassicBackend
        exposes backend.model).
    field : str
        Model field name to randomize (e.g., 'body_mass', 'geom_friction', 'body_ipos')
    ranges : Union[Tuple[float, float], Dict[int, Tuple[float, float]]]
        Range(s) for randomization. Each value is a ``[lo, hi]`` pair or a
        ``{kind, low, high, mean, std}`` spec dict. Can be:
        - Single range for scalar fields or all axes
        - Dict mapping axis indices to ranges for vector fields
    sampler : TermSampler
        Bound keyed sampler (from the randomization manager). Supplies the draws so the result is
        reproducible per (term, env, episode); entity ids key per-entity independence.
    axis_base : int
        Offset added to each axis's keying coordinate. Lets a term that calls ``randomize_field``
        more than once (e.g. mass link-scale then base-add) keep those calls on separate streams.
    env_ids : Optional[torch.Tensor]
        Environment IDs to randomize (default: all environments)
    entity_ids : Optional[torch.Tensor]
        Entity IDs to randomize (default: all entities)
    entity_names : Optional[List[str]]
        Entity names to resolve to IDs (mutually exclusive with entity_ids)
    entity_type : Optional[str]
        Type of entity for name resolution (e.g., 'body', 'geom')
        Required if entity_names is provided, otherwise inferred from field
    operation : Literal["add", "scale", "abs"]
        Operation to apply: add to current, scale current, or set absolute value
    shared_across_entities : bool
        If True, every entity gets the SAME per-env sample (the first entity's), instead of an
        independent draw each. Use to randomize a group of entities as one logical unit — e.g.
        a freejoint's 3 linear ``dof_damping`` DOFs sharing one value (PhysX exposes a single
        linear damping scalar, so this keeps the concept aligned across backends).
    num_buckets : Optional[int]
        Quantization knob mirroring the PhysX backends' material bucketing, for a config that wants
        the SAME staircase discretization on MuJoCo. ``None`` (default): draw continuously — MuJoCo
        writes the model field directly and has no per-scene material cap, so continuous is the true
        marginal. An int: quantize each axis's draw onto ``num_buckets`` stratified quantile values of
        its distribution (a shared per-axis table, keyed-shuffled, then each (env, entity) picks a
        bucket) — the same n-atom staircase the IsaacSim material writer produces. Only the VALUE set
        is discretized; per-entity granularity is unchanged. Applies to any field/operation.

    Raises
    ------
    ValueError
        If both entity_ids and entity_names are specified
        If entity_type cannot be inferred from field name
    """
    device = simulator.sim_device

    # -----------------------------------------------------------
    # 0. Pre-resolution: Name -> ID Logic
    # -----------------------------------------------------------
    if entity_names is not None:
        if entity_ids is not None:
            raise ValueError("Cannot specify both 'entity_ids' and 'entity_names'. Choose one.")
        # 1. Access the CPU model to look up names
        mj_model = simulator.backend.model

        # 2. Infer entity type if not provided
        if entity_type is None:
            # Simple heuristic based on common naming conventions
            if field.startswith("body_"):
                entity_type = "body"
            elif field.startswith("geom_"):
                entity_type = "geom"
            elif field.startswith(("jnt_", "joint_")):
                entity_type = "joint"
            elif field.startswith("site_"):
                entity_type = "site"
            elif field.startswith(("actuator_", "gear")):
                entity_type = "actuator"
            else:
                raise ValueError(
                    f"Could not infer entity type for field '{field}'. "
                    "Please provide explicit 'entity_type' (e.g., 'body', 'geom')."
                )

        # 3. Resolve names to integer list
        ids_list = resolve_entity_ids(mj_model, entity_names, entity_type)
        entity_ids = torch.tensor(ids_list, device=device, dtype=torch.long)

    # -----------------------------------------------------------
    # 1. Retrieve the Field and Determine Shapes
    # -----------------------------------------------------------
    is_warp = hasattr(simulator.backend, "warp_model_bridge")
    model_field = _field_view(simulator, field)
    full_shape = model_field.shape

    ndim = len(full_shape)
    n_world = full_shape[0]
    n_total_entities = full_shape[1]

    # -----------------------------------------------------------
    # 1.5. Validate Field Expansion
    # -----------------------------------------------------------
    # Per-env expansion (and its validation) only applies to the WarpBackend with >1 env. The
    # single-env ClassicBackend writes the model field in place — no expansion needed.
    num_envs = simulator.num_envs
    if is_warp and num_envs > 1:
        mjw_model = simulator.backend.mjw_model
        expanded_fields: set[str] = getattr(mjw_model, "_expanded_fields", set())

        if field not in expanded_fields:
            raise ValueError(
                f"Field '{field}' has not been expanded for per-environment randomization. "
                f"Did you forget to add @mujoco_required_field('{field}') to your randomization function? "
                f"Currently expanded fields: {sorted(expanded_fields) if expanded_fields else 'none'}"
            )

    # -----------------------------------------------------------
    # 2. Resolve Indices (Broadcasting Prep)
    # -----------------------------------------------------------
    if env_ids is None:
        env_ids = torch.arange(n_world, device=device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device, dtype=torch.long)

    if entity_ids is None:
        entity_ids = torch.arange(n_total_entities, device=device, dtype=torch.long)
    else:
        entity_ids = entity_ids.to(device, dtype=torch.long)

    # -- Target Axes & Ranges --
    # Keep the raw per-axis ranges (each a [lo, hi] pair OR a {kind/low/high/mean/std} spec dict)
    # rather than flattening to a float tensor, so an explicit-(mean,std) gaussian survives to the
    # sampler. ``ranges`` itself may be a dict KEYED BY AXIS INDEX (int -> range) for a vector field;
    # disambiguate that from a single spec-dict range by inspecting the keys.
    target_axes: torch.Tensor | None = None
    is_axis_map = isinstance(ranges, dict) and len(ranges) > 0 and all(isinstance(k, int) for k in ranges)
    if isinstance(ranges, dict) and len(ranges) == 0:
        raise ValueError("randomize_field: `ranges` is an empty dict; pass at least one axis -> range.")

    range_leaves: list[DistributionLike]
    if ndim == 3:
        # Vector field
        if is_axis_map:
            axis_map = cast("dict[int, DistributionLike]", ranges)
            axes_list = sorted(axis_map.keys())
            target_axes = torch.tensor(axes_list, device=device, dtype=torch.long)
            range_leaves = [axis_map[ax] for ax in axes_list]
        else:
            target_axes = torch.arange(full_shape[2], device=device, dtype=torch.long)
            range_leaves = [cast("DistributionLike", ranges)] * full_shape[2]
    else:
        # Scalar field
        if is_axis_map:
            raise ValueError("Cannot specify axis dict for a scalar (2D) field.")
        range_leaves = [cast("DistributionLike", ranges)]

    # -----------------------------------------------------------
    # 3. Create Broadcasting Views
    # -----------------------------------------------------------
    idx_env = env_ids.view(-1, 1, 1)  # (N, 1, 1)
    idx_ent = entity_ids.view(1, -1, 1)  # (1, M, 1)

    indexer: tuple[torch.Tensor, ...]
    if target_axes is not None:
        idx_ax = target_axes.view(1, 1, -1)  # (1, 1, K)
        indexer = (idx_env, idx_ent, idx_ax)
    else:
        indexer = (idx_env.squeeze(-1), idx_ent.squeeze(-1))

    # -----------------------------------------------------------
    # 4. Generate Random Values
    # -----------------------------------------------------------
    n_e = len(env_ids)
    n_n = len(entity_ids)
    n_a = len(target_axes) if target_axes is not None else 1

    # Sample each axis through the bound TermSampler so a MuJoCo config means exactly what it does on
    # every other backend AND is reproducible per (term, env, episode). Entity ids are the stable MuJoCo
    # indices (body/geom/dof), passed as a ``[1, n_n]`` coord so a per-entity draw lands on the trailing
    # dimension and survives iteration-order changes; ``axis_base + k`` is the int STREAM coord keeping
    # the K axis leaves on independent streams. Validation (log_uniform positivity, high>=low) fires at
    # spec construction, raising a clean error instead of silently writing NaN. Stack the per-axis draws
    # back into the (n_e, n_n, n_a) layout the indexer expects.
    if num_buckets is None:
        per_axis = [
            sampler.draw(
                leaf, env_ids=env_ids, coords=(axis_base + k, entity_ids[None, :]), device=device
            )  # (n_e, n_n)
            for k, leaf in enumerate(range_leaves)
        ]
    else:
        # Bucketed: quantize each axis onto num_buckets stratified quantile values (mirrors the PhysX
        # material writer). Fill a per-axis bucket table with the spec's quantiles, shuffle it with a
        # KEYED permutation (distinct stream coord axis_base + k so axes are not rank-aligned), then let
        # each (env, entity) pick a bucket via the keyed draw_int. Per-entity granularity is preserved —
        # only the VALUE set is discretized, matching IsaacSim's staircase.
        per_axis = []
        for k, leaf in enumerate(range_leaves):
            column = quantiles(DistributionSpec.parse(leaf), num_buckets, device)[
                sampler.permute(num_buckets, (axis_base + k,))
            ]
            bucket_ids = sampler.draw_int(
                0, num_buckets - 1, env_ids=env_ids, coords=(axis_base + k, entity_ids[None, :])
            )  # (n_e, n_n)
            per_axis.append(column[bucket_ids])
    random_values = torch.stack(per_axis, dim=-1)  # (n_e, n_n, n_a)

    # Share one per-env sample across the whole entity group (the first entity's draw), so a
    # multi-DOF/entity unit is randomized as one value rather than independently per entity.
    if shared_across_entities and n_n > 1:
        random_values = random_values[:, :1, :].expand(n_e, n_n, n_a).clone()

    if target_axes is None:
        random_values = random_values.squeeze(-1)

    # Match the destination dtype: the WarpBridge fields are float32, but the ClassicBackend's
    # fields are views of float64 numpy arrays — an "abs" assignment of float32 into a float64
    # destination raises a dtype-mismatch in torch's index_put. Casting here is a no-op on Warp.
    random_values = random_values.to(model_field.dtype)

    # -----------------------------------------------------------
    # 5. Apply Operation
    # -----------------------------------------------------------
    current_data = model_field[indexer]

    if operation == "add":
        model_field[indexer] = current_data + random_values
    elif operation == "scale":
        model_field[indexer] = current_data * random_values
    elif operation == "abs":
        model_field[indexer] = random_values
    else:
        raise ValueError(f"Unknown operation: {operation}")
