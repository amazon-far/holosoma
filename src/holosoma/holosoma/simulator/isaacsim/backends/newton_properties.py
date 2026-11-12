"""Resolve Newton material properties by actual shape ownership.

Materials need global Newton shape IDs: the SDK's material view assumes each
actor's shapes are contiguous, which is not true for every imported scene.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor


def _material_bindings(asset: Any) -> tuple[Any, Tensor, Tensor, Tensor]:
    """Return the manager, global material arrays and ``[env, shape]`` IDs.

    Resolve each asset instance's articulation and named bodies before collecting
    their colliders. This keeps partial randomization away from other actors,
    even when their shape IDs are interleaved in Newton's global model.
    """
    import torch
    import warp as wp
    from isaaclab.sim import SimulationContext

    simulation = SimulationContext.instance()
    if simulation is None:
        raise RuntimeError("Newton material properties require an initialized simulation context.")
    manager = simulation.physics_manager
    model = manager.get_model()
    if model is None:
        raise RuntimeError("Newton material properties require an initialized physics model.")
    friction = wp.to_torch(model.shape_material_mu)
    restitution = wp.to_torch(model.shape_material_restitution)
    if friction.ndim != 1 or friction.shape != restitution.shape:
        raise RuntimeError("Newton material channels must have matching one-dimensional global shape arrays.")

    view = asset.root_view
    articulations = view.articulation_ids.numpy()
    if articulations.ndim != 2 or articulations.shape[1] != 1:
        raise RuntimeError("Newton asset material access requires one articulation per instance.")
    if articulations.shape[0] != asset.num_instances:
        raise RuntimeError("Newton material selection must contain every asset instance.")
    selected_names = set(view.link_names)
    if not selected_names or len(selected_names) != len(view.link_names):
        raise RuntimeError("Newton material selection requires unique body names.")

    starts = model.articulation_start.numpy()
    ends = model.articulation_end.numpy()
    children = model.joint_child.numpy()
    rows = []
    for (articulation,) in articulations:
        articulation = int(articulation)
        if not 0 <= articulation < len(ends):
            raise RuntimeError("Newton material selection contains an invalid articulation ID.")
        start, end = int(starts[articulation]), int(ends[articulation])
        if not 0 <= start < end <= len(children):
            raise RuntimeError("Newton material selection contains an invalid joint range.")
        body_ids = set(int(body) for body in children[start:end])
        selected_bodies = {}
        for body in body_ids:
            if not 0 <= body < len(model.body_label):
                raise RuntimeError("Newton material selection contains an invalid body ID.")
            name = model.body_label[body].rsplit("/", 1)[-1]
            if name in selected_names:
                if name in selected_bodies:
                    raise RuntimeError(f"Newton material selection contains duplicate body name {name!r}.")
                selected_bodies[name] = body
        if selected_bodies.keys() != selected_names:
            raise RuntimeError("Newton material selection does not resolve every requested body.")
        shapes = sorted({shape for body in selected_bodies.values() for shape in model.body_shapes.get(body, [])})
        if len(shapes) != view.shape_count or any(not 0 <= shape < friction.numel() for shape in shapes):
            raise RuntimeError("Newton material selection has an inconsistent owned shape count or invalid shape ID.")
        rows.append(shapes)
    shape_ids = torch.tensor(rows, dtype=torch.int64, device=friction.device)
    return manager, friction, restitution, shape_ids


def read_materials(asset: Any) -> Tensor:
    """Read ``[env, shape, (friction, restitution)]`` into an owned CPU buffer.

    Newton has one friction coefficient. There is no dynamic-friction channel.
    """
    import torch

    _, friction, restitution, shape_ids = _material_bindings(asset)
    flat_ids = shape_ids.flatten()
    return torch.stack(
        (
            friction.index_select(0, flat_ids).reshape(shape_ids.shape),
            restitution.index_select(0, flat_ids).reshape(shape_ids.shape),
        ),
        dim=-1,
    ).cpu()


def write_materials(asset: Any, full_materials: Tensor, env_ids: Tensor) -> None:
    """Write selected rows of the full two-channel buffer and notify the solver."""
    import torch
    from newton import ModelFlags

    manager, friction, restitution, shape_ids = _material_bindings(asset)
    expected = (*shape_ids.shape, 2)
    if tuple(full_materials.shape) != expected:
        raise ValueError(f"Newton material buffer must have shape {expected}, got {tuple(full_materials.shape)}.")
    indices = env_ids.to(device=friction.device, dtype=torch.int64).contiguous()
    selected = full_materials.to(friction.device).index_select(0, indices)
    global_ids = shape_ids.index_select(0, indices).flatten()
    # Newton's articulation material view assumes contiguous shapes. Native
    # model ownership also handles interleaved shapes from other scene actors.
    friction.index_copy_(0, global_ids, selected[..., 0].reshape(-1))
    restitution.index_copy_(0, global_ids, selected[..., 1].reshape(-1))
    manager.add_model_change(ModelFlags.SHAPE_PROPERTIES)
