"""Reading and writing bulk physics properties, for whichever engine owns the asset.

What this exists for
--------------------
holosoma's domain-randomization terms read a property for every body or shape, modify it, and
write it back. Isaac Sim 5.1 and Isaac Lab 3.0's PhysX backend both expose that as a tensor view
on the asset (``asset.root_physx_view`` / ``asset.root_view``) with ``get_masses`` / ``set_masses``
and friends. Newton has no such methods: its view is a
:class:`newton.selection.ArticulationView`, which exposes ``get_attribute`` / ``set_attribute``
over named fields of the Newton ``Model`` instead.

Measured: of the seven view members the randomization terms use (``get``/``set`` of masses,
inertias, coms and material properties, plus ``max_shapes``), Newton's view provides **none**.

Why an adapter instead of moving to Isaac Lab's portable API
------------------------------------------------------------
Isaac Lab 3.0 does offer backend-portable *setters* on the asset itself
(``set_masses_index``, ``set_coms_index``, ``set_inertias_index``). They are not sufficient here:

* There is no portable equivalent for **material properties** at all -- no
  ``set_material_properties`` exists on any asset class, only on the PhysX view.
* The portable *reads* are authored defaults (``data.default_mass``), and the randomization terms
  deliberately compose onto the **current live** value. That is not an accident: the comment at
  ``randomize_rigid_body_mass`` records that composing onto current is what keeps one config
  meaning the same thing across IsaacGym, MuJoCo and Isaac Sim, and that reading defaults instead
  made a second call apply ``draw - prior_draw``.

So moving the PhysX path onto the portable API would risk changing numerics on the one engine
that is verified. This adapter keeps PhysX byte-identical to what the harnesses measured and
confines all new behaviour to Newton.

Dispatch is on the **view**, not on configuration
-------------------------------------------------
The randomization terms are Isaac Lab manager terms -- they receive ``env`` and a
``SceneEntityCfg``, not holosoma's simulator config, so a configured-engine check would have to be
threaded through every term. Instead each function asks the view whether it provides the PhysX
tensor API. That is duck typing, but on exactly the thing that differs, and it cannot disagree
with reality the way a passed-down enum could.

Newton's single friction coefficient
------------------------------------
PhysX material properties are ``[static_friction, dynamic_friction, restitution]`` per shape.
Newton has ``shape_material_mu`` (one Coulomb coefficient) and ``shape_material_restitution`` --
there is no static/dynamic split. The three-column layout is preserved at this boundary so the
randomization terms stay engine-agnostic, and on Newton:

* reads report ``mu`` in **both** friction columns;
* writes take the **dynamic** column as ``mu``.

Dynamic rather than static because Newton's ``mu`` is a sliding coefficient, and because
holosoma's own ``make_consistent`` rule already forces ``dynamic <= static``, making dynamic the
conservative of the two. This is a genuine behavioural difference between the engines rather than
a detail of this module -- it is recorded in ``docs/isaac-sim-backends.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from holosoma.simulator.isaacsim6.proxy_array import as_tensor, as_warp, as_warp_indices

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject

_MU = "shape_material_mu"
_RESTITUTION = "shape_material_restitution"
_BODY_ATTRIBUTES = {"masses": "body_mass", "inertias": "body_inertia", "coms": "body_com"}


def _view(asset: RigidObject | Articulation) -> Any:
    return asset.root_view


def _is_physx_view(view: Any) -> bool:
    """Whether this view provides the PhysX bulk-property tensor API.

    ``get_masses`` stands in for the whole family: the PhysX tensor view defines all seven
    members and Newton's ``ArticulationView`` defines none, so one probe decides the engine.
    """
    return hasattr(view, "get_masses")


def _newton_model(view: Any) -> Any:
    """The Newton ``Model`` the view reads from.

    ``ArticulationView.__init__`` stores its model as ``self.model``; ``get_attribute`` /
    ``set_attribute`` both need it passed back as the source/target.
    """
    return view.model


def _squeeze_articulations(values: torch.Tensor, attribute: str) -> torch.Tensor:
    """Drop Newton's per-articulation axis, which holosoma's layout does not have.

    Newton returns ``(num_envs, num_articulations, num_bodies, ...)`` where the PhysX tensor view
    returns ``(num_envs, num_bodies, ...)`` -- measured: ``body_mass`` is ``(4, 1, 42)`` and
    ``body_com`` is ``(4, 1, 42, 3)`` on a 4-environment G1 scene.

    Each of these views addresses one asset, so the articulation axis is 1 and dropping it is
    lossless. More than one would mean the view matched several articulations per environment, in
    which case silently taking the first would randomize the wrong bodies -- so that raises.
    """
    if values.ndim < 2:
        raise RuntimeError(f"Newton attribute {attribute!r} has unexpected shape {tuple(values.shape)}.")
    if values.shape[1] != 1:
        raise RuntimeError(
            f"Newton attribute {attribute!r} has {values.shape[1]} articulations per environment "
            f"(shape {tuple(values.shape)}); holosoma's per-asset layout assumes exactly one, so "
            f"collapsing this axis would address the wrong bodies."
        )
    return values.squeeze(1)


def _matches_physx_body_axis(values: torch.Tensor, asset: RigidObject | Articulation) -> torch.Tensor:
    """Drop the single-body axis for a rigid object, because the PhysX view does.

    Measured contract of the PhysX tensor view on a 4-environment scene -- it is not uniform, so
    this cannot be inferred:

    ======================== ================== ==================
    accessor                 Articulation       RigidObject
    ======================== ================== ==================
    ``get_masses``           ``(4, 42)``        ``(4, 1)``
    ``get_material_properties`` ``(4, 29, 3)``  ``(4, 1, 3)``
    ``get_inertias``         ``(4, 42, 9)``     ``(4, 9)``
    ======================== ================== ==================

    So masses and material properties keep a length-1 body/shape axis for a rigid object while
    inertias (and, by the same rule, coms) drop it. Newton always reports the axis, and the
    randomization terms are written against the PhysX shapes -- ``randomize_rigid_body_mass``
    does ``inertias[env_ids] *= ratios`` on the rigid-object branch, which with the extra axis
    broadcasts ``(4, 1, 9)`` against ``(4, 1)`` into ``(4, 4, 9)`` and raises. Matching the
    measured layout here keeps those terms engine-agnostic instead of teaching each one about
    two shapes.
    """
    return values if _is_articulation(asset) else values.squeeze(1)


def _restore_physx_body_axis(values: torch.Tensor, asset: RigidObject | Articulation) -> torch.Tensor:
    """Inverse of :func:`_matches_physx_body_axis`, for the write direction."""
    return values if _is_articulation(asset) else values.unsqueeze(1)


def _is_articulation(asset: RigidObject | Articulation) -> bool:
    """Whether ``asset`` is an articulation rather than a single-body rigid object.

    ``BaseArticulation`` is imported here, not at module scope. Importing it eagerly pulls in
    ``isaaclab.assets.articulation``, which does ``import omni.physics.tensors.impl.api`` -- a
    module that only exists inside a running Kit runtime. That made this module unimportable
    anywhere else, and because ``test_physics_properties_isaacsim6`` imports it at module scope,
    pytest raised a *collection* error in the Isaac Sim 5.1 environment. A collection error
    aborts the whole run, so one eager import took the entire ``no_sim`` suite down rather than
    just its own file.

    ``BaseArticulation``, not ``Articulation``: in Isaac Lab 3.0 the generic class is a
    ``FactoryBase`` that dispatches per backend, and the object the scene holds derives from the
    base, so ``isinstance(asset, Articulation)`` is always False.
    """
    from isaaclab.assets import BaseArticulation

    return isinstance(asset, BaseArticulation)


def torch_device_of(warp_array: Any) -> str:
    """The torch device string for a warp array's device.

    ``warp.array.device`` is a ``warp.Device``; its string form is ``"cuda:0"`` / ``"cpu"``, which
    torch accepts directly.
    """
    return str(warp_array.device)


def _newton_get(view: Any, attribute: str) -> torch.Tensor:
    """One Newton model attribute, with the articulation axis removed, on the host.

    Returned on the **host**, because that is what the PhysX tensor view returns for these bulk
    property arrays and the whole point of this module is that callers cannot tell the engines
    apart. Measured on the same scene: under PhysX every tensor in the CoM randomization term --
    the array itself, ``default_coms``, ``base_com_bias``, the index tensors -- is on CPU, while
    Newton hands these attributes back on ``cuda:0``.

    Leaving Newton's device as-is surfaces inside a randomization term as
    ``RuntimeError: Expected all tensors to be on the same device``, because the term's own
    companion state (``base_com_bias``) is allocated on the host. Normalising here rather than in
    each term keeps the terms engine-agnostic -- and normalising *towards PhysX* rather than
    towards the simulation device is what keeps the verified path unchanged.
    """
    values = _squeeze_articulations(as_tensor(view.get_attribute(attribute, _newton_model(view))), attribute)
    return values.cpu()


def _newton_set(view: Any, attribute: str, values: torch.Tensor, env_ids: torch.Tensor) -> None:
    """Write ``values`` for ``env_ids`` through Newton's attribute API.

    ``values`` arrives in holosoma's layout (no articulation axis); the axis is restored here so
    the write matches what :func:`_newton_get` read.

    ``set_attribute`` selects rows with a boolean **mask**, not an index list (unlike the PhysX
    view, which takes indices), so ``env_ids`` are converted. Passing indices where a mask is
    expected would reinterpret them as booleans and write the wrong environments.
    """
    import warp as wp

    mask = torch.zeros(values.shape[0], dtype=torch.bool, device=values.device)
    mask[env_ids.to(mask.device)] = True
    restored = values.unsqueeze(1).contiguous()

    # The warp array must carry the attribute's own dtype, not a plain float array. `body_inertia`
    # is `mat33`, so Newton holds it as a 3-D array of matrices; handing over the equivalent 5-D
    # float array fails with "Failed to create array with shape (4, 1, 42, 3, 3), the maximum
    # number of dimensions is 4". The dtype is read back off the live attribute rather than kept
    # in a table here, so it cannot drift from whatever Newton actually uses.
    current = view.get_attribute(attribute, _newton_model(view))
    # Back onto whichever device Newton keeps the attribute on. Reads are normalised to the host
    # (see `_newton_get`), so without this the write hands a host tensor to a device array.
    target_device = torch_device_of(current)
    restored = restored.to(target_device)
    mask = mask.to(target_device)
    view.set_attribute(attribute, _newton_model(view), wp.from_torch(restored, dtype=current.dtype), as_warp(mask))


def has_split_friction(asset: RigidObject | Articulation) -> bool:
    """Whether this engine distinguishes static from dynamic friction.

    PhysX carries both per shape; Newton has a single ``shape_material_mu``. Exposed so the
    randomization terms can refuse a request they cannot honour instead of writing a column the
    engine does not have -- see :func:`set_material_properties` for what the write actually does.
    """
    return _is_physx_view(_view(asset))


def get_masses(asset: RigidObject | Articulation) -> torch.Tensor:
    """Current per-body masses, shape ``(num_envs, num_bodies)``."""
    view = _view(asset)
    if _is_physx_view(view):
        return as_tensor(view.get_masses())
    return _newton_get(view, _BODY_ATTRIBUTES["masses"])


def set_masses(asset: RigidObject | Articulation, masses: torch.Tensor, env_ids: torch.Tensor) -> None:
    """Write per-body masses for ``env_ids``."""
    view = _view(asset)
    if _is_physx_view(view):
        view.set_masses(as_warp(masses), as_warp_indices(env_ids))
        return
    _newton_set(view, _BODY_ATTRIBUTES["masses"], masses, env_ids)


def get_inertias(asset: RigidObject | Articulation) -> torch.Tensor:
    """Current per-body inertia tensors, flattened 3x3 per body."""
    view = _view(asset)
    if _is_physx_view(view):
        return as_tensor(view.get_inertias())
    # Newton stores a 3x3 matrix per body; the PhysX view -- and so the randomization terms --
    # expect it flattened to 9. Row-major either way, so a reshape is the whole conversion.
    inertias = _newton_get(view, _BODY_ATTRIBUTES["inertias"])
    return _matches_physx_body_axis(inertias.reshape(*inertias.shape[:2], 9), asset)


def set_inertias(asset: RigidObject | Articulation, inertias: torch.Tensor, env_ids: torch.Tensor) -> None:
    """Write per-body inertia tensors for ``env_ids``."""
    view = _view(asset)
    if _is_physx_view(view):
        view.set_inertias(as_warp(inertias), as_warp_indices(env_ids))
        return
    restored = _restore_physx_body_axis(inertias, asset)
    matrices = restored.reshape(*restored.shape[:2], 3, 3).contiguous()
    _newton_set(view, _BODY_ATTRIBUTES["inertias"], matrices, env_ids)


def get_coms(asset: RigidObject | Articulation) -> torch.Tensor:
    """Current per-body centre-of-mass offsets."""
    view = _view(asset)
    if _is_physx_view(view):
        return as_tensor(view.get_coms())
    # Newton's `body_com` is a position only; the PhysX view returns position plus a CoM-frame
    # orientation, and the randomization terms index `[..., :3]` for the position. An identity
    # quaternion is appended so that layout holds. Nothing is lost on read -- Newton has no CoM
    # orientation to report -- but see `set_coms` for the write direction.
    positions = _newton_get(view, _BODY_ATTRIBUTES["coms"])
    identity = torch.zeros(*positions.shape[:2], 4, dtype=positions.dtype, device=positions.device)
    identity[..., 3] = 1.0
    return _matches_physx_body_axis(torch.cat((positions, identity), dim=-1), asset)


def set_coms(asset: RigidObject | Articulation, coms: torch.Tensor, env_ids: torch.Tensor) -> None:
    """Write per-body centre-of-mass offsets for ``env_ids``."""
    view = _view(asset)
    if _is_physx_view(view):
        view.set_coms(as_warp(coms), as_warp_indices(env_ids))
        return
    # Only the position is written: Newton has no per-body CoM orientation. Dropping the
    # quaternion is lossless for holosoma, whose CoM randomization writes `[..., :3]` and never
    # touches the rotation -- but a future term that did rotate the CoM frame would be silently
    # ignored here, which is why this is called out rather than left implicit.
    restored = _restore_physx_body_axis(coms, asset)
    _newton_set(view, _BODY_ATTRIBUTES["coms"], restored[..., :3].contiguous(), env_ids)


def max_shapes(asset: RigidObject | Articulation) -> int:
    """Number of collision shapes per environment.

    On Newton this is derived from the material array's own shape rather than a view attribute,
    so it stays consistent with whatever :func:`get_material_properties` returns.
    """
    view = _view(asset)
    if _is_physx_view(view):
        return int(view.max_shapes)
    # The warp array's own shape, without `_newton_get`'s host copy: only a dimension is wanted,
    # and copying every shape's friction value across the bus to read it is pure waste. The
    # trailing axis is the shape count once Newton's leading (env, articulation) axes are
    # accounted for, which is the same layout `get_material_properties` reports.
    return int(view.get_attribute(_MU, _newton_model(view)).shape[-1])


def get_material_properties(asset: RigidObject | Articulation) -> torch.Tensor:
    """Per-shape ``[static_friction, dynamic_friction, restitution]``.

    On Newton the two friction columns both carry ``shape_material_mu``, which has no
    static/dynamic split -- see the module docstring.
    """
    view = _view(asset)
    if _is_physx_view(view):
        return as_tensor(view.get_material_properties())

    mu = _newton_get(view, _MU)
    restitution = _newton_get(view, _RESTITUTION)
    return torch.stack((mu, mu, restitution), dim=-1)


def set_material_properties(asset: RigidObject | Articulation, materials: torch.Tensor, env_ids: torch.Tensor) -> None:
    """Write per-shape material properties for ``env_ids``.

    On Newton the dynamic-friction column becomes ``mu`` and the static column is dropped, because
    Newton has one coefficient. Dropped rather than averaged or rejected: averaging would invent a
    value the configuration never asked for, and rejecting would make every existing
    friction-randomization config fail on Newton even though a sensible single coefficient exists.
    """
    view = _view(asset)
    if _is_physx_view(view):
        view.set_material_properties(as_warp(materials), as_warp_indices(env_ids))
        return

    _newton_set(view, _MU, materials[..., 1].contiguous(), env_ids)
    _newton_set(view, _RESTITUTION, materials[..., 2].contiguous(), env_ids)
