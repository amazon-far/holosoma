"""Newton's rigid-object view, addressed by articulation rather than by body prim.

Why this exists
---------------
``isaaclab_newton.assets.RigidObject._initialize_impl`` resolves the prim carrying
``UsdPhysics.RigidBodyAPI`` beneath its configured ``prim_path`` and then builds an
``ArticulationView`` from *that* path::

    resolve_kwargs = {"predicate": has_rigid_body_api, "expected_num_matches": 1}
    _, root_prim_path_expr = resolve_matching_prims_from_source(self.cfg.prim_path, **resolve_kwargs)[0]
    self._root_view = ArticulationView(SimulationManager.get_model(), path_expr_to_glob(root_prim_path_expr))

Those are two different namespaces in Newton's model, and for holosoma's assets they do not
coincide. Measured on a 2-environment scene, both labels exist and differ::

    body_label         ['/World/envs/env_0/freebox/Geometry/baseLink', ...]   <- what it searches
    articulation_label ['/World/envs/env_0/freebox', ...]                     <- what it indexes

so the lookup raises ``KeyError: No articulations matching pattern
'/World/envs/env_*/freebox/Geometry/baseLink'`` even though the model is complete and correct.

This only bites assets whose ``RigidBodyAPI`` sits *below* the asset root. Isaac Lab's own Newton
tests spawn primitives (``CuboidCfg`` at ``/World/Cube``), where the API is on the root prim and
the two namespaces coincide -- which is why upstream does not see it. holosoma's scene objects come
from USD/URDF conversion and carry a ``Geometry/baseLink`` substructure, so they do.

Note holosoma is not passing the wrong path: ``object_spawner`` configures
``prim_path="/World/envs/env_[^/]+/{name}"``, the asset root, which is exactly what Newton
indexes. The mismatch is introduced by the resolution step above.

Why an override and not a config change
---------------------------------------
There is no configuration that reconciles them: the body prim is wherever the asset author put
``RigidBodyAPI``, and moving it would change the USD that the PhysX path (the verified one) also
consumes. Re-pointing ``prim_path`` at the inner body does not help either -- resolution would
return the same inner path and the articulation lookup would still miss.

So the fix is to keep upstream's body resolution (it correctly identifies *which* body this object
is) and then ask the model which articulation owns that body, instead of assuming the two paths
are the same string.

This subclasses the **concrete** ``isaaclab_newton`` class deliberately. The generic
``isaaclab.assets.RigidObject`` is a ``FactoryBase`` whose ``__init_subclass__`` raises
``ImportError: Factory class ... must be defined within the 'isaaclab' package``, so a subclass
outside Isaac Lab must derive from the backend implementation -- the same constraint the LiDAR
ray-caster override runs into.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from holosoma.config_types.simulator import IsaacPhysicsBackend


def _segments(path: str) -> list[str]:
    return [part for part in path.split("/") if part]


def _owning_articulation_expression(model: Any, body_prim_path: str, body_path_expr: str) -> str:
    """The articulation path expression owning the body at ``body_prim_path``.

    Newton labels an articulation by its root. For holosoma's converted single-body assets that
    is the asset prim, while the rigid body itself is a descendant (``/Geometry/baseLink``), so
    the body path has to be shortened to the articulation that contains it.

    The amount to shorten is *measured per asset*, not assumed. ``body_prim_path`` is a concrete
    path (Isaac Lab resolves it against the source instance), so the longest
    ``articulation_label`` that is an ancestor of it -- or equal to it -- is unambiguously the
    owner. The number of trailing segments that owner drops is then applied to
    ``body_path_expr``, the multi-environment pattern the view actually needs.

    Measuring beats assuming here: a first attempt inferred one global depth from the label set
    and mis-trimmed, because assets sit at different depths in the same scene (the robot's
    articulation root is ``Robot/Geometry/pelvis``, four segments deeper than ``freebox``).

    Degenerates correctly in both edge cases: when the body *is* the articulation root (Isaac
    Lab's own primitive-spawner assets, where ``RigidBodyAPI`` sits on the configured prim) the
    trim is zero and the expression is returned unchanged; when no ancestor is an articulation
    the expression is also returned unchanged, so the caller raises Isaac Lab's own error rather
    than a differently-worded one from here.
    """
    labels = [str(label) for label in getattr(model, "articulation_label", []) or []]
    body_segments = _segments(body_prim_path)

    # Ancestor test on segments, not string prefixes: "/World/envs/env_0/freebox" is not an
    # ancestor of "/World/envs/env_0/freebox_lid/Geometry/baseLink" even though it prefixes it.
    owners = [
        label
        for label in labels
        if (parts := _segments(label)) == body_segments[: len(parts)] and len(parts) <= len(body_segments)
    ]
    if not owners:
        return body_path_expr

    owner = max(owners, key=lambda label: len(_segments(label)))
    trim = len(body_segments) - len(_segments(owner))
    if trim == 0:
        return body_path_expr

    expr_segments = _segments(body_path_expr)
    view_expr = "/" + "/".join(expr_segments[: len(expr_segments) - trim])
    logger.debug(
        "Newton rigid object: body {} is owned by articulation {}; addressing view as {}",
        body_prim_path,
        owner,
        view_expr,
    )
    return view_expr


def rigid_object_cls(backend: IsaacPhysicsBackend) -> type:
    """The ``RigidObject`` class to build scene objects with, for ``backend``.

    Resolved from holosoma's configured engine rather than sniffed off the live
    ``SimulationContext``: the engine is holosoma's own choice, and Isaac Lab exposes no public
    "which backend am I" accessor to sniff reliably anyway.

    Only Newton needs an override, so every other engine gets the generic factory class and never
    imports ``isaaclab_newton`` -- matching the lazy-import rule the rest of this lane follows.
    """
    if backend is not IsaacPhysicsBackend.NEWTON:
        from isaaclab.assets import RigidObject

        # Bound to a typed local first: `isaaclab` is `ignore_missing_imports` in mypy.ini, so
        # every name it exports is `Any`, and returning one directly trips `warn_return_any`.
        generic: type = RigidObject
        return generic
    return _newton_rigid_object_cls()


def _newton_rigid_object_cls() -> type:
    """The ``RigidObject`` subclass that addresses its view by articulation.

    Imported lazily and built on demand so that nothing here is touched on the PhysX path, where
    ``isaaclab_newton`` need not even be importable.
    """
    from isaaclab_newton.assets.rigid_object import rigid_object as upstream

    class _ArticulationAddressedRigidObject(upstream.RigidObject):
        """A Newton ``RigidObject`` whose view is built from the owning articulation's path."""

        def _initialize_impl(self) -> None:
            # The body of this method mirrors `upstream.RigidObject._initialize_impl`, with one
            # line changed (the view's path expression). It cannot call `super()` and fix up
            # afterwards: the base assigns `self._root_view` unconditionally and then builds
            # `RigidObjectData` from it, so by the time control returned the failing view would
            # already have been constructed -- the `KeyError` happens *inside* `ArticulationView`.
            #
            # Every symbol is read off the upstream module rather than re-imported here, so this
            # tracks upstream's own choices (and breaks loudly, not subtly, if one is renamed).

            def has_rigid_body_api(prim: Any) -> bool:
                return bool(prim.HasAPI(upstream.UsdPhysics.RigidBodyAPI))

            resolve_kwargs = {"predicate": has_rigid_body_api, "expected_num_matches": 1}
            body_prim, body_expr = upstream.resolve_matching_prims_from_source(self.cfg.prim_path, **resolve_kwargs)[0]

            model = upstream.SimulationManager.get_model()
            # -- object view, addressed by owning articulation instead of by body prim
            view_expr = _owning_articulation_expression(model, str(body_prim.GetPath()), body_expr)
            self._root_view = upstream.ArticulationView(model, upstream.path_expr_to_glob(view_expr), verbose=False)

            # container for data access
            self._data = upstream.RigidObjectData(self.root_view, self.device)

            # Register callback to rebind simulation data after a full reset (model/state recreation).
            self._physics_ready_handle = upstream.SimulationManager.register_callback(
                lambda _: self._data._create_simulation_bindings(),
                upstream.PhysicsEvent.PHYSICS_READY,
                name=f"rigid_object_rebind_{self.cfg.prim_path}",
            )

            self._create_buffers()
            self._process_cfg()
            self.update(0.0)
            self.data.is_primed = True

    return _ArticulationAddressedRigidObject
