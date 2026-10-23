"""Unwrapping Isaac Lab 3.x ``ProxyArray`` data reads into plain ``torch.Tensor``.

Isaac Lab 3.0 is warp-first: every ``asset.data.*`` accessor returns a
:class:`isaaclab.utils.warp.ProxyArray` wrapping a ``warp.array``, rather than the
``torch.Tensor`` that 2.3.2 returned. The wrapper carries a deprecation bridge, so indexing and
arithmetic still work (warning once), which is what makes this an easy divergence to miss -- most
expressions keep working and the breakage is narrow but hard.

The narrow part matters: the bridge forwards operators but **not tensor instance methods**. Its
own docstring is explicit -- "Tensor instance methods such as ``clone()`` are not forwarded; use
explicit ``.torch`` access for those." holosoma calls ``.clone()``, ``.view()``, ``.to()`` and
similar on data reads, and passes them to ``torch`` functions that type-check their arguments, so
the wrapper has to come off at the boundary rather than being carried into holosoma's own code.

``as_tensor`` is that boundary. It is deliberately tolerant of a real tensor so the same call is
correct whichever an accessor returns -- 3.x converts accessors to ``ProxyArray`` progressively,
and a few already hand back tensors.
"""

from __future__ import annotations

from typing import Any

import torch


def as_tensor(value: Any) -> torch.Tensor:
    """Return ``value`` as a ``torch.Tensor``, unwrapping a 3.x ``ProxyArray`` if needed.

    Parameters
    ----------
    value
        The result of an Isaac Lab ``asset.data.*`` read: a ``ProxyArray`` on 3.x, or already a
        ``torch.Tensor``.

    Returns
    -------
    torch.Tensor
        A zero-copy torch view of the same memory. No data is copied, so a caller that needs an
        independent buffer must still ``.clone()`` the result.

    Raises
    ------
    TypeError
        If ``value`` is neither a tensor nor something exposing ``.torch``. Raised rather than
        returned as-is so an unexpected accessor shape surfaces here, at the boundary, instead of
        as a confusing failure deeper inside holosoma.
    """
    if isinstance(value, torch.Tensor):
        return value
    torch_view = getattr(value, "torch", None)
    if isinstance(torch_view, torch.Tensor):
        return torch_view

    # The physics *views* (``asset.root_view.get_coms()`` and friends) return a bare
    # ``warp.array`` rather than a ProxyArray, so there is no ``.torch`` to read -- the
    # conversion has to go through warp itself. Imported lazily because warp must bind to the
    # runtime's copy, which only exists once the simulation app has launched.
    import warp as wp

    if isinstance(value, wp.array):
        return wp.to_torch(value)

    raise TypeError(
        f"Expected a torch.Tensor, an Isaac Lab ProxyArray exposing `.torch`, or a warp.array, "
        f"got {type(value).__name__}."
    )


def as_warp(value: Any) -> Any:
    """Return ``value`` as a ``warp.array``, for the physics-view *setters*.

    The counterpart to :func:`as_tensor`, and needed for the same reason in the other direction.
    Isaac Sim 6.2's ``omni.physics.tensors`` uses a warp frontend, so ``root_view.set_masses`` /
    ``set_inertias`` / ``set_coms`` / ``set_material_properties`` expect warp arrays. Handing one
    a ``torch.Tensor`` does not raise a clear type error -- it reaches
    ``warp.types.type_ctype`` deep inside the extension and fails with

        TypeError: issubclass() arg 1 must be a class

    which reads like a warp/Isaac Sim version conflict rather than a wrong argument type. (It is
    additionally confusing because the bundled extension is built against warp 1.16 while Isaac
    Lab 3.0 installs 1.17.)

    Isaac Lab 2.3.2's PhysX frontend accepted torch tensors directly, which is why the 5.1 lane
    has no equivalent call.

    Raises
    ------
    TypeError
        If ``value`` is neither a warp array nor a torch tensor, rather than passing an unknown
        type through to fail obscurely inside the extension.
    """
    import warp as wp

    if isinstance(value, wp.array):
        return value
    torch_view = value.torch if hasattr(value, "torch") else value
    if isinstance(torch_view, torch.Tensor):
        # Zero-copy view of the same device memory; the setter reads it immediately.
        return wp.from_torch(torch_view.contiguous())
    raise TypeError(f"Expected a warp.array or torch.Tensor to hand a physics view, got {type(value).__name__}.")


def as_warp_indices(value: Any) -> Any:
    """Return ``value`` as a warp index array the physics-view setters accept.

    Separate from :func:`as_warp` because the index argument has a stricter contract than the
    data argument: ``omni.physics.tensors``' warp frontend runs indices through
    ``as_contiguous_uint32``, which accepts only int32/uint32 and raises otherwise. holosoma's
    ``env_ids`` are torch ``long`` (int64), as every other backend wants, so they need an
    explicit narrowing cast -- without it the failure is the same opaque

        TypeError: issubclass() arg 1 must be a class

    raised from inside warp, giving no hint that the *dtype* of the indices is the problem.
    """
    import warp as wp

    if isinstance(value, wp.array):
        return value
    tensor = as_tensor(value) if not isinstance(value, torch.Tensor) else value
    return wp.from_torch(tensor.to(torch.int32).contiguous())
