"""Retargeting: CmdSMPLH -> Retargeter -> CmdDense.

Retargeter implementations are discovered from the ``holosoma.retargeter``
entry-point group so extensions (e.g. FAR-pi ``holosoma_extensions``) can
register their own embodiment-specific retargeter without a code change here —
select it at launch via ``retargeter:=<name>``. Each entry point must resolve to
a class implementing the :class:`Retargeter` Protocol, constructible with
``(urdf_path: str, dt: float)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from holosoma_inference.pycompat import entry_points

if TYPE_CHECKING:
    # Import only for typing: smpl_retargeter pulls in mink/mujoco, which the
    # inference CI env lacks. Keeping it out of runtime import also preserves
    # the lazy-load contract — heavy impls load only when actually selected.
    from holosoma_service.retargetting.smpl_retargeter import Retargeter

_GROUP = "holosoma.retargeter"
# Lazy: keep the entry points, only .load() the one actually requested so a
# broken/heavy extension impl doesn't get imported unless it's selected.
_entry_points = {ep.name: ep for ep in entry_points(group=_GROUP)}
_registry: dict[str, type] = {}


def available_retargeters() -> list[str]:
    """Names registered under the ``holosoma.retargeter`` entry-point group."""
    return sorted(_entry_points)


def create_retargeter(name: str, urdf_path: str, dt: float) -> Retargeter:
    """Instantiate the retargeter registered under *name*.

    Raises ``ValueError`` if *name* is not registered by any installed package.
    """
    if name not in _entry_points:
        raise ValueError(f"Unknown retargeter: {name!r}. Available: {available_retargeters()}")
    if name not in _registry:
        _registry[name] = _entry_points[name].load()
    return _registry[name](urdf_path=urdf_path, dt=dt)
