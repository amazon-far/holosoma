"""Backend-agnostic asset-format selection for scene rigid objects.

Each backend loads a different subset of the tri-format sibling fields on
``RigidObjectConfig`` (``usd_file`` / ``urdf_file`` / ``xml_file``):

    backend     supported formats (in preference order)
    --------    ----------------------------------------
    MuJoCo      xml, urdf            (MjSpec cannot load USD)
    IsaacGym    urdf                 (loads URDF only)
    IsaacSim    usd, urdf            (UrdfFileCfg converts URDF->USD)

``select_asset_format`` is the single place that turns "what the object provides"
plus "what this backend supports" into the chosen ``(format, path)`` — so all three
backends share one selection rule. A format the backend cannot load is never silently
dropped: if the object only provides unsupported formats, it raises with the full
matrix (what was provided vs. what the backend supports).

Which formats a backend supports (in preference order) is owned by each simulator class
via ``BaseSimulator.get_supported_scene_formats()`` (a subclass can extend/override it);
that list is passed into ``select_asset_format`` at the call sites.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from holosoma.config_types.scene import RigidObjectConfig, SceneFileConfig


def select_asset_format(obj: RigidObjectConfig | SceneFileConfig, supported_formats: list[str]) -> tuple[str, str]:
    """Pick the asset format to load for ``obj`` on a backend supporting ``supported_formats``.

    Parameters
    ----------
    obj : RigidObjectConfig | SceneFileConfig
        The object whose per-format asset fields are inspected. Works for both the
        ``{fmt}_file`` fields on RigidObjectConfig and the ``{fmt}_path`` fields on
        SceneFileConfig — whichever sibling set the object exposes.
    supported_formats : list[str]
        Formats this backend can load, in preference order (e.g. ``["xml", "urdf"]``
        for MuJoCo). The first one the object actually provides is chosen.

    Returns
    -------
    tuple[str, str]
        ``(format, path)`` where format is one of ``"usd"``/``"urdf"``/``"xml"`` and
        path is the corresponding (unresolved) asset path from the config.

    Raises
    ------
    ValueError
        If the object provides none of the backend's supported formats. The message
        states the object's name, the formats it provides, and what the backend supports.
    """
    provided = {
        fmt: getattr(obj, f"{fmt}_file", None) or getattr(obj, f"{fmt}_path", None) for fmt in ("usd", "urdf", "xml")
    }

    for fmt in supported_formats:
        path = provided.get(fmt)
        if path:
            return fmt, path

    have = sorted(fmt for fmt, path in provided.items() if path) or ["none"]
    raise ValueError(
        f"Asset provides {have} but this backend loads "
        f"{list(supported_formats)}. Set one of the supported formats on the asset."
    )
