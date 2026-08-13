"""Pure (no-pxr) helpers for deriving collision-free body names from USD prim paths.

A multi-body USD scene file (1->N) becomes one ``RigidObject`` per rigid-body prim, and
each needs a stable identifier used BOTH as the per-body key for pattern matching
(include/exclude, object_configs) AND, namespaced as ``{file}_{id}``, as the
ObjectRegistry actor name / spawned prim leaf.

The old loader used the prim's bare leaf name, so two bodies that share a leaf under
different parents (``/world/A/box`` and ``/world/B/box`` -> both ``box``) collided. This
module instead trims the longest common ANCESTOR shared by every body and keeps the
distinguishing tail, joined with ``_``:

    /world/foo/A/box, /world/foo/B/box  ->  "A_box", "B_box"

For a FLAT file (every body a sibling under one root, e.g. ``/multibody/free_box`` and
``/multibody/static_post``) the common ancestor is that single root, so the result is the
bare leaf (``free_box`` / ``static_post``) — identical to the old behavior, so flat files
and the existing presets/tests are unaffected.

Kept pure (string ops only, no ``pxr``) so it imports and unit-tests in the CPU env, where
``pxr``/``isaaclab`` are absent.
"""

from __future__ import annotations


def _segments(prim_path: str) -> list[str]:
    """Path string -> non-empty segments (``/world/foo/A/box`` -> ``[world, foo, A, box]``)."""
    return [s for s in prim_path.split("/") if s]


def _common_prefix_len(segment_lists: list[list[str]]) -> int:
    """Number of leading segments shared by ALL paths, capped so every path keeps its leaf.

    Capping at ``min(len) - 1`` guarantees at least one trailing (distinguishing) segment
    survives for every body — so a single body, or a body whose full path is a prefix of
    another's, still yields a non-empty name.
    """
    if not segment_lists:
        return 0
    cap = min(len(s) for s in segment_lists) - 1  # always leave >=1 trailing segment
    common = 0
    while common < cap and len({s[common] for s in segment_lists}) == 1:
        common += 1
    return common


def distinguishing_names(prim_paths: list[str]) -> dict[str, str]:
    """Map each full prim path to a collision-free body identifier.

    Trims the longest common ancestor of all ``prim_paths`` and joins each path's
    remaining segments with ``_``. Returns ``{prim_path: name}`` preserving input order.

    Raises
    ------
    ValueError
        If two distinct prim paths collapse to the same identifier (only possible for
        pathological names where the ``_`` join is ambiguous, e.g. ``/a/b_c`` vs
        ``/a_b/c``). Surfaced loudly rather than silently shadowing one body.
    """
    segment_lists = [_segments(p) for p in prim_paths]
    cut = _common_prefix_len(segment_lists)

    names: dict[str, str] = {}
    used: dict[str, str] = {}
    for prim_path, segs in zip(prim_paths, segment_lists):
        name = "_".join(segs[cut:])
        if name in used:
            raise ValueError(
                f"Prim paths '{used[name]}' and '{prim_path}' both reduce to body name "
                f"'{name}' after trimming the common ancestor. Rename the conflicting prims "
                f"so their distinguishing path segments don't collide under '_' joining."
            )
        used[name] = prim_path
        names[prim_path] = name
    return names
