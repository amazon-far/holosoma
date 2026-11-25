"""Expand exact names and ``#tag`` references for robot configs and task terms.

Declare tags in ``RobotConfig.body_tags`` or ``joint_tags``, for example
``{"#legs": {"#left_leg", "#right_leg"}}``. Tags and selections are sets of names.
Expansion collapses duplicates and uses canonical universe order, or lexical order without a universe.
It rejects unknown tags, cycles, and names outside a supplied universe. ``#all`` selects the supplied universe
and cannot be declared in a tag table. Regex and glob syntax are unsupported.
"""

from __future__ import annotations

# Keep this module independent of torch and config_types so deployment readers can use saved fields directly.
import math
from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import TypeVar

TAG_PREFIX = "#"
_PLURALS = {"body": "bodies"}
_Value = TypeVar("_Value")


def _plural(kind: str) -> str:
    return _PLURALS.get(kind, f"{kind}s")


def is_tag(entry: str) -> bool:
    """Return True when ``entry`` is a ``#tag`` reference rather than an exact name."""
    return entry.startswith(TAG_PREFIX)


def expand_name_spec(
    spec: Collection[str],
    tags: Mapping[str, Collection[str]],
    universe: Sequence[str] | None = None,
    *,
    kind: str = "body",
) -> list[str]:
    """Expand a selection into unique names in canonical universe order.

    Parameters
    ----------
    spec : Collection[str]
        Exact names and/or ``#tag`` references. Declaration order has no meaning.
    tags : Mapping[str, Collection[str]]
        Tag table (``RobotConfig.body_tags`` or ``joint_tags``), keyed by the ``#tag`` name. Entries can
        reference other tags.
    universe : Sequence[str] | None
        When given, every expanded name must be in it (e.g. ``simulator.body_names``); ``None``
        skips the name check (used when only the tag structure is being validated).
    kind : str
        ``"body"`` or ``"joint"``; only used in error messages.

    Raises
    ------
    ValueError
        Unknown tag, unknown name, or tag cycle. The message lists the defined tags / known names.
    TypeError
        ``spec`` or a tag's entry list is a bare string (wrap the name in a list), or an entry is not a
        string.
    """
    if "#all" in tags:
        raise ValueError(f"{kind} tag '#all' is reserved and cannot be declared. Use a different tag for a subset.")
    seen: set[str] = set()
    universe_set = set(universe) if universe is not None else None

    def visit(entries: Collection[str], chain: tuple[str, ...]) -> None:
        if isinstance(entries, str):
            where = f"{kind} tag '{chain[-1]}'" if chain else f"{kind} spec"
            raise TypeError(
                f"{where} must be a list of names, got the string {entries!r}; wrap it in a list: [{entries!r}]"
            )
        for entry in entries:
            if not isinstance(entry, str):
                raise TypeError(f"{kind} spec entries must be strings, got {entry!r} in {list(spec)!r}")
            if entry == "#all":
                if universe is None:
                    raise ValueError(f"{kind} tag '#all' requires the known {_plural(kind)}.")
                seen.update(universe)
            elif is_tag(entry):
                if entry in chain:
                    raise ValueError(f"{kind} tag cycle: {' -> '.join((*chain, entry))}")
                if entry not in tags:
                    raise ValueError(
                        f"Unknown {kind} tag '{entry}' in {list(spec)!r}; defined {kind} tags: {sorted(tags)}"
                    )
                visit(tags[entry], (*chain, entry))
            else:
                if universe_set is not None and entry not in universe_set:
                    raise ValueError(
                        f"Unknown {kind} name '{entry}' (from {list(spec)!r}); "
                        f"known {_plural(kind)}: {list(universe or ())}"
                    )
                seen.add(entry)

    visit(spec, ())
    return sorted(seen) if universe is None else [name for name in dict.fromkeys(universe) if name in seen]


def resolve_single_name_spec(
    spec: str,
    tags: Mapping[str, Collection[str]],
    universe: Sequence[str],
    *,
    kind: str = "body",
) -> str:
    """Resolve one exact name or ``#tag`` that must expand to exactly one name."""
    names = expand_name_spec([spec], tags, universe, kind=kind)
    if len(names) != 1:
        raise ValueError(f"{kind} spec {spec!r} must resolve to exactly one {kind}; got {names}.")
    return names[0]


def validate_tag_table(tags: Mapping[str, Collection[str]], universe: Sequence[str], *, kind: str = "body") -> None:
    """Check a whole tag table: ``#``-prefixed keys and every tag expandable against ``universe``.

    Raises ``ValueError`` on a key that does not start with ``#`` (a tag is declared and referenced in the
    same form), an empty tag, or any expansion error (unknown tag/name, cycle) reported by
    :func:`expand_name_spec`.
    """
    for tag, entries in tags.items():
        if tag == "#all":
            raise ValueError(f"{kind} tag '#all' is reserved and cannot be declared. Use a different tag for a subset.")
        if not is_tag(tag) or len(tag) == len(TAG_PREFIX):
            raise ValueError(
                f"{kind} tag names are declared with a leading '{TAG_PREFIX}' ('{TAG_PREFIX}feet'): got {tag!r}"
            )
        if not entries:
            raise ValueError(f"{kind} tag '{tag}' is empty; drop it or list its {kind} names")
        expand_name_spec([tag], tags, universe, kind=kind)


def expand_name_assignments(
    table: Mapping[str, _Value],
    resolve: Callable[[Sequence[str]], Sequence[str]],
    *,
    what: str,
    kind: str = "entity",
) -> dict[str, _Value]:
    """Resolve one assignment layer, rejecting different keys that reach the same entity.

    ``resolve`` validates and expands a selection (for example ``env.robot_names.resolve_body_names``).
    Repetitions within one key are deduplicated; overlap between keys is an error even for equal
    values. The returned dict preserves first-seen entity order and does not modify the input.
    Value validation and precedence between separately validated layers belong to the caller.
    """
    values: dict[str, _Value] = {}
    key_for: dict[str, str] = {}
    for key, value in table.items():
        for name in dict.fromkeys(resolve([key])):
            if name in key_for:
                raise ValueError(
                    f"{what} keys {key_for[name]!r} and {key!r} both cover {kind} {name!r}; "
                    f"each {kind} is assigned by one key."
                )
            key_for[name] = key
            values[name] = value
    return values


def expand_joint_value_map(
    table: Mapping[str, float],
    joint_tags: Mapping[str, Collection[str]],
    dof_names: Sequence[str],
    *,
    what: str = "joint values",
    required: bool = True,
) -> dict[str, float]:
    """Resolve finite joint values into canonical DOF order without changing the inputs.

    Reject invalid names, overlapping assignments, and missing coverage when required=True.
    The reserved #all selector covers every current DOF and cannot be declared in joint_tags.
    """
    if isinstance(dof_names, str):
        raise TypeError(f"{what} dof_names must be a list of names, got the string {dof_names!r}")
    duplicates = sorted(name for name, count in Counter(dof_names).items() if count > 1)
    if duplicates:
        raise ValueError(f"{what} dof_names has duplicate entries: {duplicates}")
    validate_tag_table(joint_tags, dof_names, kind="joint")

    validated: dict[str, float] = {}
    for key, raw_value in table.items():
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"{what} value for {key!r} must be finite, got {value!r}.")
        validated[key] = value
    values = expand_name_assignments(
        validated,
        lambda spec: expand_name_spec(spec, joint_tags, dof_names, kind="joint"),
        what=what,
        kind="DOF",
    )
    missing = [dof for dof in dof_names if dof not in values]
    if missing and required:
        raise ValueError(f"{what} has no key covering DOF(s) {missing}.")
    return {dof: values[dof] for dof in dof_names if dof in values}
