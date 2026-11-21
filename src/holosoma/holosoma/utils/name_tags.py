"""Expand exact names and ``#tag`` references for robot configs and task terms.

Declare tags in ``RobotConfig.body_tags`` or ``joint_tags``, for example
``{"#legs": ["#left_leg", "#right_leg"]}``. Expansion preserves first-seen order and rejects
unknown tags, cycles, and names outside a supplied universe. Regex and glob syntax are unsupported.
"""

from __future__ import annotations

# Keep this module independent of torch and config_types so deployment readers can use saved fields directly.
import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
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
    spec: Sequence[str],
    tags: Mapping[str, Sequence[str]],
    universe: Sequence[str] | None = None,
    *,
    kind: str = "body",
    dedupe: bool = True,
) -> list[str]:
    """Expand ``spec`` (exact names and ``#tag`` references) into an ordered name list.

    Parameters
    ----------
    spec : Sequence[str]
        Exact names and/or ``#tag`` references, in the order they should appear.
    tags : Mapping[str, Sequence[str]]
        Tag table (``RobotConfig.body_tags`` or ``joint_tags``), keyed by the ``#tag`` name. Entries may
        reference other tags.
    universe : Sequence[str] | None
        When given, every expanded name must be in it (e.g. ``simulator.body_names``); ``None``
        skips the name check (used when only the tag structure is being validated).
    kind : str
        ``"body"`` or ``"joint"``; only used in error messages.
    dedupe : bool
        Drop repeated names (the default, for set-like specs). ``False`` keeps every occurrence, for
        specs where multiplicity matters (a mirror pair list).

    Raises
    ------
    ValueError
        Unknown tag, unknown name, or tag cycle. The message lists the defined tags / known names.
    TypeError
        ``spec`` or a tag's entry list is a bare string (wrap the name in a list), or an entry is not a
        string.
    """
    out: list[str] = []
    seen: set[str] = set()
    universe_set = set(universe) if universe is not None else None

    def visit(entries: Sequence[str], chain: tuple[str, ...]) -> None:
        if isinstance(entries, str):
            where = f"{kind} tag '{chain[-1]}'" if chain else f"{kind} spec"
            raise TypeError(
                f"{where} must be a list of names, got the string {entries!r}; wrap it in a list: [{entries!r}]"
            )
        for entry in entries:
            if not isinstance(entry, str):
                raise TypeError(f"{kind} spec entries must be strings, got {entry!r} in {list(spec)!r}")
            if is_tag(entry):
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
                if dedupe and entry in seen:
                    continue
                seen.add(entry)
                out.append(entry)

    visit(spec, ())
    return out


def resolve_single_name_spec(
    spec: str,
    tags: Mapping[str, Sequence[str]],
    universe: Sequence[str],
    *,
    kind: str = "body",
) -> str:
    """Resolve one exact name or ``#tag`` that must expand to exactly one name."""
    names = expand_name_spec([spec], tags, universe, kind=kind)
    if len(names) != 1:
        raise ValueError(f"{kind} spec {spec!r} must resolve to exactly one {kind}; got {names}.")
    return names[0]


def validate_tag_table(tags: Mapping[str, Sequence[str]], universe: Sequence[str], *, kind: str = "body") -> None:
    """Check a whole tag table: ``#``-prefixed keys and every tag expandable against ``universe``.

    Raises ``ValueError`` on a key that does not start with ``#`` (a tag is declared and referenced in the
    same form), an empty tag, or any expansion error (unknown tag/name, cycle) reported by
    :func:`expand_name_spec`.
    """
    for tag, entries in tags.items():
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


def expand_unique_name_spec(
    spec: Sequence[str],
    tags: Mapping[str, Sequence[str]],
    universe: Sequence[str],
    *,
    what: str,
    kind: str = "joint",
) -> list[str]:
    """Expand an ordered correspondence side, rejecting repetitions before any deduplication.

    This differs from a selection: a repeated entity inside a single composed tag is also invalid.
    Errors name the entity and the top-level source entries, including their positions.
    """
    if isinstance(spec, str):
        raise TypeError(f"{what} must be a list of names, got the string {spec!r}")
    names: list[str] = []
    source_for: dict[str, tuple[int, str]] = {}
    for index, key in enumerate(spec):
        for name in expand_name_spec([key], tags, universe, kind=kind, dedupe=False):
            if name in source_for:
                previous_index, previous_key = source_for[name]
                raise ValueError(
                    f"{what} lists {kind} {name!r} more than once: source keys "
                    f"{previous_key!r} (entry {previous_index}) and {key!r} (entry {index})."
                )
            source_for[name] = (index, key)
            names.append(name)
    return names


def expand_joint_value_map(
    table: Mapping[str, float],
    joint_tags: Mapping[str, Sequence[str]],
    dof_names: Sequence[str],
    *,
    what: str = "joint values",
    required: bool = True,
) -> dict[str, float]:
    """Return a validated exact-name map from joint values keyed by DOF names or ``#tags``.

    The result is a fresh dict in ``dof_names`` order; no input is modified. Tags may compose, and
    ``#all`` defaults to every DOF unless explicitly declared, as on ``RobotConfig``. This helper
    accepts serialized config fields without constructing a ``RobotConfig``.

    Every DOF must be covered by exactly one key. Overlapping keys are errors even when their values
    agree; repeated references within a single tag are deduplicated. ``required=False`` permits
    uncovered DOFs and omits them from the result rather than assigning an implicit value.

    Raises ``ValueError`` for duplicate DOF names, invalid tags, unknown names, overlapping or missing
    assignments, or non-finite values. ``what`` identifies the field in assignment errors.
    """
    if isinstance(dof_names, str):
        raise TypeError(f"{what} dof_names must be a list of names, got the string {dof_names!r}")
    duplicates = sorted(name for name, count in Counter(dof_names).items() if count > 1)
    if duplicates:
        raise ValueError(f"{what} dof_names has duplicate entries: {duplicates}")
    tags = {"#all": list(dof_names), **joint_tags} if dof_names else dict(joint_tags)
    validate_tag_table(tags, dof_names, kind="joint")

    validated: dict[str, float] = {}
    for key, raw_value in table.items():
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"{what} value for {key!r} must be finite, got {value!r}.")
        validated[key] = value
    values = expand_name_assignments(
        validated,
        lambda spec: expand_name_spec(spec, tags, dof_names, kind="joint"),
        what=what,
        kind="DOF",
    )
    missing = [dof for dof in dof_names if dof not in values]
    if missing and required:
        raise ValueError(f"{what} has no key covering DOF(s) {missing}.")
    return {dof: values[dof] for dof in dof_names if dof in values}
