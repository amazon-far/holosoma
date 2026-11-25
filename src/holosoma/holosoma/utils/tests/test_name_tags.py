"""Unit tests for the ``#tag`` name-spec resolver (pure, no simulator)."""

from __future__ import annotations

import copy
from collections.abc import Collection

import pytest

from holosoma.utils.name_tags import (
    expand_joint_value_map,
    expand_name_assignments,
    expand_name_spec,
    resolve_single_name_spec,
    validate_tag_table,
)

pytestmark = pytest.mark.no_sim

_TAGS = {
    "#left_leg": ["l_hip", "l_knee"],
    "#right_leg": ["r_hip", "r_knee"],
    "#legs": ["#left_leg", "#right_leg"],
    "#knees": ["l_knee", "r_knee"],
}
_UNIVERSE = ["l_hip", "l_knee", "r_hip", "r_knee", "torso"]


def test_exact_names_follow_canonical_order() -> None:
    assert expand_name_spec(["torso", "l_hip"], _TAGS, _UNIVERSE) == ["l_hip", "torso"]


def test_tag_expands_and_composes_in_canonical_order() -> None:
    assert expand_name_spec(["#legs"], _TAGS, _UNIVERSE) == ["l_hip", "l_knee", "r_hip", "r_knee"]
    # Composition + duplicates: knees are already present from #legs, torso appended last.
    assert expand_name_spec(["#legs", "#knees", "torso"], _TAGS, _UNIVERSE) == [
        "l_hip",
        "l_knee",
        "r_hip",
        "r_knee",
        "torso",
    ]


def test_resolve_single_name_spec_accepts_exact_names_and_one_name_tags() -> None:
    assert resolve_single_name_spec("torso", _TAGS, _UNIVERSE) == "torso"
    assert resolve_single_name_spec("#knees", {"#knees": ["l_knee"]}, _UNIVERSE) == "l_knee"


def test_resolve_single_name_spec_rejects_multi_name_tags() -> None:
    with pytest.raises(ValueError, match="must resolve to exactly one body"):
        resolve_single_name_spec("#knees", _TAGS, _UNIVERSE)


def test_unknown_tag_lists_defined_tags() -> None:
    with pytest.raises(ValueError, match=r"Unknown body tag '#feet'.*defined body tags: \['#knees', '#left_leg'"):
        expand_name_spec(["#feet"], _TAGS, _UNIVERSE)


@pytest.mark.parametrize(("kind", "plural"), [("joint", "joints"), ("body", "bodies")])
def test_unknown_name_lists_universe_and_kind(kind: str, plural: str) -> None:
    with pytest.raises(ValueError, match=rf"Unknown {kind} name 'nope'.*known {plural}: \['l_hip'"):
        expand_name_spec(["nope"], _TAGS, _UNIVERSE, kind=kind)


def test_bare_string_spec_is_a_type_error_naming_the_fix() -> None:
    with pytest.raises(
        TypeError, match=r"body spec must be a list of names, got the string 'torso'; wrap it in a list"
    ):
        expand_name_spec("torso", _TAGS, _UNIVERSE)


def test_bare_string_tag_entries_is_a_type_error_naming_the_tag() -> None:
    with pytest.raises(TypeError, match=r"joint tag '#a' must be a list of names, got the string 'l_hip'"):
        expand_name_spec(["#a"], {"#a": "l_hip"}, _UNIVERSE, kind="joint")


def test_diamond_reference_expands_once_without_a_cycle_error() -> None:
    tags = {"#a": ["#b", "#c"], "#b": ["l_hip"], "#c": ["#b", "r_hip"]}
    assert expand_name_spec(["#a"], tags, _UNIVERSE) == ["l_hip", "r_hip"]
    validate_tag_table(tags, _UNIVERSE)


def test_universe_none_skips_name_check() -> None:
    assert expand_name_spec(["nope", "#knees"], _TAGS, None) == ["l_knee", "nope", "r_knee"]


def test_cycle_reports_chain() -> None:
    tags = {"#a": ["#b"], "#b": ["#c"], "#c": ["#a"]}
    with pytest.raises(ValueError, match=r"body tag cycle: #a -> #b -> #c -> #a"):
        expand_name_spec(["#a"], tags, None)


def test_non_string_entry_is_type_error() -> None:
    with pytest.raises(TypeError):
        expand_name_spec([3], _TAGS, _UNIVERSE)  # type: ignore[list-item]


@pytest.mark.parametrize(
    ("tags", "match"),
    [
        ({"": ["l_hip"]}, "leading '#'"),
        ({"legs": ["l_hip"]}, "leading '#'"),
        ({"#": ["l_hip"]}, "leading '#'"),
        ({"#empty": []}, "is empty"),
        ({"#bad": ["missing_link"]}, "Unknown body name 'missing_link'"),
        ({"#a": ["#b"], "#b": ["#a"]}, "tag cycle"),
        ({"#a": ["#nope"]}, "Unknown body tag '#nope'"),
    ],
)
def test_validate_tag_table_rejects(tags: dict[str, list[str]], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validate_tag_table(tags, _UNIVERSE)


def test_joint_value_map_expands_nested_tags_in_dof_order_without_modifying_inputs() -> None:
    table = {"torso": -0.3, "#legs": 0.2}
    tags = copy.deepcopy(_TAGS)
    names = list(_UNIVERSE)
    before = copy.deepcopy((table, tags, names))
    result = expand_joint_value_map(table, tags, names)
    assert list(result.items()) == [
        ("l_hip", 0.2),
        ("l_knee", 0.2),
        ("r_hip", 0.2),
        ("r_knee", 0.2),
        ("torso", -0.3),
    ]
    result["l_hip"] = 99.0
    assert (table, tags, names) == before


def test_joint_value_map_all_is_reserved_and_covers_every_dof() -> None:
    assert expand_joint_value_map({"#all": 0.1}, {}, ["a", "b"]) == {"a": 0.1, "b": 0.1}
    with pytest.raises(ValueError, match="'#all' is reserved and cannot be declared"):
        expand_joint_value_map({"#all": 0.1}, {"#all": ["a"]}, ["a", "b"])
    with pytest.raises(ValueError, match="both cover DOF 'b'"):
        expand_joint_value_map({"#all": 0.1, "b": 0.2}, {}, ["a", "b"])


@pytest.mark.parametrize("kind", ["body", "joint"])
def test_all_expands_current_names_through_composed_tags_without_storing_membership(kind: str) -> None:
    tags = {"#whole": ["#all"]}
    assert expand_name_spec(["b", "#whole"], tags, ["a", "b"], kind=kind) == ["a", "b"]
    assert expand_name_spec(["#whole"], tags, ["b", "c", "a"], kind=kind) == ["b", "c", "a"]
    assert expand_name_spec(["#whole"], tags, [], kind=kind) == []
    assert tags == {"#whole": ["#all"]}


def test_all_and_repeated_exact_names_collapse() -> None:
    assert expand_name_spec(["b", "#all", "b", "#all"], {}, ["a", "b"]) == ["a", "b"]


def test_all_requires_a_name_universe() -> None:
    with pytest.raises(ValueError, match="'#all' requires the known bodies"):
        expand_name_spec(["#all"], {})


@pytest.mark.parametrize("members", [[], ["a"], ["a", "b"]])
def test_all_cannot_be_declared_even_when_it_matches_the_universe(members: list[str]) -> None:
    with pytest.raises(ValueError, match="'#all' is reserved and cannot be declared"):
        expand_name_spec(["#all"], {"#all": members}, ["a", "b"])


def test_joint_value_map_repeated_references_within_one_assignment_are_deduplicated() -> None:
    tags = {**_TAGS, "#pose": ["#legs", "#knees", "torso"]}
    assert expand_joint_value_map({"#pose": 0.2}, tags, _UNIVERSE) == dict.fromkeys(_UNIVERSE, 0.2)


@pytest.mark.parametrize("key", ["#knees", "l_knee"])
def test_joint_value_map_rejects_overlapping_assignments_even_when_values_agree(key: str) -> None:
    with pytest.raises(ValueError, match="both cover DOF 'l_knee'"):
        expand_joint_value_map({"#legs": 0.2, key: 0.2, "torso": 0.0}, _TAGS, _UNIVERSE)


def test_joint_value_map_requires_coverage_unless_explicitly_partial() -> None:
    with pytest.raises(ValueError, match=r"joint values has no key covering DOF\(s\) \['torso'\]"):
        expand_joint_value_map({"#legs": 0.2}, _TAGS, _UNIVERSE)
    assert expand_joint_value_map({"r_knee": 0.3, "l_hip": 0.2}, _TAGS, _UNIVERSE, required=False) == {
        "l_hip": 0.2,
        "r_knee": 0.3,
    }


@pytest.mark.parametrize(
    ("table", "match"),
    [
        ({"#missing": 0.0}, "Unknown joint tag '#missing'"),
        ({"missing": 0.0}, "Unknown joint name 'missing'"),
        ({"#all": float("nan")}, "value for '#all' must be finite"),
        ({"l_hip": float("inf")}, "value for 'l_hip' must be finite"),
        ({"l_hip": float("-inf")}, "value for 'l_hip' must be finite"),
    ],
)
def test_joint_value_map_rejects_invalid_assignments(table: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        expand_joint_value_map(table, _TAGS, _UNIVERSE, required=False)


@pytest.mark.parametrize(
    ("tags", "match"),
    [
        ({"#a": ["#b"], "#b": ["#a"]}, "joint tag cycle"),
        ({"#empty": []}, "is empty"),
        ({"#bad": ["ghost"]}, "Unknown joint name 'ghost'"),
    ],
)
def test_joint_value_map_validates_even_unused_tags(tags: dict[str, list[str]], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        expand_joint_value_map({"#all": 0.0}, tags, _UNIVERSE)


def test_joint_value_map_rejects_duplicate_dof_names() -> None:
    with pytest.raises(ValueError, match=r"dof_names has duplicate entries: \['l_hip'\]"):
        expand_joint_value_map({"#all": 0.0}, {}, ["l_hip", "l_hip"])


def test_joint_value_map_rejects_bare_string_dof_names() -> None:
    with pytest.raises(TypeError, match="dof_names must be a list of names"):
        expand_joint_value_map({"#all": 0.0}, {}, "l_hip")


def test_joint_value_map_accepts_zero_dof_robot() -> None:
    assert expand_joint_value_map({}, {}, []) == {}
    assert expand_joint_value_map({"#all": 0.1}, {}, []) == {}


def test_generic_assignments_preserve_values_and_first_seen_order() -> None:
    tags = {**_TAGS, "#payload": ["torso", "#knees", "torso"]}
    assignments = {"#payload": (0.8, 1.2), "#left_hip": (0.9, 1.1)}
    tags["#left_hip"] = ["l_hip"]
    snapshot = copy.deepcopy((tags, assignments))
    expanded = expand_name_assignments(
        assignments,
        lambda spec: expand_name_spec(spec, tags, _UNIVERSE),
        what="mass overrides",
        kind="body",
    )
    assert list(expanded.items()) == [
        ("l_knee", (0.8, 1.2)),
        ("r_knee", (0.8, 1.2)),
        ("torso", (0.8, 1.2)),
        ("l_hip", (0.9, 1.1)),
    ]
    assert (tags, assignments) == snapshot


@pytest.mark.parametrize("second", ["l_knee", "#knees"])
def test_generic_assignment_conflict_names_entity_and_both_keys(second: str) -> None:
    with pytest.raises(ValueError, match="both cover body 'l_knee'") as error:
        expand_name_assignments(
            {"#legs": (1.0, 2.0), second: (1.0, 2.0)},
            lambda spec: expand_name_spec(spec, _TAGS, _UNIVERSE),
            what="mass overrides",
            kind="body",
        )
    assert str(error.value).startswith("mass overrides keys '#legs'")
    assert repr(second) in str(error.value)
    assert "both cover body 'l_knee'" in str(error.value)


@pytest.mark.parametrize("spec", [["#left_leg", "l_knee"], ["#repeated"]])
def test_repetitions_inside_composed_tags_collapse(spec: list[str]) -> None:
    tags = {**_TAGS, "#repeated": ["#left_leg", "l_knee"]}
    assert expand_name_spec(spec, tags, _UNIVERSE) == ["l_hip", "l_knee"]


@pytest.mark.parametrize("members", [["l_knee", "l_hip", "l_knee"], {"l_hip", "l_knee"}, ("l_knee", "l_hip")])
def test_tag_membership_has_no_declaration_order(members: Collection[str]) -> None:
    assert expand_name_spec(["#selected"], {"#selected": members}, _UNIVERSE) == ["l_hip", "l_knee"]
