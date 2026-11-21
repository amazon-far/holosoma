"""Unit tests for the ``#tag`` name-spec resolver (pure, no simulator)."""

from __future__ import annotations

import copy

import pytest

from holosoma.utils.name_tags import (
    expand_joint_value_map,
    expand_name_assignments,
    expand_name_spec,
    expand_unique_name_spec,
    is_tag,
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


def test_is_tag() -> None:
    assert is_tag("#legs")
    assert not is_tag("legs")


def test_exact_names_pass_through_in_order() -> None:
    assert expand_name_spec(["torso", "l_hip"], _TAGS, _UNIVERSE) == ["torso", "l_hip"]


def test_tag_expands_and_composes_preserving_first_seen_order() -> None:
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


def test_unknown_name_lists_universe_and_kind() -> None:
    with pytest.raises(ValueError, match=r"Unknown joint name 'nope'.*known joints: \['l_hip'"):
        expand_name_spec(["nope"], _TAGS, _UNIVERSE, kind="joint")


def test_unknown_body_name_says_bodies() -> None:
    with pytest.raises(ValueError, match=r"Unknown body name 'nope'.*known bodies: \['l_hip'"):
        expand_name_spec(["nope"], _TAGS, _UNIVERSE)


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
    assert expand_name_spec(["nope", "#knees"], _TAGS, None) == ["nope", "l_knee", "r_knee"]


def test_cycle_reports_chain() -> None:
    tags = {"#a": ["#b"], "#b": ["#c"], "#c": ["#a"]}
    with pytest.raises(ValueError, match=r"body tag cycle: #a -> #b -> #c -> #a"):
        expand_name_spec(["#a"], tags, None)


def test_non_string_entry_is_type_error() -> None:
    with pytest.raises(TypeError):
        expand_name_spec([3], _TAGS, _UNIVERSE)  # type: ignore[list-item]


def test_validate_tag_table_accepts_valid_table() -> None:
    validate_tag_table(_TAGS, _UNIVERSE)


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


def test_joint_value_map_all_is_automatic_and_can_be_overridden() -> None:
    assert expand_joint_value_map({"#all": 0.1}, {}, ["a", "b"]) == {"a": 0.1, "b": 0.1}
    assert expand_joint_value_map({"#all": 0.1, "b": 0.2}, {"#all": ["a"]}, ["a", "b"]) == {
        "a": 0.1,
        "b": 0.2,
    }


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
        ("torso", (0.8, 1.2)),
        ("l_knee", (0.8, 1.2)),
        ("r_knee", (0.8, 1.2)),
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


def test_ordered_spec_keeps_tag_and_entry_order() -> None:
    assert expand_unique_name_spec(["torso", "#right_leg", "#left_leg"], _TAGS, _UNIVERSE, what="pairs.left") == [
        "torso",
        "r_hip",
        "r_knee",
        "l_hip",
        "l_knee",
    ]


@pytest.mark.parametrize("spec", [["#left_leg", "l_knee"], ["#repeated"]])
def test_ordered_spec_rejects_repetitions_including_within_one_key(spec: list[str]) -> None:
    tags = {**_TAGS, "#repeated": ["#left_leg", "l_knee"]}
    with pytest.raises(ValueError, match=r"pairs.left.*l_knee.*more than once.*source keys") as error:
        expand_unique_name_spec(spec, tags, _UNIVERSE, what="pairs.left")
    assert repr(spec[0]) in str(error.value)
    assert repr(spec[-1]) in str(error.value)
