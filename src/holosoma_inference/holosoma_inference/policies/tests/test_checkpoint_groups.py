"""Unit tests for ``_normalize_checkpoint_groups`` — the pure checkpoint-spec parser.

Grouped checkpoints: a checkpoint is a group of >= 1 ONNX files loaded together
(1 for a normal policy, N for a teacher-student policy). Explicit groupings are
RESPECTED as-given (heterogeneous sizes allowed). This is a pure function (no ONNX,
no rclpy), so these run anywhere.

``base`` imports onnx/onnxruntime/netifaces at module load; skip cleanly when those
aren't present (host env). The bazel pytest_test target has them.
"""

import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
pytest.importorskip("netifaces")

from holosoma_inference.policies.base import MAX_CHECKPOINTS, _normalize_checkpoint_groups


# --- accepted forms -----------------------------------------------------------


def test_scalar_str_is_one_single_file_checkpoint():
    assert _normalize_checkpoint_groups("a.onnx") == [["a.onnx"]]


def test_flat_list_is_one_checkpoint_per_file():
    # Back-compat: N single-file checkpoints (the historical SWITCH_POLICY meaning).
    assert _normalize_checkpoint_groups(["a.onnx", "b.onnx", "c.onnx"]) == [
        ["a.onnx"],
        ["b.onnx"],
        ["c.onnx"],
    ]


def test_single_flat_file_in_list():
    assert _normalize_checkpoint_groups(["a.onnx"]) == [["a.onnx"]]


def test_explicit_groups_respected_verbatim():
    # Heterogeneous sizes allowed and preserved exactly.
    spec = [["a", "b", "c"], ["d"], ["e", "f"]]
    assert _normalize_checkpoint_groups(spec) == [["a", "b", "c"], ["d"], ["e", "f"]]


def test_teacher_student_single_pair():
    assert _normalize_checkpoint_groups([["backbone.onnx", "student.onnx"]]) == [
        ["backbone.onnx", "student.onnx"]
    ]


def test_teacher_student_multiple_pairs():
    spec = [["bb_A", "st_A"], ["bb_B", "st_B"]]
    assert _normalize_checkpoint_groups(spec) == [["bb_A", "st_A"], ["bb_B", "st_B"]]


def test_tuples_are_accepted_and_normalized_to_lists():
    out = _normalize_checkpoint_groups((("a", "b"), ("c", "d")))
    assert out == [["a", "b"], ["c", "d"]]
    assert all(isinstance(g, list) for g in out)


def test_returns_fresh_lists_not_input_aliases():
    spec = [["a", "b"]]
    out = _normalize_checkpoint_groups(spec)
    out[0].append("mutated")
    assert spec == [["a", "b"]]  # input untouched


def test_max_checkpoints_boundary_ok():
    spec = [[f"m{i}.onnx"] for i in range(MAX_CHECKPOINTS)]
    assert len(_normalize_checkpoint_groups(spec)) == MAX_CHECKPOINTS


# --- fail loud + early --------------------------------------------------------


def test_none_rejected():
    with pytest.raises(ValueError, match="required"):
        _normalize_checkpoint_groups(None)


def test_empty_list_rejected():
    with pytest.raises(ValueError, match="empty"):
        _normalize_checkpoint_groups([])


def test_empty_group_rejected():
    with pytest.raises(ValueError, match="empty group"):
        _normalize_checkpoint_groups([["a"], []])


def test_mixed_grouped_and_flat_rejected():
    with pytest.raises(ValueError, match="mixes grouped and ungrouped"):
        _normalize_checkpoint_groups([["a", "b"], "c"])


def test_non_string_file_rejected():
    with pytest.raises(ValueError, match="non-string"):
        _normalize_checkpoint_groups([["a", 3]])


def test_empty_string_file_rejected():
    with pytest.raises(ValueError, match="non-string"):
        _normalize_checkpoint_groups([["a", ""]])


def test_too_many_checkpoints_rejected():
    spec = [[f"m{i}.onnx"] for i in range(MAX_CHECKPOINTS + 1)]
    with pytest.raises(ValueError, match="up to"):
        _normalize_checkpoint_groups(spec)


def test_wrong_type_rejected():
    with pytest.raises(TypeError, match="str, list, or list"):
        _normalize_checkpoint_groups(42)
