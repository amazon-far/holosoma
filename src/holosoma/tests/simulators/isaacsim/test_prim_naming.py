"""Unit tests for prim-path -> collision-free body name derivation (no pxr, no isaaclab).

``distinguishing_names`` is pure string logic (the Fix-4 hierarchical-name disambiguation), so
it runs in the CPU env where ``pxr``/``isaaclab`` are absent — unlike the rest of
``object_spawner``. It pins the contract every multi-body USD scene file relies on:

- a FLAT file (all bodies siblings under one root) reduces to bare leaf names — backward
  compatible with the pre-Fix-4 behavior and the shipped multibody.usda;
- bodies that share a leaf under different parents stay distinct by keeping the distinguishing
  hierarchy tail joined with ``_``;
- a genuine post-join collision fails loud rather than silently shadowing a body.
"""

from __future__ import annotations

import pytest

from holosoma.simulator.isaacsim.prim_naming import distinguishing_names

# Pure string logic (no pxr/isaaclab); runs in the no_sim CPU job, not the isaacsim job.
pytestmark = pytest.mark.no_sim


def test_flat_file_reduces_to_leaf_names():
    # The shipped multibody.usda shape: two bodies as siblings under one root. The common
    # ancestor is that root, so names are the bare leaves (== pre-Fix-4 behavior).
    names = distinguishing_names(["/multibody/free_box", "/multibody/static_post"])
    assert names == {"/multibody/free_box": "free_box", "/multibody/static_post": "static_post"}


def test_single_body_keeps_its_leaf():
    # One body: common-prefix cap leaves >=1 trailing segment, so the leaf survives.
    assert distinguishing_names(["/world/foo/box"]) == {"/world/foo/box": "box"}


def test_colliding_leaves_keep_distinguishing_parent():
    # The motivating case from the task: same leaf 'box' under different parents A/B. The common
    # ancestor '/world/foo' is trimmed; the distinguishing tail (parent + leaf) is kept.
    names = distinguishing_names(["/world/foo/A/box", "/world/foo/B/box"])
    assert names == {"/world/foo/A/box": "A_box", "/world/foo/B/box": "B_box"}


def test_only_common_prefix_trimmed_not_all_shared_segments():
    # Trimming stops at the FIRST differing segment; shared deeper segments downstream of a
    # divergence are retained (here both end in '/box', but 'box' is kept because the paths
    # already diverged at A/B upstream).
    names = distinguishing_names(["/r/A/x/box", "/r/B/x/box"])
    assert names == {"/r/A/x/box": "A_x_box", "/r/B/x/box": "B_x_box"}


def test_mixed_depths_leave_each_a_leaf():
    # A path that is a prefix of another must still keep a trailing segment for itself; the cap
    # (min length - 1) guarantees it.
    names = distinguishing_names(["/r/a", "/r/a/b"])
    assert names["/r/a"] and names["/r/a/b"]
    assert names["/r/a"] != names["/r/a/b"]


def test_post_join_collision_fails_loud():
    # Pathological names where the '_' join is ambiguous: after trimming the common '/root', the
    # tails [a, b_c] and [a_b, c] both join to 'a_b_c'. Surface loudly rather than silently
    # shadow one body.
    with pytest.raises(ValueError, match="(?i)collide|reduce"):
        distinguishing_names(["/root/a/b_c", "/root/a_b/c"])


def test_order_preserved():
    paths = ["/r/z/box", "/r/a/box", "/r/m/box"]
    assert list(distinguishing_names(paths).keys()) == paths
