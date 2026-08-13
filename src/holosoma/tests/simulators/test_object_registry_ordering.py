"""Tests for ObjectRegistry index/pose ordering.

``resolve_indices`` must group results in the SAME order that ``get_object_indices``
and ``get_initial_poses_batch`` emit (request order), not sorted-by-position order.
The erroneous behavior guarded against: a reset of the form
``poses = get_actor_initial_poses(names); set_actor_states(names, ..., poses)`` walks
the poses and the resolved actors in lockstep, so if resolution is re-sorted by
position it applies one actor's pose to another whenever the requested name order
differs from the registry's position order.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from holosoma.simulator.shared.object_registry import ObjectRegistry, ObjectType  # noqa: E402


def _registry(num_envs: int = 1) -> ObjectRegistry:
    """Registry with robot at position 0, objA at 1, objB at 2."""
    reg = ObjectRegistry(device="cpu")
    reg.setup_ranges(num_envs, robot_count=1, scene_count=0, individual_count=2)

    def pose(x):
        p = torch.zeros(num_envs, 7)
        p[:, 0] = x
        p[:, 6] = 1.0  # qw
        return p

    reg.register_object("robot", ObjectType.ROBOT, 0, pose(0.0))
    reg.register_object("objA", ObjectType.INDIVIDUAL, 0, pose(10.0))
    reg.register_object("objB", ObjectType.INDIVIDUAL, 1, pose(20.0))
    reg.finalize_registration()
    return reg


def _banded_registry(num_envs: int = 2) -> ObjectRegistry:
    """Registry with all three bands: robot @ pos0, scene table @ pos0, two free boxes @ pos0/1.

    Layout per env (interleaved, env-major): [robot, table, box0, box1] => objects_per_env=4.
    A distinct constant x per object lets a pose-swap from a wrong decode be detected.
    """
    reg = ObjectRegistry(device="cpu")
    reg.setup_ranges(num_envs, robot_count=1, scene_count=1, individual_count=2)

    def pose(x):
        p = torch.zeros(num_envs, 7)
        # Distinct x per (object, env): base x, plus 0.1*env so each env's value is unique too.
        p[:, 0] = x + 0.1 * torch.arange(num_envs, dtype=torch.float32)
        p[:, 6] = 1.0  # qw
        return p

    reg.register_object("robot", ObjectType.ROBOT, 0, pose(0.0))
    reg.register_object("table", ObjectType.SCENE, 0, pose(100.0))
    reg.register_object("box0", ObjectType.INDIVIDUAL, 0, pose(200.0))
    reg.register_object("box1", ObjectType.INDIVIDUAL, 1, pose(300.0))
    reg.finalize_registration()
    return reg


def test_resolve_indices_preserves_request_order():
    """resolve_indices must echo the request name order, not sorted-position order."""
    reg = _registry()
    # Request out of position order: objB (pos 2) before objA (pos 1).
    indices = reg.get_object_indices(["objB", "objA"], torch.tensor([0]))
    resolved = reg.resolve_indices(indices)
    assert [name for name, _ in resolved] == ["objB", "objA"]


def test_initial_pose_roundtrip_does_not_swap_actors():
    """The reset pattern (initial_poses -> resolve) must not cross actors' poses."""
    reg = _registry()
    names = ["objB", "objA"]  # out of position order
    env_ids = torch.tensor([0])

    poses = reg.get_initial_poses_batch(names, env_ids)  # block layout, request order
    indices = reg.get_object_indices(names, env_ids)
    resolved = reg.resolve_indices(indices)

    # Walk states sequentially per resolved group, exactly as set_actor_states_by_index does.
    offset = 0
    applied = {}
    for name, sub_env_ids in resolved:
        n = len(sub_env_ids)
        applied[name] = poses[offset : offset + n, 0].tolist()  # x-coord
        offset += n

    # Each actor must receive ITS OWN initial x (objA=10, objB=20), not swapped.
    assert applied["objA"] == [10.0]
    assert applied["objB"] == [20.0]


# ----- multi-env banding (num_envs >= 2): the real interleaved decode math -----
# At num_envs=1 the interleaved decode (env=idx//objects_per_env, pos=idx%objects_per_env)
# collapses to identity (objects_per_env rows, one env), so a transposed/identity decode would
# pass. With num_envs>=2 the row<->(env,pos) mapping is non-trivial and the banding actually
# matters, so these would FAIL under a transposed decode (env=idx%num_envs, pos=idx//num_envs).


def test_resolve_indices_multi_env_decodes_env_and_position():
    """num_envs=2: resolve_indices must split flat indices into the RIGHT (name, env) groups.

    get_object_indices(["box1","box0"], [0,1]) emits [3,7,2,6] (interleaved, env-major). The
    decode must group these as box1->[0,1], box0->[0,1] (request order preserved). A transposed
    decode (pos=idx//num_envs) would yield wrong names/env-groupings and fail here.
    """
    reg = _banded_registry(num_envs=2)
    indices = reg.get_object_indices(["box1", "box0"], torch.tensor([0, 1]))
    # objects_per_env=4: box1@pos3 -> [3,7], box0@pos2 -> [2,6].
    assert indices.tolist() == [3, 7, 2, 6]

    resolved = reg.resolve_indices(indices)
    assert [name for name, _ in resolved] == ["box1", "box0"]
    assert resolved[0][1].tolist() == [0, 1]  # box1 in envs 0 and 1
    assert resolved[1][1].tolist() == [0, 1]  # box0 in envs 0 and 1


def test_resolve_indices_multi_env_single_object_spans_envs():
    """A single object's flat indices across envs decode back to that one object in every env."""
    reg = _banded_registry(num_envs=3)  # objects_per_env=4
    table_idx = reg.get_object_indices(["table"], torch.tensor([0, 1, 2]))
    # table@pos1: env*4 + 1 -> [1, 5, 9].
    assert table_idx.tolist() == [1, 5, 9]
    resolved = reg.resolve_indices(table_idx)
    assert len(resolved) == 1
    name, env_ids = resolved[0]
    assert name == "table"
    assert env_ids.tolist() == [0, 1, 2]


def test_get_names_by_type_banding():
    """get_names_by_type returns each band's members (registration order), ROBOT/SCENE/INDIVIDUAL."""
    reg = _banded_registry(num_envs=2)
    assert reg.get_names_by_type(ObjectType.ROBOT) == ["robot"]
    assert reg.get_names_by_type(ObjectType.SCENE) == ["table"]
    assert reg.get_names_by_type(ObjectType.INDIVIDUAL) == ["box0", "box1"]


def test_resolved_objects_equals_resolve_indices_of_arange_multi_env():
    """resolved_objects() MUST equal resolve_indices(arange(total)) — the documented contract.

    proxy.clone() uses resolved_objects(); proxy[arange(total)] uses resolve_indices() directly;
    any ordering/grouping deviation between them is a silent clone-vs-index bug. Checked at
    num_envs>=2 where the decode is non-trivial.
    """
    reg = _banded_registry(num_envs=2)
    total = len(reg.objects) * reg.num_envs  # 4 objects * 2 envs = 8

    cached = reg.resolved_objects()
    direct = reg.resolve_indices(torch.arange(total))

    assert [name for name, _ in cached] == [name for name, _ in direct]
    for (cn, ce), (dn, de) in zip(cached, direct):
        assert cn == dn
        assert torch.equal(ce, de)

    # And the contract's intent: every registered object appears once, paired with ALL envs.
    assert [name for name, _ in cached] == ["robot", "table", "box0", "box1"]
    for _, env_ids in cached:
        assert env_ids.tolist() == [0, 1]
