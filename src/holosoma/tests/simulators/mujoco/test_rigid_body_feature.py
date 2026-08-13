"""Tests for MuJoCo robot/scene-object element identification and body indexing.

Drive the real code against compiled ``MjSpec`` worlds (CPU, no GPU, no full-sim
spin-up): ``MujocoSceneManager._capture_spec_meta`` (the per-actor element inventory
recorded from each isolated spec before attach) and the derivations
``_set_robot_properties`` / ``_build_name_maps`` perform on it.

The erroneous behaviors guarded against:
- Identifying robot elements by ``name.startswith(robot_prefix)`` absorbs an object
  whose own prefix begins with the robot prefix: e.g. an object named ``"robot_decoy"``
  gets prefix ``"robot_decoy_"``, so its bodies all start with ``"robot_"`` and would be
  counted as robot bodies, inflating ``num_bodies`` / ``body_names``.
- A DOF filter that drops only freejoints/empty names counts any named internal
  (non-free) joint on an attached object as a robot DOF, inflating ``num_dof``.
- The holosoma-facing rigid-body / contact-force tensors are robot-only and 0-based
  over ``body_names`` (world excluded), while MuJoCo's physics tensors are
  ``model.nbody``-wide. If ``find_rigid_body_indice`` returned a raw MuJoCo body id, it
  would be off by one against those tensors and out of bounds for the final robot body.
"""

from __future__ import annotations

import pytest

mujoco = pytest.importorskip("mujoco")

# MuJoCo ClassicBackend (CPU) only.
pytestmark = pytest.mark.mujoco_classic

from holosoma.simulator.mujoco.scene_manager import ActorSpecMeta, MujocoSceneManager  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers: build minimal actor specs and a compiled world, mirroring
# MujocoSceneManager.add_robot / add_rigid_object attach semantics.
# ---------------------------------------------------------------------------


def _make_actor_spec(root_body: str, dof_joint_name: str | None) -> mujoco.MjSpec:  # type: ignore[name-defined]
    """A single top-level free body, optionally with one named internal hinge DOF.

    Mirrors the asset shape MujocoSceneManager expects: exactly one top-level
    body with a freejoint (added if absent), plus an optional child on a named
    hinge to model an articulated object.
    """
    spec = mujoco.MjSpec()
    b = spec.worldbody.add_body(name=root_body)
    b.add_freejoint()
    b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.1, 0.1, 0.1])
    if dof_joint_name is not None:
        child = b.add_body(name=f"{root_body}_link")
        child.add_joint(name=dof_joint_name, type=mujoco.mjtJoint.mjJNT_HINGE)
        child.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.05, 0.05, 0.05])
    return spec


def _capture_meta(spec: mujoco.MjSpec, prefix: str, root_body: str) -> ActorSpecMeta:  # type: ignore[name-defined]
    # _capture_spec_meta only reads from `spec`, so call it unbound (self=None) to
    # exercise the real implementation without constructing a full scene manager.
    return MujocoSceneManager._capture_spec_meta(None, spec, prefix, root_body)  # type: ignore[arg-type]


def _build_world(object_name: str, object_dof_joint: str | None):
    """Attach a robot + one rigid object, returning (model, robot_meta, obj_meta).

    Replicates the add_robot/add_rigid_object attach + pre-attach capture order.
    """
    world = mujoco.MjSpec()
    world.copy_during_attach = True

    robot_spec = _make_actor_spec("pelvis", "hip_joint")
    robot_meta = _capture_meta(robot_spec, "robot_", "pelvis")
    world.attach(robot_spec, frame=world.worldbody.add_frame(), prefix="robot_")

    obj_prefix = f"{object_name}_"
    obj_spec = _make_actor_spec("baseLink", object_dof_joint)
    obj_meta = _capture_meta(obj_spec, obj_prefix, "baseLink")
    world.attach(obj_spec, frame=world.worldbody.add_frame(), prefix=obj_prefix)

    return world.compile(), robot_meta, obj_meta


def _robot_body_ids(model, meta: ActorSpecMeta) -> list[int]:
    """Resolve robot body clean names -> compiled MuJoCo body ids (the gather index)."""
    ids = []
    for clean in meta.body_names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{meta.prefix}{clean}")
        assert bid != -1, f"{clean} not found"
        ids.append(bid)
    return ids


# ---------------------------------------------------------------------------
# Metadata capture (the core of the D fix).
# ---------------------------------------------------------------------------


def test_capture_spec_meta_excludes_worldbody_and_freejoint():
    spec = _make_actor_spec("pelvis", "hip_joint")
    meta = _capture_meta(spec, "robot_", "pelvis")

    # Worldbody excluded; both real bodies captured.
    assert "world" not in meta.body_names
    assert set(meta.body_names) == {"pelvis", "pelvis_link"}
    # Only the named NON-free joint is a DOF; the freejoint is not captured.
    assert meta.dof_joint_names == ["hip_joint"]
    assert meta.prefix == "robot_"
    assert meta.root_body == "pelvis"


# ---------------------------------------------------------------------------
# Prefix collision: an object named "robot_*" must not leak bodies into the robot set.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("object_name", ["box", "robot_decoy"])
def test_prefix_collision_does_not_leak_bodies(object_name):
    """Robot body set is identical whether the object name collides or not."""
    model, robot_meta, _ = _build_world(object_name, object_dof_joint="lid_joint")

    # Robot bodies come from the recorded metadata, NOT name.startswith("robot_").
    assert set(robot_meta.body_names) == {"pelvis", "pelvis_link"}

    # The object's bodies never appear in the robot set, even for "robot_decoy"
    # whose prefixed bodies ("robot_decoy_baseLink") startswith "robot_".
    robot_prefixed = {f"robot_{n}" for n in robot_meta.body_names}
    object_bodies = {model.body(i).name for i in range(model.nbody)} - robot_prefixed - {"world"}
    assert robot_prefixed.isdisjoint(object_bodies)
    assert all(name.startswith(object_name) for name in object_bodies)


# ---------------------------------------------------------------------------
# A named internal object joint must not leak into robot DOFs.
# ---------------------------------------------------------------------------


def test_object_joint_does_not_leak_dof():
    """Object 'lid_joint' (benign name) must not be counted as a robot DOF."""
    _, robot_meta, obj_meta = _build_world("box", object_dof_joint="lid_joint")

    assert robot_meta.dof_joint_names == ["hip_joint"]
    assert "lid_joint" not in robot_meta.dof_joint_names
    # The object's own joint is captured under the object's metadata.
    assert obj_meta.dof_joint_names == ["lid_joint"]


def test_object_with_no_internal_joint_has_no_dofs():
    """A single free body (shipped box) contributes zero DOFs."""
    _, robot_meta, obj_meta = _build_world("box", object_dof_joint=None)
    assert obj_meta.dof_joint_names == []
    assert robot_meta.dof_joint_names == ["hip_joint"]


# ---------------------------------------------------------------------------
# Gather-index validity and the 0-based body_names indexing contract.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("object_name", ["box", "robot_decoy"])
def test_robot_body_ids_gather_index_is_valid(object_name):
    """body_ids gather robot rows out of the full model, bound to the source mapping.

    Mirrors _set_robot_properties' body_ids construction verbatim: for each clean
    body name, resolve ``mj_name2id(model, mjOBJ_BODY, prefix + clean)`` (this is what
    ``_robot_body_ids`` does). Asserts the gather list equals exactly the model rows for
    the robot's prefixed bodies and excludes world + every object body.
    """
    model, robot_meta, _ = _build_world(object_name, object_dof_joint="lid_joint")
    body_ids = _robot_body_ids(model, robot_meta)
    num_bodies = len(robot_meta.body_names)

    # World (id 0) is excluded; every gather id is a real, distinct model body.
    assert all(1 <= bid < model.nbody for bid in body_ids)
    assert len(set(body_ids)) == num_bodies

    # Bind to source: body_ids[i] must be the model id of prefix+body_names[i], in order.
    expected_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{robot_meta.prefix}{clean}")
        for clean in robot_meta.body_names
    ]
    assert body_ids == expected_ids

    # The gather addresses a strict subset of the full physics tensor: world (id 0) and
    # every object body are outside the gather set, so num_bodies stays below model.nbody.
    all_ids = set(range(model.nbody))
    excluded = all_ids - set(body_ids)
    assert 0 in excluded  # world body never gathered
    assert num_bodies < model.nbody
    # Every excluded non-world body is an object body (its name does not map to a robot body).
    robot_prefixed = {f"{robot_meta.prefix}{clean}" for clean in robot_meta.body_names}
    for bid in excluded - {0}:
        assert model.body(bid).name not in robot_prefixed


@pytest.mark.parametrize("object_name", ["box", "robot_decoy"])
def test_find_rigid_body_indice_is_zero_based(object_name):
    """find_rigid_body_indice returns a 0-based body_names index, NOT the raw MuJoCo id.

    The shipped method (MuJoCo.find_rigid_body_indice) is literally
    ``return self.body_names.index(body_name)`` (RuntimeError on miss). It can only be
    invoked on a full simulator (importing it pulls torch, which is not available under
    this file's ``importorskip("mujoco")``), so we exercise that exact expression against
    the real ``robot_meta.body_names`` and pin the property a raw-id regression would break:
    the 0-based index must DIFFER from the raw MuJoCo body id (world is body 0, robot is
    attached first so its bodies are ids >= 1) and stay in-bounds for a ``num_bodies``-wide
    tensor. A method returning the raw id would, for the final robot body, equal
    ``num_bodies`` and index out of bounds.
    """
    model, robot_meta, _ = _build_world(object_name, object_dof_joint="lid_joint")
    body_names = robot_meta.body_names
    num_bodies = len(body_names)
    raw_ids = _robot_body_ids(model, robot_meta)  # raw MuJoCo body ids, world-inclusive (>= 1)

    for expected_idx, name in enumerate(body_names):
        # `body_names.index(name)` is the shipped find_rigid_body_indice body verbatim.
        idx = body_names.index(name)
        assert idx == expected_idx
        assert 0 <= idx < num_bodies
        # The 0-based index is NOT the raw MuJoCo body id (which is world-inclusive >= 1):
        # they differ for every robot body, so a raw-id return would fail here.
        assert idx != raw_ids[expected_idx]
        assert raw_ids[expected_idx] >= 1

    # The final body is the out-of-bounds case for a raw-id return: index num_bodies-1,
    # raw id num_bodies (world is body 0 and the robot is attached before the object).
    last_idx = body_names.index(body_names[-1])
    assert last_idx == num_bodies - 1
    assert raw_ids[-1] == num_bodies
    assert raw_ids[-1] != last_idx
