"""Fixed-base robot spec handling — pure MJCF-string tests, no simulator build.

Mirrors ``MujocoSceneManager.add_robot``'s isolated-spec path on a tiny 2-hinge + 1-slide arm:
``_ensure_single_root_body`` applies the one root-joint rule (``_set_root_joint``) shared by the
robot, rigid-object and scene-file paths -- a welded body has no root freejoint (an authored one is
dropped), a free body has exactly one -- and ``resolve_actor_addressing`` classifies a static robot
by root body id instead of resolving a freejoint, rejecting a static actor whose root body has a
joint. The rigid-object case drives the real ``add_rigid_object`` on a temp MJCF.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

    import mujoco

    from holosoma.config_types.simulator import SimulatorConfig
else:
    # no_sim lane runs in an env without the mujoco package (isaacsim conda); skip when unavailable.
    mujoco = pytest.importorskip("mujoco")

from holosoma.config_types.scene import ObjectPatternConfig, RigidObjectConfig, SceneFileConfig
from holosoma.simulator.mujoco.robot_addressing import resolve_actor_addressing, validate_robot_joint_types
from holosoma.simulator.mujoco.scene_manager import MujocoSceneManager

pytestmark = pytest.mark.no_sim

_PREFIX = "robot_"
_ARM = "arm"

# Root body ``base`` with three 1-DOF joints below it (hinge, slide, hinge): nq == 3 when welded.
_ARM_LINKS = """
    <body name="base">{root_joint}<geom size="0.1"/>
      <body name="link0" pos="0 0 0.2"><joint name="j0" type="hinge" axis="0 0 1"/><geom size="0.1"/>
        <body name="link1" pos="0 0 0.2"><joint name="j1" type="slide" axis="0 0 1"/><geom size="0.1"/>
          <body name="link2" pos="0 0 0.2"><joint name="j2" type="hinge" axis="0 1 0"/><geom size="0.1"/></body>
        </body>
      </body>
    </body>
"""
_MOTORS = '<motor name="j0" joint="j0"/><motor name="j1" joint="j1"/><motor name="j2" joint="j2"/>'


def _arm_mjcf(root_joint: str = "") -> str:
    return (
        f"<mujoco><worldbody>{_ARM_LINKS.format(root_joint=root_joint)}</worldbody>"
        f"<actuator>{_MOTORS}</actuator></mujoco>"
    )


def _ensure_root(spec: mujoco.MjSpec, *, free: bool) -> str:
    return MujocoSceneManager._ensure_single_root_body(spec, _ARM, free=free)


def _attach(robot_spec: mujoco.MjSpec) -> mujoco.MjModel:
    """Attach with a prefix at a unit frame and compile, as ``add_robot`` does."""
    validate_robot_joint_types(robot_spec, _ARM)
    world = mujoco.MjSpec()
    world.copy_during_attach = True
    world.attach(robot_spec, frame=world.worldbody.add_frame(), prefix=_PREFIX)
    return world.compile()


def _free_joint_count(model: mujoco.MjModel) -> int:
    return int((model.jnt_type == mujoco.mjtJoint.mjJNT_FREE).sum())


def _scene_manager() -> MujocoSceneManager:
    """A real scene manager; its constructor reads only ``simulator_config.sim.fps``."""
    return MujocoSceneManager(cast("SimulatorConfig", SimpleNamespace(sim=SimpleNamespace(fps=200))))


@pytest.mark.parametrize("root_joint", ["<freejoint/>", '<joint type="free"/>'], ids=["freejoint", "joint-free"])
def test_fixed_base_drops_authored_root_freejoint(root_joint: str) -> None:
    spec = mujoco.MjSpec.from_string(_arm_mjcf(root_joint))
    assert any(j.type == mujoco.mjtJoint.mjJNT_FREE for j in spec.worldbody.bodies[0].joints)

    root = _ensure_root(spec, free=False)
    assert root == "base"
    model = _attach(spec)
    assert model.nq == 3
    assert model.nv == 3
    assert _free_joint_count(model) == 0
    # The root body is welded: a direct child of the world with no joint of its own.
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{_PREFIX}base")
    assert model.body_parentid[base_id] == 0
    assert model.body_jntnum[base_id] == 0


def test_welding_drops_the_keyframe_authored_for_the_freejoint() -> None:
    # A keyframe's qpos is sized for the authored freejoint (7 + 3); after welding it would fail to compile.
    xml = _arm_mjcf("<freejoint/>").replace(
        "</mujoco>", '<keyframe><key name="home" qpos="0 0 1 1 0 0 0 0 0 0"/></keyframe></mujoco>'
    )
    welded = mujoco.MjSpec.from_string(xml)
    _ensure_root(welded, free=False)
    model = _attach(welded)
    assert model.nq == 3
    assert model.nkey == 0
    # A floating base keeps the authored freejoint and its keyframe.
    floating = mujoco.MjSpec.from_string(xml)
    _ensure_root(floating, free=True)
    model = _attach(floating)
    assert model.nq == 10
    assert model.nkey == 1


def test_fixed_base_without_authored_freejoint_stays_welded() -> None:
    spec = mujoco.MjSpec.from_string(_arm_mjcf())
    _ensure_root(spec, free=False)
    model = _attach(spec)
    assert model.nq == 3
    assert _free_joint_count(model) == 0


@pytest.mark.parametrize("root_joint", ["", "<freejoint/>"], ids=["added", "authored"])
def test_floating_base_gets_exactly_one_freejoint(root_joint: str) -> None:
    spec = mujoco.MjSpec.from_string(_arm_mjcf(root_joint))
    _ensure_root(spec, free=True)
    model = _attach(spec)
    assert _free_joint_count(model) == 1
    assert model.nq == 7 + 3
    assert model.nv == 6 + 3


@pytest.mark.parametrize("root_joint", ["", "<freejoint/>"], ids=["jointless", "authored-freejoint"])
@pytest.mark.parametrize("fixed", [True, False], ids=["fixed", "free"])
def test_rigid_object_root_joint_follows_fixed(tmp_path: Path, root_joint: str, fixed: bool) -> None:
    # The real add_rigid_object path: a fixed object compiles welded (an authored freejoint is dropped)
    # with its pose baked into the attach frame; a free object has exactly one freejoint (authored or added).
    asset = tmp_path / "arm.xml"
    asset.write_text(_arm_mjcf(root_joint))
    position = [1.0, 2.0, 3.0]
    yaw90_wxyz = [0.7071068, 0.0, 0.0, 0.7071068]  # config orientation is [w, x, y, z]
    cfg = RigidObjectConfig(xml_file=str(asset), fixed=fixed, position=position, orientation=yaw90_wxyz)

    scene_manager = _scene_manager()
    scene_manager.add_rigid_object("obj", cfg, ["xml"])
    assert scene_manager.rigid_object_root_bodies == {"obj": "obj_base"}
    model = scene_manager.compile()

    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obj_base")
    assert model.body_parentid[base_id] == 0
    if fixed:
        assert _free_joint_count(model) == 0
        assert model.body_jntnum[base_id] == 0
        assert model.nq == 3
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(data.xpos[base_id], position)
        # xquat is wxyz as well; a config read as xyzw would come back as a 90 deg roll [w, w, 0, 0].
        np.testing.assert_allclose(data.xquat[base_id], yaw90_wxyz, atol=1e-6)
    else:
        assert _free_joint_count(model) == 1
        assert model.body_jntnum[base_id] == 1
        assert model.nq == 7 + 3
        # A free object attaches at the unit frame; its configured pose is written later through the freejoint.
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(data.xpos[base_id], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(data.xquat[base_id], [1.0, 0.0, 0.0, 0.0])


class _AddressingSceneManager:
    """The three attributes ``resolve_actor_addressing`` reads off a ``MujocoSceneManager``."""

    def __init__(self, robot_root_body: str) -> None:
        self.robot_root_body = robot_root_body
        self.rigid_object_root_bodies: dict[str, str] = {}
        self.scene_file_bodies: dict[str, tuple[str, bool]] = {}


def test_static_robot_addressed_by_body_id_not_freejoint() -> None:
    spec = mujoco.MjSpec.from_string(_arm_mjcf("<freejoint/>"))
    _ensure_root(spec, free=False)
    model = _attach(spec)
    scene_manager = cast("MujocoSceneManager", _AddressingSceneManager(f"{_PREFIX}base"))

    object_addrs, static_object_body_ids = resolve_actor_addressing(model, scene_manager, {"robot"})
    assert "robot" not in object_addrs
    assert static_object_body_ids == {"robot": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{_PREFIX}base")}


def test_static_actor_with_root_joint_fails_loud() -> None:
    # The authored freejoint is left in place (no _ensure_single_root_body): a static classification
    # of a body that can still move must be rejected, not addressed by body id.
    spec = mujoco.MjSpec.from_string(_arm_mjcf("<freejoint/>"))
    model = _attach(spec)
    assert _free_joint_count(model) == 1
    scene_manager = cast("MujocoSceneManager", _AddressingSceneManager(f"{_PREFIX}base"))
    with pytest.raises(ValueError, match=r"Static actor 'robot' has 1 joint\(s\) on its root body 'robot_base'"):
        resolve_actor_addressing(model, scene_manager, {"robot"})


def test_welded_robot_not_in_static_names_fails_loud() -> None:
    # Without the static classification the resolver demands a freejoint; a welded robot has none.
    spec = mujoco.MjSpec.from_string(_arm_mjcf())
    _ensure_root(spec, free=False)
    model = _attach(spec)
    scene_manager = cast("MujocoSceneManager", _AddressingSceneManager(f"{_PREFIX}base"))
    with pytest.raises(ValueError, match="No freejoint on root body"):
        resolve_actor_addressing(model, scene_manager, set())


def test_floating_robot_addressing_unchanged() -> None:
    spec = mujoco.MjSpec.from_string(_arm_mjcf())
    _ensure_root(spec, free=True)
    model = _attach(spec)
    scene_manager = cast("MujocoSceneManager", _AddressingSceneManager(f"{_PREFIX}base"))

    object_addrs, static_object_body_ids = resolve_actor_addressing(model, scene_manager, set())
    assert static_object_body_ids == {}
    assert object_addrs == {"robot": {"qpos_addr": 0, "qvel_addr": 0}}


# Two top-level bodies: 'free' authors a root freejoint (structurally free), 'post' is jointless (static).
_SCENE_FILE_MJCF = """
<mujoco><worldbody>
  <body name="free" pos="0 0 1"><freejoint/><geom size="0.1"/></body>
  <body name="post" pos="1 0 0.5"><geom size="0.1"/></body>
</worldbody></mujoco>
"""


def _scene_file_model(
    tmp_path: Path, object_configs: dict[str, ObjectPatternConfig] | None
) -> tuple[mujoco.MjModel, MujocoSceneManager]:
    """Drive the real ``add_scene_file`` + ``compile`` on the two-body file; the robot slot stays unused."""
    asset = tmp_path / "scene.xml"
    asset.write_text(_SCENE_FILE_MJCF)
    scene_manager = _scene_manager()
    scene_manager.add_scene_file("s", SceneFileConfig(xml_path=str(asset), object_configs=object_configs), ["xml"])
    return scene_manager.compile(), scene_manager


def _body_joints(model: mujoco.MjModel, name: str) -> int:
    return int(model.body_jntnum[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)])


def test_scene_file_bodies_follow_their_authored_structure(tmp_path: Path) -> None:
    model, scene_manager = _scene_file_model(tmp_path, object_configs=None)

    assert scene_manager.scene_file_bodies == {"s_free": ("s_free", False), "s_post": ("s_post", True)}
    assert _free_joint_count(model) == 1 and model.nq == 7
    assert _body_joints(model, "s_free") == 1 and _body_joints(model, "s_post") == 0


def test_scene_file_static_override_drops_an_authored_root_freejoint(tmp_path: Path) -> None:
    # The same root-joint rule as the robot / rigid-object paths: a body forced static ends up welded, so the
    # static classification and the compiled joint state agree (the body cannot fall while addressed by id).
    model, scene_manager = _scene_file_model(tmp_path, {"free": ObjectPatternConfig(fixed=True)})

    assert scene_manager.scene_file_bodies == {"s_free": ("s_free", True), "s_post": ("s_post", True)}
    assert _free_joint_count(model) == 0 and model.nq == 0
    assert _body_joints(model, "s_free") == 0
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "s_free")], [0, 0, 1])


def test_scene_file_free_override_adds_a_root_freejoint(tmp_path: Path) -> None:
    model, scene_manager = _scene_file_model(tmp_path, {"post": ObjectPatternConfig(fixed=False)})

    assert scene_manager.scene_file_bodies == {"s_free": ("s_free", False), "s_post": ("s_post", False)}
    assert _free_joint_count(model) == 2 and model.nq == 14
    assert _body_joints(model, "s_post") == 1
