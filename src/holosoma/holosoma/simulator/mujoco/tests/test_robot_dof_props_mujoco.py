"""``apply_dof_properties_from_config`` / ``read_dof_limits``: the config's per-DOF lists land on the compiled model
and the limit tensors are read back from it (pure).

Compiles a tiny inline robot through the same isolated-spec -> capture meta -> attach-with-prefix path
``MujocoSceneManager.add_robot`` uses, with MJCF-authored ``armature``/``frictionloss`` that differ from
the config lists, and asserts the config values win on the robot DOFs only.
"""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

if TYPE_CHECKING:
    import mujoco

    from holosoma.config_types.robot import RobotConfig
else:
    # no_sim lane runs in an env without the mujoco package (isaacsim conda); skip when unavailable.
    mujoco = pytest.importorskip("mujoco")

from loguru import logger

from holosoma.simulator.mujoco.robot_addressing import (
    apply_dof_properties_from_config,
    build_name_maps,
    read_dof_limits,
    validate_dof_actuators,
)
from holosoma.simulator.mujoco.scene_manager import ActorSpecMeta, MujocoSceneManager

pytestmark = pytest.mark.no_sim

_PREFIX = "robot_"
_DOFS = ["j0", "j1", "j2"]
_MJCF = """
<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <body name="base"><geom size="0.1"/>
      <body name="link0">
        <joint name="j0" type="hinge" axis="0 0 1" range="-1 1" armature="0.5" frictionloss="0.2"/><geom size="0.1"/>
        <body name="link1">
          <joint name="j1" type="slide" axis="0 0 1" armature="0.5" frictionloss="0.2"/><geom size="0.1"/>
          <body name="link2">
            <joint name="j2" type="hinge" axis="0 0 1" armature="0.5" frictionloss="0.2"/><geom size="0.1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="j0" joint="j0" ctrlrange="-5 5"/><motor name="j1" joint="j1" ctrlrange="-20 20" forcerange="-2 2"/>
    <motor name="j2" joint="j2"/>
  </actuator>
</mujoco>
"""
_JOINT_CLAMPED_MJCF = (
    _MJCF.replace('<joint name="j0"', '<joint name="j0" actuatorfrcrange="-2 3"')
    .replace('<joint name="j1"', '<joint name="j1" actuatorfrcrange="-1 1"')
    .replace('<joint name="j2"', '<joint name="j2" actuatorfrcrange="-4 4"')
    # Deliberately put the last joint's actuator first.
    .replace('<motor name="j0"', '<motor name="j2" joint="j2"/><motor name="j0"')
    .replace('    <motor name="j2" joint="j2"/>\n', "")
)


def _compile_robot(mjcf: str = _MJCF) -> tuple[mujoco.MjModel, ActorSpecMeta]:
    robot_spec = mujoco.MjSpec.from_string(mjcf)
    root_body = robot_spec.worldbody.bodies[0]
    root_body.add_freejoint()
    # _capture_spec_meta reads no instance state, so call it unbound (self=None).
    meta = MujocoSceneManager._capture_spec_meta(None, robot_spec, _PREFIX, str(root_body.name))  # type: ignore[arg-type]
    world = mujoco.MjSpec()
    world.copy_during_attach = True
    world.attach(robot_spec, frame=world.worldbody.add_frame(), prefix=_PREFIX)
    return world.compile(), meta


def _robot(
    dof_names: list[str],
    armature: list[float] | None = None,
    friction: list[float] | None = None,
    *,
    lower: list[float] | None = None,
    upper: list[float] | None = None,
    velocity: list[float] | None = None,
    effort: list[float] | None = None,
) -> RobotConfig:
    """Duck-typed stand-in for the RobotConfig fields read by the DOF-property helpers."""
    positions = None
    if lower is not None:
        assert upper is not None
        positions = dict(zip(dof_names, zip(lower, upper)))
    return cast(
        "RobotConfig",
        SimpleNamespace(
            dof_names=dof_names,
            joint_tags={},
            dof_pos_limits=positions,
            dof_vel_limits=None if velocity is None else dict(zip(dof_names, velocity)),
            dof_effort_limits=None if effort is None else dict(zip(dof_names, effort)),
            dof_armature=None if armature is None else dict(zip(dof_names, armature)),
            dof_joint_friction=None if friction is None else dict(zip(dof_names, friction)),
            default_dof_positions=lambda: dict.fromkeys(dof_names, 0.0),
        ),
    )


def _dof_addrs(model: mujoco.MjModel, meta: ActorSpecMeta) -> list[int]:
    clean_to_prefixed, _ = build_name_maps(meta)
    return [
        int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, clean_to_prefixed[d])]) for d in _DOFS
    ]


@pytest.fixture
def info_logs() -> Iterator[list[str]]:
    msgs: list[str] = []
    sink_id = logger.add(lambda m: msgs.append(m.record["message"]), level="INFO")
    yield msgs
    logger.remove(sink_id)


def test_config_values_override_authored_values_on_robot_dofs_only(info_logs: list[str]) -> None:
    model, meta = _compile_robot()
    armature, friction = [0.01, 0.02, 0.03], [0.1, 0.0, 0.3]

    apply_dof_properties_from_config(model, meta, _robot(_DOFS, armature, friction))

    addrs = _dof_addrs(model, meta)
    np.testing.assert_allclose(model.dof_armature[addrs], armature)
    np.testing.assert_allclose(model.dof_frictionloss[addrs], friction)
    # The root freejoint's 6 DOFs keep their defaults.
    free_addrs = [i for i in range(model.nv) if i not in addrs]
    assert len(free_addrs) == 6
    assert np.all(model.dof_armature[free_addrs] == 0.0)
    assert np.all(model.dof_frictionloss[free_addrs] == 0.0)
    # Every authored value differed, so both properties are reported (j1 friction 0.2 -> 0 included).
    assert any("dof_armature: config overrides the MJCF value on 3/3 DOFs" in m for m in info_logs)
    assert any("dof_frictionloss: config overrides the MJCF value on 3/3 DOFs" in m for m in info_logs)


def test_matching_values_are_not_reported(info_logs: list[str]) -> None:
    model, meta = _compile_robot()
    apply_dof_properties_from_config(model, meta, _robot(_DOFS, [0.5] * 3, [0.2] * 3))
    assert not [m for m in info_logs if "config overrides the MJCF value" in m]


def test_zero_dof_robot_is_a_noop() -> None:
    model, meta = _compile_robot()
    before = model.dof_armature.copy()
    apply_dof_properties_from_config(model, meta, _robot([], [], []))
    np.testing.assert_array_equal(model.dof_armature, before)


def test_unknown_dof_name_raises() -> None:
    model, meta = _compile_robot()
    with pytest.raises(ValueError, match="Robot joint 'nope' has no asset matches"):
        apply_dof_properties_from_config(model, meta, _robot(["nope"], [0.0], [0.0]))


def _joint_ids(model: mujoco.MjModel, meta: ActorSpecMeta) -> list[int]:
    clean_to_prefixed, _ = build_name_maps(meta)
    return [int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, clean_to_prefixed[d])) for d in _DOFS]


def test_none_lists_keep_the_authored_values(info_logs: list[str]) -> None:
    model, meta = _compile_robot(_JOINT_CLAMPED_MJCF)
    before = {
        "jnt_range": model.jnt_range.copy(),
        "jnt_limited": model.jnt_limited.copy(),
        "actuator_ctrlrange": model.actuator_ctrlrange.copy(),
        "actuator_ctrllimited": model.actuator_ctrllimited.copy(),
        "actuator_forcerange": model.actuator_forcerange.copy(),
        "actuator_forcelimited": model.actuator_forcelimited.copy(),
        "jnt_actfrcrange": model.jnt_actfrcrange.copy(),
        "jnt_actfrclimited": model.jnt_actfrclimited.copy(),
        "dof_armature": model.dof_armature.copy(),
        "dof_frictionloss": model.dof_frictionloss.copy(),
    }

    apply_dof_properties_from_config(model, meta, _robot(_DOFS))

    for name, expected in before.items():
        np.testing.assert_array_equal(getattr(model, name), expected)
    assert not [m for m in info_logs if "config overrides" in m]


def test_config_limits_land_on_jnt_range_and_ctrlrange(info_logs: list[str]) -> None:
    model, meta = _compile_robot()
    joints = _joint_ids(model, meta)
    assert list(model.jnt_limited[joints]) == [1, 0, 0]  # only j0 authored a range

    apply_dof_properties_from_config(
        model, meta, _robot(_DOFS, lower=[-2.0, 0.0, -0.5], upper=[2.0, 0.1, 0.5], effort=[10.0, 20.0, 30.0])
    )

    np.testing.assert_allclose(model.jnt_range[joints], [[-2.0, 2.0], [0.0, 0.1], [-0.5, 0.5]])
    assert list(model.jnt_limited[joints]) == [1, 1, 1]
    actuators = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{_PREFIX}{d}") for d in _DOFS]
    np.testing.assert_allclose(model.actuator_ctrlrange[actuators], [[-10, 10], [-20, 20], [-30, 30]])
    assert list(model.actuator_ctrllimited[actuators]) == [1, 1, 1]
    # The force clamp follows too: j1's authored forcerange +-2 would otherwise cap the configured 20.
    np.testing.assert_allclose(model.actuator_forcerange[actuators], [[-10, 10], [-20, 20], [-30, 30]])
    assert list(model.actuator_forcelimited[actuators]) == [1, 1, 1]
    # j0 authored [-1, 1] and ctrlrange 5; the unlimited joints/motors count as changed too (inf -> value).
    assert any("jnt_range lower: config overrides the MJCF value on 3/3 DOFs" in m for m in info_logs)
    assert any("actuator_ctrlrange (effort): config overrides the MJCF value on 3/3 DOFs" in m for m in info_logs)


def test_read_dof_limits_reports_the_model_values() -> None:
    model, meta = _compile_robot()
    inf = float("inf")

    position, velocity, effort = read_dof_limits(model, meta, _robot(_DOFS))

    assert position == [[-1.0, 1.0], [-inf, inf], [-inf, inf]]  # unlimited joints read as +-inf
    assert velocity == [inf, inf, inf]  # MuJoCo has no joint velocity limit
    assert effort == [5.0, 2.0, inf]  # the tightest active clamp: j1's forcerange 2 beats its ctrlrange 20


def test_read_dof_limits_reports_effort_available_in_both_directions() -> None:
    model, meta = _compile_robot()
    actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{_PREFIX}j1")
    model.actuator_ctrlrange[actuator] = [-2.0, 10.0]
    model.actuator_ctrllimited[actuator] = 1
    model.actuator_forcerange[actuator] = [-4.0, 8.0]
    model.actuator_forcelimited[actuator] = 1

    _, _, effort = read_dof_limits(model, meta, _robot(_DOFS))

    assert effort[1] == 2.0  # negative commands are ctrl-limited to 2, even though positive commands can reach 8


def test_read_dof_limits_after_apply_returns_the_config_values() -> None:
    model, meta = _compile_robot()
    robot = _robot(
        _DOFS, lower=[-2.0, 0.0, -0.5], upper=[2.0, 0.1, 0.5], velocity=[1.0, 2.0, 3.0], effort=[10.0, 20.0, 30.0]
    )
    apply_dof_properties_from_config(model, meta, robot)

    position, velocity, effort = read_dof_limits(model, meta, robot)

    assert position == [[-2.0, 2.0], [0.0, 0.1], [-0.5, 0.5]]
    assert velocity == [1.0, 2.0, 3.0]
    assert effort == [10.0, 20.0, 30.0]


@pytest.mark.parametrize("effort", [None, [20.0, 30.0, 10.0], [0.0, 0.0, 0.0]])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_joint_force_clamps_match_readback_and_applied_effort(effort: list[float] | None, sign: float) -> None:
    model, meta = _compile_robot(_JOINT_CLAMPED_MJCF)
    # Public order differs from both joint and actuator order; the first DOF is linear.
    robot = _robot(
        ["j1", "j2", "j0"],
        effort=effort,
    )
    free_before = model.jnt_actfrcrange[0].copy()
    apply_dof_properties_from_config(model, meta, robot)

    _, _, reported = read_dof_limits(model, meta, robot)
    assert reported == (effort if effort is not None else [1.0, 4.0, 2.0])
    np.testing.assert_array_equal(model.jnt_actfrcrange[0], free_before)
    assert model.jnt_actfrclimited[0] == 0

    data = mujoco.MjData(model)
    data.ctrl[:] = sign * 100.0
    mujoco.mj_forward(model, data)
    # Read actual generalized forces in original asset order, independently of config order.
    forces = data.qfrc_actuator[_dof_addrs(model, meta)]
    expected = [10.0, 20.0, 30.0] if effort is not None else [2.0 if sign < 0 else 3.0, 1.0, 4.0]
    if effort == [0.0, 0.0, 0.0]:
        expected = [0.0, 0.0, 0.0]
    np.testing.assert_allclose(forces, sign * np.array(expected))


def test_inactive_joint_force_clamp_does_not_reduce_effort() -> None:
    model, meta = _compile_robot(_JOINT_CLAMPED_MJCF)
    joints = _joint_ids(model, meta)
    model.jnt_actfrclimited[joints] = 0
    _, _, effort = read_dof_limits(model, meta, _robot(_DOFS))
    assert effort == [5.0, 2.0, float("inf")]


def test_one_way_joint_force_clamp_requires_an_override() -> None:
    model, meta = _compile_robot(_JOINT_CLAMPED_MJCF)
    model.jnt_actfrcrange[_joint_ids(model, meta)[0]] = [2.0, 4.0]
    names, _ = build_name_maps(meta)
    with pytest.raises(ValueError, match="joint actuatorfrcrange.*does not provide effort in both directions"):
        validate_dof_actuators(model, meta, _DOFS, names)
    apply_dof_properties_from_config(model, meta, _robot(_DOFS, effort=[10.0, 20.0, 30.0]))
    assert len(validate_dof_actuators(model, meta, _DOFS, names)) == 3


@pytest.mark.parametrize("tagged", [False, True])
def test_sparse_overrides_follow_joint_identity_and_preserve_other_asset_values(tagged: bool) -> None:
    model, meta = _compile_robot()
    reference, reference_meta = _compile_robot()
    names = ["j2", "j0", "j1"]
    robot: Any = _robot(names)
    robot.joint_tags = {"#selected": ["j2", "j0", "j2"]}
    robot.dof_armature = {"#selected": 0.37} if tagged else {"j2": 0.37, "j0": 0.37}
    robot.dof_effort_limits = {"j1": 7.0}
    robot.dof_pos_limits = {"j0": (-0.4, 0.6)}
    before_armature = reference.dof_armature.copy()
    before_limits = read_dof_limits(reference, reference_meta, _robot(names))
    apply_dof_properties_from_config(model, meta, robot)
    positions, velocities, efforts = read_dof_limits(model, meta, robot)
    assert positions[0] == before_limits[0][0]  # j2 omitted from position overrides
    assert positions[1] == [-0.4, 0.6]
    assert efforts[0:2] == before_limits[2][0:2]
    assert efforts[2] == 7.0
    addresses = _dof_addrs(model, meta)
    assert model.dof_armature[addresses[0]] == model.dof_armature[addresses[2]] == 0.37
    assert model.dof_armature[addresses[1]] == before_armature[addresses[1]]
    np.testing.assert_array_equal(model.dof_frictionloss, reference.dof_frictionloss)


def test_invalid_later_property_prevents_all_model_writes() -> None:
    model, meta = _compile_robot()
    robot: Any = _robot(_DOFS)
    robot.joint_tags = {"#pair": ["j0", "j1"]}
    robot.dof_pos_limits = {"j2": (-0.1, 0.2)}
    robot.dof_effort_limits = {"j2": 3.0}
    robot.dof_joint_friction = {"#pair": 0.1, "j0": 0.1}
    fields = ("jnt_range", "actuator_ctrlrange", "dof_armature", "dof_frictionloss")
    before = {field: getattr(model, field).copy() for field in fields}
    with pytest.raises(ValueError, match="keys '#pair' and 'j0' both cover DOF 'j0'"):
        apply_dof_properties_from_config(model, meta, robot)
    for field in fields:
        np.testing.assert_array_equal(getattr(model, field), before[field])
