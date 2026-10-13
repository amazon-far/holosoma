"""Robot articulation contract checks (``robot_addressing.validate_*``) — pure, no simulator build.

Compiles tiny inline MJCF robots through the same isolated-spec -> capture meta -> attach-with-prefix
path ``MujocoSceneManager.add_robot`` uses, then asserts each rule documented in ``robot_addressing``:
named 1-DOF joints only (hinge/slide; the unnamed root freejoint is exempt), one pure torque motor per
DOF resolved by name (any actuator order, extra actuators allowed and left alone; a DOF without one
gets a motor at load), distinct scalar DOF qpos/qvel addresses in any config order.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import mujoco
else:
    # no_sim lane runs in an env without the mujoco package (isaacsim conda); skip when unavailable.
    mujoco = pytest.importorskip("mujoco")

from loguru import logger

from holosoma.simulator.mujoco.robot_addressing import (
    add_missing_dof_motors,
    build_name_maps,
    resolve_dof_addresses,
    validate_dof_actuators,
    validate_dof_addressing,
    validate_robot_joint_types,
)
from holosoma.simulator.mujoco.scene_manager import ActorSpecMeta, MujocoSceneManager

pytestmark = pytest.mark.no_sim

_PREFIX = "robot_"
_ROBOT = "test_robot"

# Root body ``base`` (no joint here; the real path adds the freejoint), then one child link per joint.
# j1 is a slide (prismatic) joint: explicitly allowed by the contract.
_LINKS = """
    <body name="base"><geom size="0.1"/>
      <body name="link0"><joint name="j0" type="hinge" axis="0 0 1"/><geom size="0.1"/>
        <body name="link1"><joint name="j1" type="slide" axis="0 0 1"/><geom size="0.1"/>
          <body name="link2"><joint name="j2" type="hinge" axis="0 0 1"/><geom size="0.1"/></body>
        </body>
      </body>
    </body>
"""
_MOTORS = '<motor name="j0" joint="j0"/><motor name="j1" joint="j1"/><motor name="j2" joint="j2"/>'


def _mjcf(worldbody: str, actuators: str = "", extra: str = "") -> str:
    return f"<mujoco><worldbody>{worldbody}</worldbody>{extra}<actuator>{actuators}</actuator></mujoco>"


def _compile_robot(robot_xml: str) -> tuple[mujoco.MjModel, ActorSpecMeta, dict[str, str]]:
    """Mirror ``add_robot``: add the root freejoint, capture meta on the isolated spec, attach with a prefix."""
    robot_spec = mujoco.MjSpec.from_string(robot_xml)
    root_body = robot_spec.worldbody.bodies[0]
    root_body.add_freejoint()
    validate_robot_joint_types(robot_spec, _ROBOT)
    # _capture_spec_meta reads no instance state, so call it unbound (self=None).
    meta = MujocoSceneManager._capture_spec_meta(None, robot_spec, _PREFIX, str(root_body.name))  # type: ignore[arg-type]
    clean_to_prefixed, _ = build_name_maps(meta)
    world = mujoco.MjSpec()
    world.copy_during_attach = True
    frame = world.worldbody.add_frame()
    world.attach(robot_spec, frame=frame, prefix=_PREFIX)
    return world.compile(), meta, clean_to_prefixed


def _check_actuators(actuators: str, extra: str = "") -> list[int]:
    model, meta, clean_to_prefixed = _compile_robot(_mjcf(_LINKS, actuators, extra))
    return validate_dof_actuators(model, meta, meta.dof_joint_names, clean_to_prefixed)


@pytest.fixture
def warnings() -> Iterator[list[str]]:
    """Capture loguru WARNING messages (loguru does not feed pytest's caplog)."""
    msgs: list[str] = []
    sink_id = logger.add(lambda m: msgs.append(m.record["message"]), level="WARNING")
    yield msgs
    logger.remove(sink_id)


# --- validate_robot_joint_types ---------------------------------------------------------------------


def test_joint_types_reject_ball_joint() -> None:
    spec = mujoco.MjSpec.from_string(
        _mjcf(
            '<body name="base"><geom size="0.1"/><body name="link0"><joint name="j0" type="ball"/>'
            '<geom size="0.1"/></body></body>'
        )
    )
    with pytest.raises(ValueError, match=r"'j0', 'link0', 'ball'"):
        validate_robot_joint_types(spec, _ROBOT)


def test_joint_types_reject_unnamed_hinge() -> None:
    spec = mujoco.MjSpec.from_string(
        _mjcf(
            '<body name="base"><geom size="0.1"/><body name="link0"><joint type="hinge"/>'
            '<geom size="0.1"/></body></body>'
        )
    )
    with pytest.raises(ValueError, match=r"'<unnamed>', 'link0', 'hinge'"):
        validate_robot_joint_types(spec, _ROBOT)


def test_joint_types_accept_unnamed_freejoint_root() -> None:
    # t1 shape: an explicit, unnamed <freejoint/> on the root body plus named hinges.
    spec = mujoco.MjSpec.from_string(
        _mjcf(
            '<body name="base"><freejoint/><geom size="0.1"/><body name="link0">'
            '<joint name="j0" type="hinge"/><geom size="0.1"/></body></body>'
        )
    )
    assert [j.name for j in spec.joints if j.type == mujoco.mjtJoint.mjJNT_FREE] == [""]
    validate_robot_joint_types(spec, _ROBOT)


def test_joint_types_accept_hinge_and_slide() -> None:
    spec = mujoco.MjSpec.from_string(_mjcf(_LINKS))
    spec.worldbody.bodies[0].add_freejoint()  # what _ensure_single_root_body does (unnamed)
    validate_robot_joint_types(spec, _ROBOT)


# --- validate_dof_actuators -------------------------------------------------------------------------


def test_actuators_accept_motor_per_dof_including_slide() -> None:
    model, meta, clean_to_prefixed = _compile_robot(_mjcf(_LINKS, _MOTORS))
    assert meta.dof_joint_names == ["j0", "j1", "j2"]
    assert model.nu == 3
    assert validate_dof_actuators(model, meta, meta.dof_joint_names, clean_to_prefixed) == [0, 1, 2]
    # The same model also satisfies the contiguous-addressing rule (DOFs follow the 7/6-wide freejoint).
    qpos_addrs, qvel_addrs = resolve_dof_addresses(model, meta.dof_joint_names, clean_to_prefixed)
    assert (qpos_addrs, qvel_addrs) == ([7, 8, 9], [6, 7, 8])
    validate_dof_addressing(qpos_addrs, qvel_addrs)
    data = mujoco.MjData(model)
    data.ctrl[:] = [3.0, -2.0, 1.0]
    mujoco.mj_forward(model, data)
    assert data.qfrc_actuator[qvel_addrs] == pytest.approx([3.0, -2.0, 1.0])


def test_actuators_reject_missing_motor() -> None:
    with pytest.raises(ValueError, match=re.escape("DOF 2 'j2' has no actuator named 'robot_j2'")) as info:
        _check_actuators('<motor name="j0" joint="j0"/><motor name="j1" joint="j1"/>')
    assert "actuators present: ['robot_j0', 'robot_j1']" in str(info.value)


@pytest.mark.parametrize(
    ("actuators", "expected_ids"),
    [
        (_MOTORS + '<motor name="spare" joint="j0"/>', [0, 1, 2]),
        ('<motor name="spare" joint="j0"/>' + _MOTORS, [1, 2, 3]),
    ],
    ids=["spare-last", "spare-first"],
)
def test_actuators_accept_extra_actuator(actuators: str, expected_ids: list[int]) -> None:
    # An actuator that drives no DOF by name is allowed; it is absent from the returned ids, so
    # apply_torques_at_dof never writes its ctrl column.
    assert _check_actuators(actuators) == expected_ids


def test_actuators_accept_any_order_resolved_by_name() -> None:
    ids = _check_actuators('<motor name="j1" joint="j1"/><motor name="j0" joint="j0"/><motor name="j2" joint="j2"/>')
    assert ids == [1, 0, 2]


def test_actuators_reject_misnamed_actuator() -> None:
    with pytest.raises(ValueError, match=r"DOF 0 'j0' has no actuator named 'robot_j0'.*'robot_motor_j0'"):
        _check_actuators(
            '<motor name="motor_j0" joint="j0"/><motor name="j1" joint="j1"/><motor name="j2" joint="j2"/>'
        )


# --- add_missing_dof_motors ------------------------------------------------------------------------------


def test_missing_motors_are_added_per_dof_and_pass_validation() -> None:
    spec = mujoco.MjSpec.from_string(_mjcf(_LINKS, '<motor name="j1" joint="j1" ctrlrange="-3 3"/>'))
    spec.worldbody.bodies[0].add_freejoint()

    assert add_missing_dof_motors(spec, _ROBOT) == ["j0", "j2"]

    assert [a.name for a in spec.actuators] == ["j1", "j0", "j2"]  # authored motor untouched, in place
    assert add_missing_dof_motors(spec, _ROBOT) == []  # idempotent
    meta = MujocoSceneManager._capture_spec_meta(None, spec, _PREFIX, "base")  # type: ignore[arg-type]
    clean_to_prefixed, _ = build_name_maps(meta)
    world = mujoco.MjSpec()
    world.copy_during_attach = True
    world.attach(spec, frame=world.worldbody.add_frame(), prefix=_PREFIX)
    model = world.compile()
    assert validate_dof_actuators(model, meta, meta.dof_joint_names, clean_to_prefixed) == [1, 0, 2]
    j1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{_PREFIX}j1")
    j0 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{_PREFIX}j0")
    assert model.actuator_ctrllimited[j1] and not model.actuator_ctrllimited[j0]  # added motor: no ctrlrange
    assert model.actuator_gear[j0, 0] == 1.0


def test_added_motor_is_a_pure_torque_motor_under_a_position_default_class() -> None:
    # A top-level <default><position .../></default> would make add_actuator() a position actuator; the
    # added motor is forced back to gain 1 / no bias.
    spec = mujoco.MjSpec.from_string(_mjcf(_LINKS, "", extra='<default><position kp="10"/></default>'))
    spec.worldbody.bodies[0].add_freejoint()
    assert add_missing_dof_motors(spec, _ROBOT) == ["j0", "j1", "j2"]
    model = spec.compile()
    assert list(model.actuator_biastype) == [int(mujoco.mjtBias.mjBIAS_NONE)] * 3
    assert list(model.actuator_gainprm[:, 0]) == [1.0] * 3


def test_actuators_reject_position_actuator() -> None:
    with pytest.raises(ValueError, match=r"biastype mjBIAS_AFFINE; only pure torque motors are supported"):
        _check_actuators(
            '<position name="j0" joint="j0" kp="10"/><motor name="j1" joint="j1"/><motor name="j2" joint="j2"/>'
        )


def test_actuators_reject_tendon_transmission() -> None:
    tendon = '<tendon><fixed name="t0"><joint joint="j0" coef="1"/></fixed></tendon>'
    with pytest.raises(ValueError, match=r"joint transmission; got trntype=mjTRN_TENDON"):
        _check_actuators(
            '<general name="j0" tendon="t0"/><motor name="j1" joint="j1"/><motor name="j2" joint="j2"/>', tendon
        )


def test_actuators_reject_nonunit_gear() -> None:
    with pytest.raises(ValueError, match=r"robot_j0.*gear\[0\]=2.0.*expected 1.0"):
        _check_actuators(
            '<motor name="j0" joint="j0" gear="2"/><motor name="j1" joint="j1"/><motor name="j2" joint="j2"/>'
        )


@pytest.mark.parametrize(
    "attributes",
    ['gainprm="2"', 'gaintype="affine" gainprm="1 0.2 0.3"'],
    ids=["nonunit-gain", "state-dependent-gain"],
)
def test_actuators_reject_gain_that_changes_joint_effort(attributes: str) -> None:
    with pytest.raises(ValueError, match="robot_j0.*fixed unit gain"):
        _check_actuators(
            f'<general name="j0" joint="j0" {attributes}/><motor name="j1" joint="j1"/><motor name="j2" joint="j2"/>'
        )


@pytest.mark.parametrize("dyntype", ["filter", "integrator"])
def test_actuators_reject_activation_dynamics(dyntype: str) -> None:
    with pytest.raises(ValueError, match="robot_j0.*no activation dynamics"):
        _check_actuators(
            f'<general name="j0" joint="j0" dyntype="{dyntype}" dynprm="0.1"/>'
            '<motor name="j1" joint="j1"/><motor name="j2" joint="j2"/>'
        )


@pytest.mark.parametrize(
    ("attribute", "value"),
    [("ctrlrange", "1 2"), ("ctrlrange", "0 2"), ("forcerange", "-2 -1"), ("forcerange", "-2 0")],
)
def test_actuators_reject_one_directional_effort_ranges(attribute: str, value: str) -> None:
    with pytest.raises(ValueError, match=rf"robot_j0.*{attribute}=.*does not provide effort in both directions"):
        _check_actuators(
            f'<motor name="j0" joint="j0" {attribute}="{value}"/>'
            '<motor name="j1" joint="j1"/><motor name="j2" joint="j2"/>'
        )


def test_actuators_accept_explicitly_disabled_zero_effort_range() -> None:
    assert _check_actuators(
        '<motor name="j0" joint="j0" ctrlrange="0 0"/><motor name="j1" joint="j1"/><motor name="j2" joint="j2"/>'
    ) == [0, 1, 2]


# --- validate_dof_addressing ------------------------------------------------------------------------


@pytest.mark.parametrize(("qpos", "qvel"), [([], []), ([7, 8, 9], [6, 7, 8]), ([0], [0])])
def test_addressing_accepts_empty_and_contiguous(qpos: list[int], qvel: list[int]) -> None:
    validate_dof_addressing(qpos, qvel)


def test_addressing_accepts_gaps_and_permutations() -> None:
    validate_dof_addressing([7, 8, 10], [8, 7, 6])


@pytest.mark.parametrize(
    ("qpos", "qvel", "message"),
    [
        ([7, 7], [6, 7], "unique and nonnegative"),
        ([7], [-1], "unique and nonnegative"),
        ([7, 8], [6], "address counts differ"),
    ],
)
def test_addressing_rejects_invalid_scalar_addresses(qpos: list[int], qvel: list[int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_dof_addressing(qpos, qvel)
