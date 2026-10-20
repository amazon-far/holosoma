"""Preserved authored drives contribute real effort outside selected-motor feedback."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
mujoco = pytest.importorskip("mujoco")

from holosoma.config_types.robot import RobotConfig, RobotInitState  # noqa: E402
from holosoma.simulator.mujoco.robot_addressing import (  # noqa: E402
    add_missing_dof_motors,
    validate_dof_actuators,
)
from holosoma.simulator.mujoco.scene_manager import ActorSpecMeta  # noqa: E402
from tests.simulators._robot_presets import test_arm_fixed as arm_preset  # noqa: E402
from tests.simulators.mujoco._build import build_classic_sim  # noqa: E402

pytestmark = pytest.mark.mujoco_classic

_AUTHORED_SERVO_XML = """
<mujoco>
  <compiler angle="radian"/>
  <default><geom contype="0" conaffinity="0"/></default>
  <worldbody>
    <body name="base">
      <geom size=".01"/>
      <body name="link">
        <joint name="hinge" axis="0 0 1" armature="1"/>
        <geom type="capsule" size=".01" fromto="0 0 0 1 0 0" mass="1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="servo" joint="hinge" kp="100" ctrlrange="-2 2" forcerange="-80 80"/>
  </actuator>
</mujoco>
"""


def test_synthesized_motor_preserves_authored_servo_dynamics() -> None:
    """Exercise synthesis and validation, then integrate the real two-drive joint."""
    spec = mujoco.MjSpec.from_string(_AUTHORED_SERVO_XML)
    authored = spec.compile()
    assert add_missing_dof_motors(spec, "authored_drives") == ["hinge"]
    model = spec.compile()
    meta = ActorSpecMeta(
        prefix="",
        root_body="base",
        body_names=["base", "link"],
        dof_joint_names=["hinge"],
        actuator_names=["servo", "hinge"],
    )
    selected = validate_dof_actuators(model, meta, ["hinge"], {"hinge": "hinge"})
    motor, servo = model.actuator("hinge").id, model.actuator("servo").id
    assert model.nu == 2
    assert selected == [motor]
    assert servo not in selected
    for field in ("actuator_gainprm", "actuator_biasprm", "actuator_gear", "actuator_ctrlrange", "actuator_forcerange"):
        np.testing.assert_array_equal(getattr(model, field)[servo], getattr(authored, field)[0])

    data = mujoco.MjData(model)
    data.qpos[0] = 0.5
    mujoco.mj_forward(model, data)
    assert data.ctrl[motor] == data.ctrl[servo] == 0.0
    assert data.actuator_force[motor] == 0.0
    assert data.actuator_force[servo] == pytest.approx(-50.0)
    assert data.qfrc_actuator[0] == pytest.approx(-50.0)
    assert data.qacc[0] < 0.0
    mujoco.mj_step(model, data)
    assert data.qvel[0] < 0.0
    assert data.qpos[0] < 0.5


@pytest.mark.parametrize("effort_limit", [None, 5.0], ids=["authored-limits", "config-joint-clamp"])
def test_classic_feedback_and_limits_distinguish_selected_motor_from_combined_effort(
    tmp_path: Path,
    effort_limit: float | None,
) -> None:
    """The public load/command/feedback path preserves drives and enforces the combined joint clamp."""
    asset = tmp_path / "authored_servo.xml"
    asset.write_text(_AUTHORED_SERVO_XML)
    robot = RobotConfig(
        dof_names=["hinge"],
        body_names=["base", "link"],
        init_state=RobotInitState(position=[0.0, 0.0, 2.0], default_joint_positions={"hinge": 0.5}),
        asset=replace(arm_preset.asset, xml_file=str(asset)),
        dof_effort_limits=None if effort_limit is None else dict(zip(["hinge"], [effort_limit])),
    )
    sim = build_classic_sim(robot=robot)
    try:
        model, data = sim.root_model, sim.root_data
        motor, servo = model.actuator("robot_hinge").id, model.actuator("robot_servo").id
        joint = model.joint("robot_hinge")
        qpos_id, qvel_id = int(joint.qposadr[0]), int(joint.dofadr[0])
        assert model.nu == 2
        assert sim.dof_actuator_ids == [motor]
        np.testing.assert_array_equal(model.actuator_ctrlrange[servo], [-2.0, 2.0])
        np.testing.assert_array_equal(model.actuator_forcerange[servo], [-80.0, 80.0])
        assert model.actuator_gainprm[servo, 0] == 100.0
        _, _, reported = sim.get_dof_limits_properties()
        assert reported.tolist() == [float("inf") if effort_limit is None else effort_limit]

        # The nonzero servo target also proves motor commands do not overwrite its ctrl.
        for servo_target, command in [(0.0, 0.0), (0.25, 12.0), (-0.25, -12.0)]:
            data.qpos[qpos_id], data.qvel[qvel_id] = 0.5, 0.0
            data.ctrl[servo] = servo_target
            sim.apply_torques_at_dof(torch.tensor([[command]]))
            sim.forward_kinematics()
            expected_motor = command if effort_limit is None else float(np.clip(command, -effort_limit, effort_limit))
            expected_servo = max(-80.0, min(80.0, 100.0 * (servo_target - 0.5)))
            combined = expected_motor + expected_servo
            expected_joint = combined if effort_limit is None else float(np.clip(combined, -effort_limit, effort_limit))
            assert data.ctrl[servo] == servo_target
            assert data.actuator_force[servo] == pytest.approx(expected_servo)
            torch.testing.assert_close(sim.get_dof_forces(), torch.tensor([expected_motor]))
            assert data.qfrc_actuator[qvel_id] == pytest.approx(expected_joint)
            assert expected_joint < 0.0
            assert data.ncon == 0
            sim.simulate_at_each_physics_step()
            assert data.qvel[qvel_id] < 0.0
            assert data.qpos[qpos_id] < 0.5
    finally:
        sim.close()
