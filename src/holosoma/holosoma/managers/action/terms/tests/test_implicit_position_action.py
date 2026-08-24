from __future__ import annotations

from types import SimpleNamespace
from typing import Literal, cast

import pytest
import torch

from holosoma.config_types.robot import RobotConfig, RobotControlConfig
from holosoma.managers.action.terms.joint_control import JointPositionTargetActionTerm
from holosoma.simulator.isaacsim.actuator_cfg import build_actuator_configs, joint_drive_target_type

pytestmark = pytest.mark.no_sim


class _FakeActuatorCfg:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _custom_implicit_actuators() -> dict[str, str]:
    return {"custom": "implicit"}


def _robot_config(
    control_mode: Literal["explicit_torque", "implicit_position_target"] = "explicit_torque",
    implicit_actuator_builder: str | None = None,
) -> RobotConfig:
    control = RobotControlConfig(
        control_type="P",
        stiffness={"hip_pitch": 40.0},
        damping={"hip_pitch": 2.5},
        action_scale=0.25,
        action_clip_value=100.0,
        clip_actions=True,
        clip_torques=True,
        action_scales_by_effort_limit_over_p_gain=True,
        control_mode=control_mode,
        implicit_actuator_builder=implicit_actuator_builder,
    )
    return cast(
        "RobotConfig",
        SimpleNamespace(
            control=control,
            dof_names=["left_hip_pitch_joint"],
            dof_effort_limit_list=[139.0],
            dof_vel_limit_list=[20.0],
            dof_armature_list=[0.01017752],
            dof_joint_friction_list=[0.01],
        ),
    )


def test_control_mode_validation() -> None:
    with pytest.raises(ValueError, match="position control"):
        RobotControlConfig(
            control_type="T",
            stiffness={},
            damping={},
            action_scale=1.0,
            action_clip_value=1.0,
            clip_actions=True,
            clip_torques=True,
            control_mode="implicit_position_target",
        )

    with pytest.raises(ValueError, match="requires control_mode"):
        RobotControlConfig(
            control_type="P",
            stiffness={},
            damping={},
            action_scale=1.0,
            action_clip_value=1.0,
            clip_actions=True,
            clip_torques=True,
            implicit_actuator_builder="package.module:builder",
        )


def test_actuator_backend_preserves_the_robot_contract() -> None:
    explicit = build_actuator_configs(
        _robot_config(), ideal_pd_actuator_cls=_FakeActuatorCfg, implicit_actuator_cls=_FakeActuatorCfg
    )["left_hip_pitch_joint"]
    assert explicit.effort_limit == 139.0
    assert explicit.velocity_limit == 20.0
    assert explicit.stiffness == 0
    assert explicit.damping == 0

    implicit = build_actuator_configs(
        _robot_config("implicit_position_target"),
        ideal_pd_actuator_cls=_FakeActuatorCfg,
        implicit_actuator_cls=_FakeActuatorCfg,
    )["left_hip_pitch_joint"]
    assert implicit.effort_limit_sim == 139.0
    assert implicit.velocity_limit_sim == 20.0
    assert implicit.stiffness == 40.0
    assert implicit.damping == 2.5
    assert implicit.armature == 0.01017752


def test_implicit_backend_enables_urdf_position_drives() -> None:
    assert joint_drive_target_type(_robot_config()) == "none"
    assert joint_drive_target_type(_robot_config("implicit_position_target")) == "position"


def test_custom_implicit_actuator_builder_uses_manager_path_syntax() -> None:
    robot = _robot_config(
        "implicit_position_target",
        implicit_actuator_builder=(
            "holosoma.managers.action.terms.tests.test_implicit_position_action:_custom_implicit_actuators"
        ),
    )
    assert build_actuator_configs(
        robot,
        ideal_pd_actuator_cls=_FakeActuatorCfg,
        implicit_actuator_cls=_FakeActuatorCfg,
    ) == {"custom": "implicit"}


def test_implicit_action_captures_torque_after_physics_step(monkeypatch: pytest.MonkeyPatch) -> None:
    term = JointPositionTargetActionTerm.__new__(JointPositionTargetActionTerm)
    term._randomize_torque_rfi = False
    term._substep_idx = 0
    term._actions_after_delay = torch.tensor([[0.5, -0.5]])
    term.action_scales = torch.tensor([0.2, 0.4])
    term.torques = torch.zeros(1, 2)
    term.torques_substep = torch.zeros(1, 1, 2)
    term.dof_pos_substep = torch.zeros(1, 1, 2)
    term.dof_vel_substep = torch.zeros(1, 1, 2)

    class Simulator:
        dof_pos = torch.tensor([[1.0, 2.0]])
        dof_vel = torch.tensor([[3.0, 4.0]])
        target = None

        def get_applied_torques_at_dof(self):
            return torch.tensor([[5.0, 6.0]])

        def apply_position_targets_at_dof(self, target):
            self.target = target.clone()

    simulator = Simulator()
    term.env = SimpleNamespace(simulator=simulator, default_dof_pos=torch.tensor([[0.1, 0.2]]))
    monkeypatch.setattr(
        term,
        "_compute_torques",
        lambda _: pytest.fail("implicit action must not compute explicit PD torque"),
    )

    term.apply_actions()

    assert torch.equal(simulator.target, torch.tensor([[0.2, 0.0]]))
    assert torch.count_nonzero(term.torques) == 0

    term._capture_applied_torques()

    assert torch.equal(term.torques, torch.tensor([[5.0, 6.0]]))
    assert torch.equal(term.torques_substep, torch.tensor([[[5.0, 6.0]]]))
