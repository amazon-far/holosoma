"""Unitree SDK motor order survives a different canonical G1 joint order."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
mujoco = pytest.importorskip("mujoco")

from holosoma.bridge.unitree.unitree_sdk2py_bridge import UnitreeSdk2Bridge  # noqa: E402
from holosoma.bridge.unitree.unitree_sdk2py_bridge_mp import LowCommand  # noqa: E402
from holosoma.config_types.robot import RobotConfig  # noqa: E402
from holosoma.config_types.simulator import BridgeConfig  # noqa: E402
from holosoma.config_values.robot import g1_29dof  # noqa: E402
from tests.simulators.mujoco._build import build_classic_sim  # noqa: E402

pytestmark = pytest.mark.mujoco_classic

# Fixed G1 SDK motor slots, independent of the config and bridge mapping under test.
_SDK_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


class _DryUnitree(UnitreeSdk2Bridge):
    def _init_sdk_components(self) -> None:
        """Replace only transport initialization; retain real torque and state methods."""
        self.low_cmd = None


def _reordered_g1() -> RobotConfig:
    order = list(range(29))
    # A three-cycle also distinguishes a permutation from its inverse.
    order[0], order[14], order[28] = order[28], order[0], order[14]

    def reorder(values: list[float] | None) -> list[float] | None:
        return None if values is None else [values[i] for i in order]

    # Inherit bridge.controlled_dof_names unchanged from the shipped preset.
    return replace(
        g1_29dof,
        dof_names=[g1_29dof.dof_names[i] for i in order],
    )


def test_reordered_g1_keeps_sdk_state_and_physical_motor_order() -> None:
    robot = _reordered_g1()
    assert robot.bridge.controlled_dof_names == _SDK_JOINT_NAMES
    assert [robot.dof_names[i] for i in (0, 14, 28)] == [
        "right_wrist_yaw_joint",
        "left_hip_pitch_joint",
        "waist_pitch_joint",
    ]
    sim = build_classic_sim(robot=robot)
    try:
        bridge = _DryUnitree(sim, robot, BridgeConfig(enabled=False))
        assert bridge.num_motor == 29
        model, data = sim.root_model, sim.root_data

        # Write native engine addresses by physical joint name, without using the canonical
        # address lists or bridge indices. Every SDK slot has distinctive q/dq/ddq.
        slots = np.arange(1, 30, dtype=np.float32)
        expected_q = slots * np.float32(0.01)
        expected_dq = slots * np.float32(-0.02)
        expected_ddq = slots * np.float32(0.3)
        for i, name in enumerate(_SDK_JOINT_NAMES):
            joint = model.joint(f"robot_{name}")
            data.qpos[joint.qposadr[0]] = expected_q[i]
            data.qvel[joint.dofadr[0]] = expected_dq[i]
            data.qacc[joint.dofadr[0]] = expected_ddq[i]

        # This is the getter used by Unitree publish_low_state. Read before forward dynamics
        # replaces the deliberately written accelerations.
        actual_q, actual_dq, actual_ddq = bridge._get_dof_states()
        np.testing.assert_array_equal(actual_q, expected_q)
        np.testing.assert_array_equal(actual_dq, expected_dq)
        np.testing.assert_array_equal(actual_ddq, expected_ddq)

        # SDK slot 0 commands +1 Nm. Nonzero gains at the known SDK q/dq targets ensure
        # compute_torques also gathers feedback in SDK order: every PD error must be zero.
        expected_torques = np.zeros(29, dtype=np.float32)
        expected_torques[0] = 1.0
        bridge.low_cmd = LowCommand(
            tau_ff=expected_torques.tolist(),
            kp=[2.0] * 29,
            kd=[0.5] * 29,
            q_target=expected_q.tolist(),
            dq_target=expected_dq.tolist(),
        )
        torques = bridge.compute_torques()
        np.testing.assert_array_equal(torques, expected_torques)
        data.ctrl[:] = 0.0
        sim.apply_torques_at_dof(torch.as_tensor(torques, dtype=torch.float32), dof_indices=bridge._apply_indices)

        # Verify the native actuator and its joint transmission, not the mapping's own indices.
        hip = model.joint("robot_left_hip_pitch_joint")
        hip_motor = model.actuator("robot_left_hip_pitch_joint")
        assert model.actuator_trntype[hip_motor.id] == mujoco.mjtTrn.mjTRN_JOINT
        assert model.actuator_trnid[hip_motor.id, 0] == hip.id
        expected_ctrl = np.zeros(model.nu)
        expected_ctrl[hip_motor.id] = 1.0
        np.testing.assert_array_equal(data.ctrl, expected_ctrl)

        # The command must become generalized torque on that physical hinge alone.
        mujoco.mj_forward(model, data)
        expected_force = np.zeros(model.nv)
        expected_force[hip.dofadr[0]] = 1.0
        np.testing.assert_allclose(data.qfrc_actuator, expected_force, atol=1e-7)
    finally:
        sim.close()
