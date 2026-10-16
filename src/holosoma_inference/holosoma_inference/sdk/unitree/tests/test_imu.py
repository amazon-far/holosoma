"""Offline H1 IMU frame and SDK state-layout regression tests."""

import unittest
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from holosoma_inference.config.config_values.robot import h1_19dof
from holosoma_inference.sdk.unitree.imu import torso_imu_to_pelvis
from holosoma_inference.sdk.unitree.unitree_interface import UnitreeInterface


class TestTorsoImu(unittest.TestCase):
    def test_zero_waist_preserves_inputs(self):
        quat = np.array([0.5, 0.5, 0.5, 0.5])
        gyro = np.array([1.2, -2.4, 3.6])
        q, g = torso_imu_to_pelvis(quat, gyro, 0.0, 0.0)
        np.testing.assert_array_equal(q, quat)
        np.testing.assert_array_equal(g, gyro)

    def test_rotating_waist_on_stationary_pelvis(self):
        yaw, speed = 0.7, -1.3
        torso_q = Rotation.from_euler("z", yaw).as_quat()[[3, 0, 1, 2]]
        q, g = torso_imu_to_pelvis(torso_q, np.array([0.0, 0.0, speed]), yaw, speed)
        np.testing.assert_allclose(q, [1.0, 0.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(g, 0.0, atol=1e-12)

    def test_tilted_moving_pelvis_and_both_waist_directions(self):
        pelvis = Rotation.from_euler("xyz", [0.3, -0.2, 0.6])
        pelvis_gyro = np.array([0.4, -0.7, 1.2])
        for yaw in (-1.0, -0.2, 0.0, 0.9):
            with self.subTest(yaw=yaw):
                waist = Rotation.from_euler("z", yaw)
                torso = pelvis * waist
                speed = 0.8
                torso_gyro = waist.inv().apply(pelvis_gyro + [0.0, 0.0, speed])
                q, g = torso_imu_to_pelvis(
                    torso.as_quat()[[3, 0, 1, 2]], torso_gyro, yaw, speed,
                )
                restored = Rotation.from_quat(q[[1, 2, 3, 0]])
                np.testing.assert_allclose(restored.as_matrix(), pelvis.as_matrix(), atol=1e-12)
                np.testing.assert_allclose(g, pelvis_gyro, atol=1e-12)

    def _interface(self, enabled):
        # Bypass DDS construction; every motor has a distinct value to catch
        # accidentally using DDS slot 10 rather than torso's physical slot 6.
        interface = object.__new__(UnitreeInterface)
        interface.robot_config = replace(h1_19dof, h1_imu_on_torso=enabled)
        interface._low_state = SimpleNamespace(
            imu_state=SimpleNamespace(quaternion=[1.0, 0.0, 0.0, 0.0], gyroscope=[0.1, 0.2, 0.3]),
            motor_state=[SimpleNamespace(q=i*0.03, dq=i*0.05) for i in range(20)],
        )
        return interface

    def test_disabled_sdk_preserves_raw_imu_and_joint_mapping(self):
        interface = self._interface(False)
        state = interface._get_low_state_sdk2py()
        self.assertEqual(state.shape, (1, 51))
        np.testing.assert_array_equal(state[0, 3:7], [1.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(state[0, 29:32], [0.1, 0.2, 0.3])
        np.testing.assert_allclose(state[0, 7:26], np.array(h1_19dof.joint2motor)*0.03)
        np.testing.assert_allclose(state[0, 32:51], np.array(h1_19dof.joint2motor)*0.05)

    def test_enabled_sdk_uses_mapped_waist_and_keeps_joint_state(self):
        interface = self._interface(True)
        state = interface._get_low_state_sdk2py()
        waist = Rotation.from_euler("z", 6*0.03)
        expected_gyro = waist.apply([0.1, 0.2, 0.3]) - [0.0, 0.0, 6*0.05]
        restored = Rotation.from_quat(state[0, 3:7][[1, 2, 3, 0]])
        np.testing.assert_allclose(restored.as_matrix(), waist.inv().as_matrix(), atol=1e-12)
        np.testing.assert_allclose(state[0, 29:32], expected_gyro, atol=1e-12)
        np.testing.assert_allclose(state[0, 7:26], np.array(h1_19dof.joint2motor)*0.03)


if __name__ == "__main__":
    unittest.main()
