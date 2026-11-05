"""Floating-base forward kinematics of a URDF, computed with pinocchio rather than a simulator."""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.transform import Rotation


class UrdfForwardKinematics:
    """Body link-origin poses for a floating-base robot, in caller-chosen joint and body orders."""

    def __init__(self, urdf_path: str, joint_names: list[str], body_names: list[str]):
        import pinocchio as pin

        self._pin = pin
        self._model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        self._data = self._model.createData()

        missing_joints = [name for name in joint_names if not self._model.existJointName(name)]
        missing_bodies = [name for name in body_names if not self._model.existFrame(name, pin.FrameType.BODY)]
        if missing_joints or missing_bodies:
            raise ValueError(f"{urdf_path} lacks joints {missing_joints} and bodies {missing_bodies}")

        joints = [self._model.joints[self._model.getJointId(name)] for name in joint_names]
        if any(joint.nq != 1 for joint in joints):
            raise ValueError(f"{urdf_path}: only 1-DoF joints are supported, got {joint_names}")
        self._joint_q_idx = np.array([joint.idx_q for joint in joints])
        self._frame_ids = [self._model.getFrameId(name, pin.FrameType.BODY) for name in body_names]

    def __call__(
        self, joint_pos: torch.Tensor, root_pos: torch.Tensor, root_quat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(body_pos, body_quat)`` shaped ``(num_frames, num_bodies, 3 | 4)``, quats xyzw.

        ``joint_pos`` is ``(num_frames, num_joints)`` in ``joint_names`` order; ``root_pos`` and
        ``root_quat`` (xyzw) place the URDF's root link.
        """
        num_frames = joint_pos.shape[0]
        q = np.zeros((num_frames, self._model.nq))
        q[:, :3] = root_pos.detach().cpu().double().numpy()
        q[:, 3:7] = root_quat.detach().cpu().double().numpy()
        q[:, self._joint_q_idx] = joint_pos.detach().cpu().double().numpy()

        pos = np.empty((num_frames, len(self._frame_ids), 3))
        rot = np.empty((num_frames, len(self._frame_ids), 3, 3))
        for frame, configuration in enumerate(q):
            self._pin.framesForwardKinematics(self._model, self._data, configuration)
            for body, frame_id in enumerate(self._frame_ids):
                placement = self._data.oMf[frame_id]
                pos[frame, body] = placement.translation
                rot[frame, body] = placement.rotation

        quat = Rotation.from_matrix(rot.reshape(-1, 3, 3)).as_quat().reshape(num_frames, -1, 4)
        to_input = {"device": joint_pos.device, "dtype": joint_pos.dtype}
        return torch.from_numpy(pos).to(**to_input), torch.from_numpy(quat).to(**to_input)
