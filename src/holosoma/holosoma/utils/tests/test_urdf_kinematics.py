from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import numpy.typing as npt
import pytest
import torch
from scipy.spatial.transform import Rotation

from holosoma.config_values.robot import g1_29dof
from holosoma.utils.path import resolve_asset_path
from holosoma.utils.urdf_kinematics import UrdfForwardKinematics

pytest.importorskip("pinocchio")
pytestmark = pytest.mark.no_sim

URDF_PATH = resolve_asset_path(g1_29dof.asset.urdf_file, g1_29dof.asset.asset_root)
JOINT_NAMES = list(g1_29dof.dof_names)
BODY_NAMES = list(g1_29dof.body_names)


Array = npt.NDArray[np.float64]


def _vector(text: str) -> Array:
    return np.array(text.split(), dtype=np.float64)


def _reference_fk(joint_pos: Array, root_pos: Array, root_quat: Array) -> tuple[Array, Array]:
    """Compose the URDF's joint origins and axes directly, one frame at a time."""
    joints: dict[str, tuple[str, str, Array, Rotation, Array | None]] = {}
    for joint in ET.parse(URDF_PATH).getroot().iter("joint"):  # noqa: S314  (the repo's own asset)
        # A missing <origin> is the identity.
        origin = joint.find("origin")
        origin_attrib = {} if origin is None else origin.attrib
        axis = joint.find("axis")
        child, parent = joint.find("child"), joint.find("parent")
        assert child is not None and parent is not None
        joints[child.attrib["link"]] = (
            parent.attrib["link"],
            joint.attrib["name"],
            _vector(origin_attrib.get("xyz", "0 0 0")),
            Rotation.from_euler("xyz", _vector(origin_attrib.get("rpy", "0 0 0"))),
            None if axis is None else _vector(axis.attrib["xyz"]),
        )

    def pose(link: str, angles: dict[str, float], root: tuple[Array, Rotation]) -> tuple[Array, Rotation]:
        if link not in joints:
            return root
        parent, name, xyz, rpy, axis = joints[link]
        parent_pos, parent_rot = pose(parent, angles, root)
        rot = parent_rot * rpy
        if axis is not None:
            rot = rot * Rotation.from_rotvec(axis * angles.get(name, 0.0))
        return parent_pos + parent_rot.apply(xyz), rot

    pos = np.empty((len(joint_pos), len(BODY_NAMES), 3))
    quat = np.empty((len(joint_pos), len(BODY_NAMES), 4))
    for frame in range(len(joint_pos)):
        angles = dict(zip(JOINT_NAMES, joint_pos[frame], strict=True))
        root = (root_pos[frame], Rotation.from_quat(root_quat[frame]))
        for body, name in enumerate(BODY_NAMES):
            body_pos, body_rot = pose(name, angles, root)
            pos[frame, body] = body_pos
            quat[frame, body] = body_rot.as_quat()
    return pos, quat


def test_matches_composing_the_urdf_by_hand() -> None:
    generator = torch.Generator().manual_seed(0)
    num_frames = 8
    joint_pos = (torch.rand(num_frames, len(JOINT_NAMES), generator=generator) - 0.5) * 2.0
    root_pos = torch.randn(num_frames, 3, generator=generator)
    root_quat = torch.nn.functional.normalize(torch.randn(num_frames, 4, generator=generator), dim=-1)

    body_pos, body_quat = UrdfForwardKinematics(URDF_PATH, JOINT_NAMES, BODY_NAMES)(joint_pos, root_pos, root_quat)
    expected_pos, expected_quat = _reference_fk(
        joint_pos.double().numpy(), root_pos.double().numpy(), root_quat.double().numpy()
    )

    assert np.abs(body_pos.double().numpy() - expected_pos).max() < 1e-5
    # q and -q are the same rotation.
    assert (1.0 - np.abs((body_quat.double().numpy() * expected_quat).sum(axis=-1))).max() < 1e-6


def test_returns_the_input_device_and_dtype() -> None:
    joint_pos = torch.zeros(2, len(JOINT_NAMES), dtype=torch.float64)
    root_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 2, dtype=torch.float64)

    body_pos, body_quat = UrdfForwardKinematics(URDF_PATH, JOINT_NAMES, BODY_NAMES)(
        joint_pos, torch.zeros(2, 3, dtype=torch.float64), root_quat
    )

    assert body_pos.shape == (2, len(BODY_NAMES), 3)
    assert body_quat.shape == (2, len(BODY_NAMES), 4)
    assert body_pos.dtype == body_quat.dtype == torch.float64


def test_unknown_names_raise() -> None:
    with pytest.raises(ValueError, match="no_such_joint.*no_such_link"):
        UrdfForwardKinematics(URDF_PATH, [*JOINT_NAMES, "no_such_joint"], [*BODY_NAMES, "no_such_link"])
