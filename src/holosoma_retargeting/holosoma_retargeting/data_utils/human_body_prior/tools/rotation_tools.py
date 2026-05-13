# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
#
# If you use this code in a research publication please consider citing the following:
#
# Expressive Body Capture: 3D Hands, Face, and Body from a Single Image <https://arxiv.org/abs/1904.05866>
#
#
# Code Developed by:
# Nima Ghorbani <https://nghorbani.github.io/>
#
# 2020.12.12


import numpy as np
import torch
from human_body_prior.tools import tgm_conversion as tgm
from torch.nn import functional as F


def local2global_pose(local_pose, kintree):
    bs = local_pose.shape[0]

    local_pose = local_pose.view(bs, -1, 3, 3)

    global_pose = local_pose.clone()

    for jId in range(len(kintree)):
        parent_id = kintree[jId]
        if parent_id >= 0:
            global_pose[:, jId] = torch.matmul(global_pose[:, parent_id], global_pose[:, jId])

    return global_pose


def em2euler(em):
    '''

    :param em: rotation in expo-map (3,)
    :return: rotation in euler angles (3,)
    '''
    from transforms3d.euler import axangle2euler

    theta = np.sqrt((em ** 2).sum())
    axis = em / theta
    return np.array(axangle2euler(axis, theta))


def euler2em(ea):
    '''

    :param ea: rotation in euler angles (3,)
    :return: rotation in expo-map (3,)
    '''
    from transforms3d.euler import euler2axangle
    axis, theta = euler2axangle(*ea)
    return np.array(axis * theta)


def remove_zrot(pose):
    noZ = em2euler(pose[:3].copy())
    noZ[2] = 0
    pose[:3] = euler2em(noZ).copy()
    return pose


def matrot2aa(pose_matrot):
    '''
    :param pose_matrot: Nx3x3
    :return: Nx3
    '''
    bs = pose_matrot.size(0)
    homogen_matrot = F.pad(pose_matrot, [0, 1])
    pose = tgm.rotation_matrix_to_angle_axis(homogen_matrot)
    return pose


def aa2matrot(pose):
    '''
    :param Nx3
    :return: pose_matrot: Nx3x3
    '''
    bs = pose.size(0)
    num_joints = pose.size(1) // 3
    pose_body_matrot = tgm.angle_axis_to_rotation_matrix(pose)[:, :3, :3].contiguous()  # .view(bs, num_joints*9)
    return pose_body_matrot


def noisy_zrot(rot_in):
    '''
    :param rot_in: np.array Nx3 rotations in axis-angle representation
    :return:
        randomize the zrotations and reutn in the same shape as input.
        the firt element t of T will be added a random angle and this addition will happen to all frames
    '''
    is_batched = False
    if rot_in.ndim == 2: is_batched = True
    if not is_batched:
        rot_in = rot_in[np.newaxis]

    rnd_zrot = np.random.uniform(-np.pi, np.pi)
    rot_out = []
    for bId in range(len(rot_in)):
        pose_cpu = rot_in[bId]
        pose_euler = em2euler(pose_cpu)

        pose_euler[2] += rnd_zrot

        pose_aa = euler2em(pose_euler)
        if np.any(np.isnan(pose_aa)):
            rot_out.append(pose_cpu)
        else:
            rot_out.append(pose_aa.copy())

    return np.array(rot_out)

from typing import Union, List


def rotate_points_xyz(mesh_v: np.ndarray, Rxyz: Union[List[int], np.ndarray]):
    '''

    :param mesh_v: Nxnum_vx3
    :param Rxyz: Nx3 or 3 in degrees
    :return:
    '''
    if Rxyz is not None:
        Rxyz = list(Rxyz)
        Rxyz = np.repeat(np.array(Rxyz).reshape(1, 3), repeats=len(mesh_v), axis=0)

    mesh_v_rotated = []

    for fId in range(mesh_v.shape[0]):
        angle = np.radians(Rxyz[fId, 0])
        rx = np.array([
            [1., 0., 0.],
            [0., np.cos(angle), -np.sin(angle)],
            [0., np.sin(angle), np.cos(angle)]
        ])

        angle = np.radians(Rxyz[fId, 1])
        ry = np.array([
            [np.cos(angle), 0., np.sin(angle)],
            [0., 1., 0.],
            [-np.sin(angle), 0., np.cos(angle)]
        ])

        angle = np.radians(Rxyz[fId, 2])
        rz = np.array([
            [np.cos(angle), -np.sin(angle), 0.],
            [np.sin(angle), np.cos(angle), 0.],
            [0., 0., 1.]
        ])
        mesh_v_rotated.append(rz.dot(ry.dot(rx.dot(mesh_v[fId].T))).T)

    return np.array(mesh_v_rotated)


def tmat(R, t):
    ''' Creates a batch of transformation matrices
        Args:
            - R: NxBx3x3 array of a batch of rotation matrices
            - t: NxBx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    '''
    # No padding left or right, only add an extra row

    bs = R.shape[0]

    return torch.cat([F.pad(R.view(-1, 3, 3), [0, 0, 0, 1]),
                      F.pad(t.view(-1, 3, 1), [0, 0, 0, 1], value=1)], dim=2).view(bs, -1, 4, 4)


def batch_rigid_transform(rot_mats, joints, parents):
    """
    Applies a batch of rigid transformations to the joints

    Parameters
    ----------
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.tensor BxNx3
        Locations of joints
    parents : torch.tensor BxN
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32

    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """

    joints = torch.unsqueeze(joints, dim=-1)  # BxNx3X1

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transform_chain = [tmat(rot_mats[:, 0], rel_joints[:, 0])[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[parents[i]], tmat(rot_mats[:, i], rel_joints[:, i])[:, 0])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    return posed_joints
