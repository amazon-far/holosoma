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
# 2018.12.13

import numpy as np
import torch
from transforms3d.axangles import axangle2mat, mat2axangle

from human_body_prior.tools.rotation_tools import aa2matrot, matrot2aa


def _axis_angle_to_matrix(vec: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(vec)
    if angle == 0.0:
        return np.eye(3)
    axis = vec / angle
    return axangle2mat(axis, angle)


def _matrix_to_axis_angle(mat: np.ndarray) -> np.ndarray:
    axis, angle = mat2axangle(mat)
    return np.asarray(axis) * angle


def test_aa2matrot_matches_transforms3d():
    aa = torch.randn(10, 3)
    ours = aa2matrot(aa).detach().cpu().numpy()
    expected = np.stack([
        _axis_angle_to_matrix(vec) for vec in aa.detach().cpu().numpy()
    ])
    np.testing.assert_allclose(ours, expected, atol=1e-5)


def test_matrot2aa_round_trip():
    aa = torch.randn(10, 3)
    mats = aa2matrot(aa)
    recovered = matrot2aa(mats).detach().cpu().numpy()
    np.testing.assert_allclose(recovered, aa.detach().cpu().numpy(), atol=1e-5)


def test_matrot2aa_matches_transforms3d():
    mats = np.stack([
        _axis_angle_to_matrix(vec) for vec in np.random.randn(10, 3)
    ])
    ours = matrot2aa(torch.tensor(mats, dtype=torch.float32)).detach().cpu().numpy()
    expected = np.stack([
        _matrix_to_axis_angle(mat) for mat in mats
    ])
    np.testing.assert_allclose(ours, expected, atol=1e-5)
