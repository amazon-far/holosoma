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
# 2018.01.02

from types import SimpleNamespace

import numpy as np
import torch

from human_body_prior.models.vposer_model import VPoser


def _make_model_ps(num_neurons: int = 32, latent_dim: int = 16):
    return SimpleNamespace(model_params=SimpleNamespace(num_neurons=num_neurons, latentD=latent_dim))


def test_sample_poses_is_deterministic():
    torch.manual_seed(42)
    vposer = VPoser(_make_model_ps())

    out_a = vposer.sample_poses(num_poses=2, seed=123)
    out_b = vposer.sample_poses(num_poses=2, seed=123)

    pose_a = out_a['pose_body'].detach().cpu().numpy()
    pose_b = out_b['pose_body'].detach().cpu().numpy()

    assert pose_a.shape == (2, vposer.num_joints, 3)
    np.testing.assert_allclose(pose_a, pose_b, atol=1e-6)
