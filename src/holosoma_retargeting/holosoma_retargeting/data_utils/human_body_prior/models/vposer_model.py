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
from human_body_prior.models.model_components import BatchFlatten
from human_body_prior.tools.rotation_tools import matrot2aa
from torch import nn
from torch.nn import functional as F


class ContinousRotReprDecoder(nn.Module):
    def __init__(self):
        super(ContinousRotReprDecoder, self).__init__()

    def forward(self, module_input):
        reshaped_input = module_input.view(-1, 3, 2)

        b1 = F.normalize(reshaped_input[:, :, 0], dim=1)

        dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
        b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=1)

        return torch.stack([b1, b2, b3], dim=-1)


class NormalDistDecoder(nn.Module):
    def __init__(self, num_feat_in, latentD):
        super(NormalDistDecoder, self).__init__()

        self.mu = nn.Linear(num_feat_in, latentD)
        self.logvar = nn.Linear(num_feat_in, latentD)

    def forward(self, Xout):
        return torch.distributions.normal.Normal(self.mu(Xout), F.softplus(self.logvar(Xout)))


class VPoser(nn.Module):
    def __init__(self, model_ps):
        super(VPoser, self).__init__()

        num_neurons, self.latentD = model_ps.model_params.num_neurons, model_ps.model_params.latentD

        self.num_joints = 21
        n_features = self.num_joints * 3

        self.encoder_net = nn.Sequential(
            BatchFlatten(),
            nn.BatchNorm1d(n_features),
            nn.Linear(n_features, num_neurons),
            nn.LeakyReLU(),
            nn.BatchNorm1d(num_neurons),
            nn.Dropout(0.1),
            nn.Linear(num_neurons, num_neurons),
            nn.Linear(num_neurons, num_neurons),
            NormalDistDecoder(num_neurons, self.latentD)
        )

        self.decoder_net = nn.Sequential(
            nn.Linear(self.latentD, num_neurons),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(num_neurons, num_neurons),
            nn.LeakyReLU(),
            nn.Linear(num_neurons, self.num_joints * 6),
            ContinousRotReprDecoder(),
        )

    def encode(self, pose_body):
        '''
        :param Pin: Nx(numjoints*3)
        :param rep_type: 'matrot'/'aa' for matrix rotations or axis-angle
        :return:
        '''
        return self.encoder_net(pose_body)

    def decode(self, Zin):
        bs = Zin.shape[0]

        prec = self.decoder_net(Zin)

        return {
            'pose_body': matrot2aa(prec.view(-1, 3, 3)).view(bs, -1, 3),
            'pose_body_matrot': prec.view(bs, -1, 9)
        }


    def forward(self, pose_body):
        '''
        :param Pin: aa: Nx1xnum_jointsx3 / matrot: Nx1xnum_jointsx9
        :param input_type: matrot / aa for matrix rotations or axis angles
        :param output_type: matrot / aa
        :return:
        '''

        q_z = self.encode(pose_body)
        q_z_sample = q_z.rsample()
        decode_results = self.decode(q_z_sample)
        decode_results.update({'poZ_body_mean': q_z.mean, 'poZ_body_std': q_z.scale, 'q_z': q_z})
        return decode_results

    def sample_poses(self, num_poses, seed=None):
        np.random.seed(seed)

        some_weight = [a for a in self.parameters()][0]
        dtype = some_weight.dtype
        device = some_weight.device
        self.eval()
        with torch.no_grad():
            Zgen = torch.tensor(np.random.normal(0., 1., size=(num_poses, self.latentD)), dtype=dtype, device=device)

        return self.decode(Zgen)
