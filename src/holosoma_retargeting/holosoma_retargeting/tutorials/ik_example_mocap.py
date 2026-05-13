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
# 2021.02.12
from pathlib import Path
from typing import Union, Iterable

import numpy as np
import torch
from colour import Color
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.models.ik_engine import IK_Engine
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.tools.omni_tools import get_support_data_dir
from torch import nn

def compute_vertex_normal_batched(vertices, indices):
    from pytorch3d.structures import Meshes
    return Meshes(verts=vertices, faces=indices.expand(len(vertices), -1, -1)).verts_normals_packed().view(-1, vertices.shape[1],3)

class SourceKeyPoints(nn.Module):
    def __init__(self,
                 bm: Union[str, BodyModel],
                 vids: Iterable[int],
                 kpts_colors: Union[np.ndarray, None] = None ,
                 ):
        super(SourceKeyPoints, self).__init__()

        self.bm = BodyModel(bm, persistant_buffer=False) if isinstance(bm, str) else bm
        self.bm_f = []#self.bm.f
        self.vids = vids
        self.kpts_colors = np.array([Color('grey').rgb for _ in vids]) if kpts_colors == None else kpts_colors

    def forward(self, body_parms):
        new_body = self.bm(**body_parms)

        vn = compute_vertex_normal_batched(new_body.v, new_body.f)
        virtual_markers = new_body.v[:, self.vids] + 0.0095 * vn[:, self.vids]

        return {'source_kpts':virtual_markers, 'body': new_body}

support_dir = Path(get_support_data_dir()) / 'dowloads'
vposer_expr_dir = str(support_dir / 'vposer_v2_05') #'TRAINED_MODEL_DIRECTORY'  in this directory the trained model along with the model code exist
bm_fname =  str((support_dir / 'models' / 'smplx' / 'neutral' / 'model.npz').resolve()) #'PATH_TO_SMPLX_model.npz'  obtain from https://smpl-x.is.tue.mpg.de/downloads
sample_amass_fname = str((support_dir / 'amass_sample.npz').resolve())# a sample npz file from AMASS

comp_device = torch.device('cuda')


sample_amass = np.load(sample_amass_fname, allow_pickle=True)

print('sample_amass keys: ', list(sample_amass.keys()))

vids = sample_amass['vids'].tolist()

red = Color("red")
blue = Color("blue")
kpts_colors = [c.rgb for c in list(red.range_to(blue, len(vids)))]

# create source and target key points and make sure they are index aligned
data_loss = torch.nn.MSELoss(reduction='sum')

stepwise_weights = [
    {'data': 10., 'poZ_body': .01, 'betas': .5},
                    ]
# optimizer_args = {'type':'ADAM', 'max_iter':500, 'lr':1e-1, 'tolerance_change': 1e-5}
optimizer_args = {'type':'LBFGS', 'max_iter':300, 'lr':1, 'tolerance_change': 1e-4, 'history_size':200}
ik_engine = IK_Engine(vposer_expr_dir=vposer_expr_dir,
                      verbosity=2,
                      display_rc=(2, 2),
                      data_loss=data_loss,
                      stepwise_weights=stepwise_weights,
                      optimizer_args=optimizer_args).to(comp_device)

markers_orig = sample_amass['markers']
rnd_frame_ids = np.random.choice(len(markers_orig), 10, replace=False)

target_pts = markers_orig[rnd_frame_ids]
target_pts = torch.Tensor(target_pts).type(torch.float).to(comp_device)
source_pts = SourceKeyPoints(bm=bm_fname, vids=vids, kpts_colors=kpts_colors).to(comp_device)

ik_res = ik_engine(source_pts, target_pts)

ik_res_detached = {k: v.detach() for k, v in ik_res.items()}
nan_mask = torch.isnan(ik_res_detached['trans']).sum(-1) != 0
if nan_mask.sum() != 0: raise ValueError('Sum results were NaN!')
