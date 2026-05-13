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
import torch.nn as nn

# from smplx.lbs import lbs
from human_body_prior.body_model.lbs import lbs
import sys

class BodyModel(nn.Module):

    def __init__(self,
                 bm_fname,
                 num_betas=10,
                 num_dmpls=None, dmpl_fname=None,
                 num_expressions=None,
                 use_posedirs=True,
                 model_type=None,
                 dtype=torch.float32,
                 persistant_buffer=False):

        super(BodyModel, self).__init__()

        '''
        :param bm_fname: path to a SMPL model as pkl file
        :param num_betas: number of shape parameters to include.
        :param device: default on gpu
        :param dtype: float precision of the computations
        :return: verts, trans, pose, betas 
        '''

        self.dtype = dtype


        # -- Load SMPL params --
        if bm_fname.endswith('.npz'):
            smpl_dict = np.load(bm_fname, encoding='latin1')
        else:
            raise ValueError(f'bm_fname must be a .npz file: {bm_fname}')

        # these are supposed for later convenient look up
        self.num_betas = num_betas
        self.num_dmpls = num_dmpls
        self.num_expressions = num_expressions

        npose_params = smpl_dict['posedirs'].shape[2] // 3
        if model_type:
            self.model_type = model_type
        else:
            self.model_type = {12:'flame', 69: 'smpl', 153: 'smplh', 162: 'smplx', 45: 'mano',
                           105: 'animal_horse', 102: 'animal_dog'}[npose_params]

        assert self.model_type in ['smpl', 'smplh', 'smplx', 'mano', 'mano', 'animal_horse', 'animal_dog', 'flame', 'animal_rat'], ValueError(
            'model_type should be in smpl/smplh/smplx/mano.')

        self.use_dmpl = False
        if num_dmpls is not None:
            if dmpl_fname is not None:
                self.use_dmpl = True
            else:
                raise (ValueError('dmpl_fname should be provided when using dmpls!'))

        if self.use_dmpl and self.model_type in ['smplx', 'mano', 'animal_horse', 'animal_dog']: raise (
            NotImplementedError('DMPLs only work with SMPL/SMPLH models for now.'))

        self.use_expression = self.model_type in ['smplx','flame'] and num_expressions is not None

        # Mean template vertices
        self.comp_register('init_v_template', torch.tensor(smpl_dict['v_template'][None], dtype=dtype), persistent=persistant_buffer)

        self.comp_register('f', torch.tensor(smpl_dict['f'].astype(np.int32), dtype=torch.int32), persistent=persistant_buffer)

        num_total_betas = smpl_dict['shapedirs'].shape[-1]
        if num_betas < 1:
            num_betas = num_total_betas

        shapedirs = smpl_dict['shapedirs'][:, :, :num_betas]
        self.comp_register('shapedirs', torch.tensor(shapedirs, dtype=dtype), persistent=persistant_buffer)

        if self.use_expression:
            if smpl_dict['shapedirs'].shape[-1] > 300:
                begin_shape_id = 300
            else:
                begin_shape_id = 10
                num_expressions = smpl_dict['shapedirs'].shape[-1] - 10

            exprdirs = smpl_dict['shapedirs'][:, :, begin_shape_id:(begin_shape_id + num_expressions)]
            self.comp_register('exprdirs', torch.tensor(exprdirs, dtype=dtype), persistent=persistant_buffer)

            expression = torch.tensor(np.zeros((1, num_expressions)), dtype=dtype)
            self.comp_register('init_expression', expression, persistent=persistant_buffer)

        if self.use_dmpl:
            dmpldirs = np.load(dmpl_fname)['eigvec']

            dmpldirs = dmpldirs[:, :, :num_dmpls]
            self.comp_register('dmpldirs', torch.tensor(dmpldirs, dtype=dtype), persistent=persistant_buffer)

        # Regressor for joint locations given shape - 6890 x 24
        self.comp_register('J_regressor', torch.tensor(smpl_dict['J_regressor'], dtype=dtype), persistent=persistant_buffer)

        # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*30 x 207
        if use_posedirs:
            posedirs = smpl_dict['posedirs']
            # print(self.model_type, posedirs.shape)
            posedirs = posedirs.reshape([posedirs.shape[0] * 3, -1]).T
            self.comp_register('posedirs', torch.tensor(posedirs, dtype=dtype), persistent=persistant_buffer)
        else:
            self.posedirs = None

        # indices of parents for each joints
        kintree_table = smpl_dict['kintree_table'].astype(np.int32)
        self.comp_register('kintree_table', torch.tensor(kintree_table, dtype=torch.int32), persistent=persistant_buffer)

        # LBS weights
        # weights = np.repeat(smpl_dict['weights'][np.newaxis], batch_size, axis=0)
        weights = smpl_dict['weights']
        self.comp_register('weights', torch.tensor(weights, dtype=dtype), persistent=persistant_buffer)

        self.comp_register('init_trans', torch.zeros((1,3), dtype=dtype), persistent=persistant_buffer)
        # self.register_parameter('trans', nn.Parameter(trans, requires_grad=True))

        # root_orient
        # if self.model_type in ['smpl', 'smplh']:
        self.comp_register('init_root_orient', torch.zeros((1,3), dtype=dtype), persistent=persistant_buffer)

        # pose_body
        if self.model_type in ['smpl', 'smplh', 'smplx']:
            self.comp_register('init_pose_body', torch.zeros((1,63), dtype=dtype), persistent=persistant_buffer)
        elif self.model_type == 'animal_horse':
            self.comp_register('init_pose_body', torch.zeros((1,105), dtype=dtype), persistent=persistant_buffer)
        elif self.model_type == 'flame':
            self.comp_register('init_pose_body', torch.zeros((1,3), dtype=dtype), persistent=persistant_buffer)
        elif self.model_type in ['animal_dog','animal_rat']:
            self.comp_register('init_pose_body', torch.zeros((1,102), dtype=dtype), persistent=persistant_buffer)

        # pose_hand
        if self.model_type in ['smpl']:
            self.comp_register('init_pose_hand', torch.zeros((1,1*3*2), dtype=dtype), persistent=persistant_buffer)
        elif self.model_type in ['smplh', 'smplx']:
            self.comp_register('init_pose_hand', torch.zeros((1,15*3*2), dtype=dtype), persistent=persistant_buffer)
        elif self.model_type in ['mano']:
            self.comp_register('init_pose_hand', torch.zeros((1,15*3), dtype=dtype), persistent=persistant_buffer)

        # face poses
        if self.model_type in ['smplx','flame']:
            self.comp_register('init_pose_jaw', torch.zeros((1,1*3), dtype=dtype), persistent=persistant_buffer)
            self.comp_register('init_pose_eye', torch.zeros((1,2*3), dtype=dtype), persistent=persistant_buffer)

        self.comp_register('init_betas', torch.zeros((1,num_betas), dtype=dtype), persistent=persistant_buffer)

        if self.use_dmpl:
            self.comp_register('init_dmpls', torch.zeros((1,num_dmpls), dtype=dtype), persistent=persistant_buffer)

    def comp_register(self, name, value, persistent=False):
        if sys.version_info[0] > 2:
            self.register_buffer(name, value, persistent)
        else:
            self.register_buffer(name, value)

    def r(self):
        from human_body_prior.tools.omni_tools import copy2cpu as c2c
        return c2c(self.forward().v)

    def forward(self, root_orient=None, pose_body=None, pose_hand=None, pose_jaw=None, pose_eye=None, betas=None,
                trans=None, dmpls=None, expression=None, v_template =None, joints=None, v_shaped=None, return_dict=False,  **kwargs):
        '''

        :param root_orient: Nx3
        :param pose_body:
        :param pose_hand:
        :param pose_jaw:
        :param pose_eye:
        :param kwargs:
        :return:
        '''
        batch_size = 1
        # compute batchsize by any of the provided variables
        for arg in [root_orient,pose_body,pose_hand,pose_jaw,pose_eye,betas,trans, dmpls,expression, v_template,joints]:
            if arg is not None:
                batch_size = arg.shape[0]
                break

        # assert not (v_template is not None and betas is not None), ValueError('vtemplate and betas could not be used jointly.')
        assert self.model_type in ['smpl', 'smplh', 'smplx', 'mano', 'animal_horse', 'animal_dog', 'flame', 'animal_rat'], ValueError(
            'model_type should be in smpl/smplh/smplx/mano')
        if root_orient is None:  root_orient = self.init_root_orient.expand(batch_size, -1)
        if self.model_type in ['smplh', 'smpl']:
            if pose_body is None:  pose_body = self.init_pose_body.expand(batch_size, -1)
            if pose_hand is None:  pose_hand = self.init_pose_hand.expand(batch_size, -1)
        elif self.model_type == 'smplx':
            if pose_body is None:  pose_body = self.init_pose_body.expand(batch_size, -1)
            if pose_hand is None:  pose_hand = self.init_pose_hand.expand(batch_size, -1)
            if pose_jaw is None:  pose_jaw = self.init_pose_jaw.expand(batch_size, -1)
            if pose_eye is None:  pose_eye = self.init_pose_eye.expand(batch_size, -1)
        elif self.model_type == 'flame':
            if pose_body is None:  pose_body = self.init_pose_body.expand(batch_size, -1)
            if pose_jaw is None:  pose_jaw = self.init_pose_jaw.expand(batch_size, -1)
            if pose_eye is None:  pose_eye = self.init_pose_eye.expand(batch_size, -1)
        elif self.model_type in ['mano',]:
            if pose_hand is None:  pose_hand = self.init_pose_hand.expand(batch_size, -1)
        elif self.model_type in ['animal_horse','animal_dog', 'animal_rat']:
            if pose_body is None:  pose_body = self.init_pose_body.expand(batch_size, -1)

        if pose_hand is None and self.model_type not in ['animal_horse', 'animal_dog', 'animal_rat','flame']:
            pose_hand = self.init_pose_hand.expand(batch_size, -1)

        if trans is None: trans = self.init_trans.expand(batch_size, -1)
        if v_template is None: v_template = self.init_v_template.expand(batch_size, -1,-1)
        if betas is None: betas = self.init_betas.expand(batch_size, -1)

        if self.model_type in ['smplh', 'smpl']:
            full_pose = torch.cat([root_orient, pose_body, pose_hand], dim=-1)
        elif self.model_type == 'smplx':
            full_pose = torch.cat([root_orient, pose_body, pose_jaw, pose_eye, pose_hand], dim=-1)  # orient:3, body:63, jaw:3, eyel:3, eyer:3, handl, handr
        elif self.model_type == 'flame':
            full_pose = torch.cat([root_orient, pose_body, pose_jaw, pose_eye], dim=-1)  # orient:3, body:63, jaw:3, eyel:3, eyer:3, handl, handr
        elif self.model_type in ['mano', ]:
            full_pose = torch.cat([root_orient, pose_hand], dim=-1)
        elif self.model_type in ['animal_horse', 'animal_dog', 'animal_rat']:
            full_pose = torch.cat([root_orient, pose_body], dim=-1)

        if self.use_dmpl:
            if dmpls is None: dmpls = self.init_dmpls.expand(batch_size, -1)
            shape_components = torch.cat([betas, dmpls], dim=-1)
            shapedirs = torch.cat([self.shapedirs, self.dmpldirs], dim=-1)
        elif self.use_expression:
            if expression is None: expression = self.init_expression.expand(batch_size, -1)
            shape_components = torch.cat([betas, expression], dim=-1)
            shapedirs = torch.cat([self.shapedirs, self.exprdirs], dim=-1)
        else:
            shape_components = betas
            shapedirs = self.shapedirs

        verts, Jtr = lbs(betas=shape_components, pose=full_pose, v_template=v_template,
                            shapedirs=shapedirs, posedirs=self.posedirs,
                            J_regressor=self.J_regressor, parents=self.kintree_table[0].long(),
                            lbs_weights=self.weights, joints=joints, v_shaped=v_shaped,
                            dtype=self.dtype)

        Jtr = Jtr + trans.unsqueeze(dim=1)
        verts = verts + trans.unsqueeze(dim=1)

        res = {}
        res['v'] = verts
        res['f'] = self.f
        res['Jtr'] = Jtr  # Todo: ik can be made with vposer
        # res['bStree_table'] = self.kintree_table

        # if self.model_type == 'smpl':
        #     res['pose_body'] = pose_body
        # elif self.model_type == 'smplh':
        #     res['pose_body'] = pose_body
        #     res['pose_hand'] = pose_hand
        # elif self.model_type == 'smplx':
        #     res['pose_body'] = pose_body
        #     res['pose_hand'] = pose_hand
        #     res['pose_jaw'] = pose_jaw
        #     res['pose_eye'] = pose_eye
        # elif self.model_type in ['mano', 'mano']:
        #     res['pose_hand'] = pose_hand
        res['full_pose'] = full_pose

        if not return_dict:
            class result_meta(object):
                pass

            res_class = result_meta()
            for k, v in res.items():
                res_class.__setattr__(k, v)
            res = res_class

        return res


