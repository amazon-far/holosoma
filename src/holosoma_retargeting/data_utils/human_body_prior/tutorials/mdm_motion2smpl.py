# -*- coding: utf-8 -*-
#
# Code Developed by:
# Nima Ghorbani <https://nghorbani.github.io/>
#
# 2022.10.20
# SMPL-X Solver for MDM: Human Motion Diffusion Model

import os.path as osp
from pathlib import Path
from typing import List, Dict
from typing import Union

import numpy as np
import torch
from colour import Color
from loguru import logger
from scipy.spatial.transform import Rotation as R
from torch import nn

from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.models.ik_engine import IK_Engine
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.tools.omni_tools import create_list_chunks
from tqdm import tqdm
from body_visualizer.tools.vis_tools import render_smpl_params
from body_visualizer.tools.vis_tools import imagearray2file
import os.path as osp
from glob import glob

import numpy as np
import torch
from loguru import logger
from human_body_prior.tools.omni_tools import get_support_data_dir

from body_visualizer.tools.vis_tools import imagearray2file
from body_visualizer.tools.vis_tools import render_smpl_params
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import get_support_data_dir
class SourceKeyPoints(nn.Module):
    def __init__(self,
                 bm: Union[str, BodyModel],
                 n_joints: int = 22,
                 kpts_colors: Union[np.ndarray, None] = None,
                 num_betas=16
                 ):
        super(SourceKeyPoints, self).__init__()

        self.bm = BodyModel(bm, num_betas=num_betas, persistant_buffer=False) if isinstance(bm, str) else bm
        self.bm_f = []  # self.bm.f
        self.n_joints = n_joints
        self.kpts_colors = np.array(
            [Color('grey').rgb for _ in range(n_joints)]) if kpts_colors == None else kpts_colors

    def forward(self, body_parms):
        new_body = self.bm(**body_parms)

        return {'source_kpts': new_body.Jtr[:, :self.n_joints], 'body': new_body}


def transform_smpl_coordinate(bm_fname: Path, trans: np.ndarray,
                              root_orient: np.ndarray, betas: np.ndarray, rotxyz: Union[np.ndarray, List]) -> Dict:
    """
    rotates smpl parameters while taking into account non-zero center of rotation for smpl
    Parameters
    ----------
    bm_fname: body model filename
    trans: Nx3
    root_orient: Nx3
    betas: num_betas
    rotxyz: desired XYZ rotation in degrees

    Returns
    -------

    """
    if isinstance(rotxyz, list):
        rotxyz = np.array(rotxyz).reshape(1, 3)
    if betas.ndim == 1: betas = betas[None]
    if betas.ndim == 2 and betas.shape[0] != 1:
        logger.warning(
            f'betas should be the same for the entire sequence. 2D np.array with 1 x num_betas: {betas.shape}. taking the mean')
        betas = np.mean(betas, keepdims=True, axis=0)
    transformation_euler = np.deg2rad(rotxyz)

    coord_change_matrot = R.from_euler('XYZ', transformation_euler.reshape(1, 3)).as_matrix().reshape(3, 3)
    bm = BodyModel(bm_fname=bm_fname,
                   num_betas=betas.shape[1])
    pelvis_offset = c2c(bm(**{'betas': torch.from_numpy(betas).type(torch.float32)}).Jtr[[0], 0])

    root_matrot = R.from_rotvec(root_orient).as_matrix().reshape([-1, 3, 3])

    transformed_root_orient_matrot = np.matmul(coord_change_matrot, root_matrot.T).T
    transformed_root_orient = R.from_matrix(transformed_root_orient_matrot).as_rotvec()
    transformed_trans = np.matmul(coord_change_matrot, (trans + pelvis_offset).T).T - pelvis_offset

    return {'root_orient': transformed_root_orient.astype(np.float32),
            'trans': transformed_trans.astype(np.float32), }


def convert_mdm_mp4_to_amass_npz(skeleton_movie_fname, out_fname=None, save_render=False, comp_device='cuda:0',
                                 surface_model_type = 'smplx', gender = 'neutral', batch_size=128, verbosity=0):
    """

    :param skeleton_movie_fname: either a result npy file or a motion numpy array [nframes, njoints, 3]
    :param surface_model_type:
    :param gender:
    :param batch_size:
    :param verbosity: 0: silent, 1: text, 2: visual with psbody.mesh
    :return:
    """

    support_base_dir = Path(get_support_data_dir())
    support_dir = support_base_dir / 'dowloads'#'../../../support_data/dowloads'
    logger.info(f'found support_dir: {support_dir}')
    # 'TRAINED_MODEL_DIRECTORY'  in this directory the trained model along with the model code exist
    vposer_expr_dir = str((support_dir / 'vposer_v2_05').resolve())

    # 'PATH_TO_SMPLX_model.npz'  obtain from https://smpl-x.is.tue.mpg.de/downloads
    bm_fname = str((support_dir / 'models' / surface_model_type / gender / 'model.npz').resolve())

    if isinstance(skeleton_movie_fname, np.ndarray):
        assert out_fname is not None, 'when passing motion file out_fname should be provided'
        motion = skeleton_movie_fname
    else:
        assert osp.exists(skeleton_movie_fname), skeleton_movie_fname
        parsed_name = osp.basename(skeleton_movie_fname).replace('.mp4', '').replace('sample', '').replace('rep', '')

        sample_idx, rep_idx = [int(e) for e in parsed_name.split('_')]
        mdm_fname = osp.join(osp.dirname(skeleton_movie_fname), 'results.npy')
        mdm_data = np.load(mdm_fname, allow_pickle=True).tolist()
        total_num_samples = mdm_data['num_samples']
        absl_idx = rep_idx * total_num_samples + sample_idx
        motion = mdm_data['motion'][absl_idx].transpose(2, 0, 1)  # [nframes, njoints, 3]

    if out_fname is None:
        out_fname = skeleton_movie_fname.replace('.mp4', '.npz')

    render_out_fname = out_fname.replace('.npz', f'_{surface_model_type}.mp4')
    if osp.exists(render_out_fname):
        logger.warning(f'render output already exists: {render_out_fname}. skipping...')
        return
    n_joints = 22
    num_betas = 16

    if osp.exists(out_fname):
        d = np.load(out_fname)
    else:
        # comp_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        red = Color("red")
        blue = Color("blue")
        kpts_colors = [c.rgb for c in list(red.range_to(blue, n_joints))]

        # create source and target key points and make sure they are index aligned
        data_loss = torch.nn.MSELoss(reduction='sum')

        stepwise_weights = [
            {'data': 10., 'poZ_body': .01, 'betas': .5},
        ]

        optimizer_args = {'type': 'LBFGS', 'max_iter': 300, 'lr': 1, 'tolerance_change': 1e-4, 'history_size': 200}
        ik_engine = IK_Engine(vposer_expr_dir=vposer_expr_dir,
                              verbosity=verbosity,
                              display_rc=(2, 2),
                              data_loss=data_loss,
                              num_betas=num_betas,
                              stepwise_weights=stepwise_weights,
                              optimizer_args=optimizer_args).to(comp_device)

        all_results = {}
        batched_frames = create_list_chunks(np.arange(len(motion)), batch_size, overlap_size=0, cut_smaller_batches=False)
        if verbosity<2:
            batched_frames = tqdm(batched_frames, desc='VPoser Advanced IK')
        for cur_frame_ids in batched_frames:

            target_pts = torch.from_numpy(motion[cur_frame_ids, :n_joints]).to(comp_device)
            source_pts = SourceKeyPoints(bm=bm_fname, n_joints=n_joints, kpts_colors=kpts_colors, num_betas=num_betas).to(
                comp_device)

            ik_res = ik_engine(source_pts, target_pts, {})

            ik_res_detached = {k: c2c(v) for k, v in ik_res.items()}
            nan_mask = np.isnan(ik_res_detached['trans']).sum(-1) != 0
            if nan_mask.sum() != 0: raise ValueError('Sum results were NaN!')
            for k, v in ik_res_detached.items():
                if k not in all_results: all_results[k] = []
                all_results[k].append(v)

        d = {k: np.concatenate(v, axis=0) for k, v in all_results.items()}
        d['betas'] = np.median(d['betas'], axis=0)

        transformed_d = transform_smpl_coordinate(bm_fname=bm_fname, trans=d['trans'], root_orient=d['root_orient'],
                                                  betas=d['betas'], rotxyz=[90, 0, 0])
        d.update(transformed_d)
        d['poses'] = np.concatenate([d['root_orient'], d['pose_body'], np.zeros([len(d['pose_body']), 99])], axis=1)

        d['surface_model_type'] = surface_model_type
        d['gender'] = gender
        d['mocap_frame_rate'] = 30
        d['num_betas'] = num_betas
        np.savez(out_fname, **d)
        logger.success(f'created: {out_fname}')

    if save_render:
        bm = BodyModel(bm_fname=bm_fname, num_betas=num_betas)
        smpl_dict = np.load(bm_fname)
        mean_pose_hand = np.repeat(np.concatenate([smpl_dict['hands_meanl'], smpl_dict['hands_meanr']])[None], axis=0, repeats=len(motion))

        body_parms = {**d, 'betas': np.repeat(d['betas'][None], axis=0, repeats=len(motion)), 'pose_hand':mean_pose_hand}
        body_parms = {k:torch.from_numpy(v) for k,v in body_parms.items() if k in ['root_orient', 'trans', 'pose_body', 'pose_hand']}

        img_array = render_smpl_params(bm, body_parms, [-90, 0, 0])[None, None]
        imagearray2file(img_array, outpath=render_out_fname, fps=30)
        logger.success(f'created: {render_out_fname}')

    logger.info(f'You can visualize these results as any amass npz file or in Blender via blender_smplx_addon.')

if __name__ == '__main__':
    import argparse
    from glob import glob
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help='skeleton movie.mp4 filename that is to be converted into SMPL')
    parser.add_argument("--pattern", type=str, help='filename pattern for skeleton */*/movies.mp4 to be converted into SMPL')
    parser.add_argument("--batch_size", type=int, default=128, help='batch size for inverse kinematics')
    parser.add_argument("--model_type", type=str, default='smplx', help='model_type; e.g. smplx/smpl')
    parser.add_argument("--device", type=str, default='cuda:0', help='computation device')
    parser.add_argument("--gender", type=str, default='neutral', help='gender; e.g. neutral')
    parser.add_argument("--save_render", type=bool, default=True, help='render IK results')
    parser.add_argument("--verbosity", type=int, default=0, help='0: silent, 1: text, 2: display')
    params = parser.parse_args()
    # params = {
    #     'input':'/home/nima/opt/code-repos/motion-diffusion-model/save/humanml_trans_enc_512/samples_humanml_trans_enc_512_000200000_seed10/sample00_rep00.mp4',
    #          'save_render': True
    #           }
    if (params.input is None) and (params.pattern is None):
        raise ValueError('either input or pattern should be provided')
    if not params.input is None:
        convert_mdm_mp4_to_amass_npz(skeleton_movie_fname=params.input,
                                 surface_model_type=params.model_type,
                                 gender=params.gender,
                                 batch_size=params.batch_size,
                                 save_render=params.save_render,
                                 verbosity=params.verbosity)
    else:
        assert params.pattern is not None
        mp4_fnames = glob(params.pattern)
        print(f'found {len(mp4_fnames)} mp4 files')
        for mp4_fname in mp4_fnames:

            if mp4_fname.endswith('_smplx.mp4'):
                print(f'skipping smplx render file: {mp4_fname}')
                continue

            convert_mdm_mp4_to_amass_npz(skeleton_movie_fname=mp4_fname,
                                         surface_model_type=params.model_type,
                                         gender=params.gender,
                                         batch_size=params.batch_size,
                                         save_render=params.save_render,
                                         verbosity=params.verbosity)
