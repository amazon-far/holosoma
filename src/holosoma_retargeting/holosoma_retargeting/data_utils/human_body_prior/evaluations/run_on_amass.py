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
# AMASS: Archive of Motion Capture as Surface Shapes <https://arxiv.org/abs/1904.03278>
#
#
# Code Developed by:
# Nima Ghorbani <https://www.linkedin.com/in/nghorbani/>
#
# 2019.05.28

import json
import os

import torch
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.data.dataloader import VPoserDS
from human_body_prior.tools.model_loader import load_vposer
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.tools.omni_tools import makepath
from human_body_prior.train.vposer_smpl import VPoserTrainer
from torch.utils.data import DataLoader
from tqdm import tqdm


def evaluate_model(dataset_dir, vp_model, vp_ps, batch_size=5, save_upto_bnum=10, splitname='test'):
    assert splitname in ['test', 'train', 'vald']
    comp_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ds_name = dataset_dir.split('/')[-2]

    vp_model.eval()
    vp_model = vp_model.to(comp_device)

    with torch.no_grad():
        bm = BodyModel(vp_ps.bm_fname, batch_size=1, num_betas=16).to(comp_device)

    ds = VPoserDS(dataset_dir=os.path.join(dataset_dir, splitname))
    ds = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

    outpath = os.path.join(vp_ps.work_dir, 'evaluations', 'ds_%s'%ds_name, os.path.basename(vp_ps.best_model_fname).replace('.pt',''), '%s_samples'%splitname)
    print('dumping to %s'%outpath)

    for bId, dorig in enumerate(ds):
        dorig = {k: dorig[k].to(comp_device) for k in dorig.keys()}

        imgpath = makepath(os.path.join(outpath, '%s-%03d.png' % (vp_ps.expr_code, bId)), isfile=True)
        VPoserTrainer.vis_results(dorig, bm, vp_model, imgpath, view_angles=[0, 180])#, view_angles = [0, 180, 90])

        if bId> save_upto_bnum: break


def evaluate_error(dataset_dir, vp_model, vp_ps, batch_size=512):
    vp_model.eval()

    ds_name = dataset_dir.split('/')[-2]

    comp_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    bm = BodyModel(vp_ps.bm_fname, batch_size=batch_size, num_betas=16).to(comp_device)
    vp_model = vp_model.to(comp_device)

    # from psbody.mesh import Mesh, MeshViewer
    # from human_body_prior.tools.omni_tools import colors
    # import time
    # mv = MeshViewer()

    final_errors = {}
    # for splitname in ['test']:
    for splitname in ['test', 'train', 'vald']:

        ds = VPoserDS(dataset_dir=os.path.join(dataset_dir, splitname))
        print('%s dataset size: %s'%(splitname,len(ds)))
        ds = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=True)#batchsize for bm is fixed so drop the last one

        loss_mean = []
        with torch.no_grad():
            for dorig in tqdm(ds):
                dorig = {k: dorig[k].to(comp_device) for k in dorig.keys()}

                MESH_SCALER = 1000

                drec = vp_model(**dorig)
                for k in dorig: # whatever field is missing from drec copy from dorig
                    if k not in drec:
                        drec[k] = dorig[k]
                        if len(final_errors)==0 and len(loss_mean) == 0: 
                            print('Field %s is not predicted by the model and is copied from the original data.'%k)

                with torch.no_grad():
                    body_orig = bm(**dorig).v
                    body_rec = bm(**drec).v

                # body_orig_mesh = Mesh(c2c(body_orig[0]), c2c(bm.f), vc= colors['blue'])
                # body_rec_mesh = Mesh(c2c(body_rec[0]), c2c(bm.f), vc=colors['red'])
                # mv.set_dynamic_meshes([body_rec_mesh, body_orig_mesh])
                # time.sleep(0.2)

                # loss_mean.append(torch.mean(torch.sqrt(torch.pow((mesh_orig - mesh_rec)* MESH_SCALER, 2))))
                loss_mean.append(torch.mean(torch.abs(body_orig - body_rec)* MESH_SCALER))

        final_errors[splitname] = {'v2v_mae': float(c2c(torch.stack(loss_mean).mean()))}
        print(splitname, final_errors[splitname])

    outpath = makepath(os.path.join(vp_ps.work_dir, 'evaluations', 'ds_%s'%ds_name, os.path.basename(vp_ps.best_model_fname).replace('.pt','.json')), isfile=True)
    with open(outpath, 'w') as f:
        json.dump(final_errors,f)

    return final_errors

if __name__ == '__main__':
    expr_code = '008_SV01_T00'
    # data_code = '007_00_00'

    expr_dir = '/ps/project/human_body_prior/VPoser/smpl/pytorch/%s'%expr_code

    vp_model, vp_ps = load_vposer(expr_dir)
    dataset_dir = vp_ps.dataset_dir
    # dataset_dir = '/ps/project/human_body_prior/VPoser/data/%s/smpl/pytorch/stage_III'%data_code

    print('dataset_dir: %s'%dataset_dir)
    # # # for splitname in ['test']:
    # for splitname in ['train', 'test', 'vald']:
    #    evaluate_model(dataset_dir, vp_model, vp_ps, batch_size=3, save_upto_bnum=5, splitname=splitname)

    final_errors = evaluate_error(dataset_dir, vp_model, vp_ps, batch_size=512)
    print('[%s] [DS: %s] -- %s' % (vp_ps.best_model_fname, dataset_dir,  ', '.join(['%s: %.2e'%(k, v['v2v_mae']) for k,v in final_errors.items()])))

