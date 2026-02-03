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
# Code Developed by: Nima Ghorbani <https://www.linkedin.com/in/nghorbani/>
# 2018.01.02

import glob
import os
import os.path as osp

from omegaconf import OmegaConf
from loguru import logger

def exprdir2model(expr_dir, model_cfg_override: dict = None):
    if not os.path.exists(expr_dir): raise ValueError(f'Could not find the experiment directory: {expr_dir}')

    model_snapshots_dir = osp.join(expr_dir, 'snapshots')
    available_ckpts = sorted(glob.glob(osp.join(model_snapshots_dir, '*.ckpt')), key=osp.getmtime)
    assert len(available_ckpts) > 0, ValueError('No checkpoint found at {}'.format(model_snapshots_dir))
    trained_weights_fname = available_ckpts[-1]

    model_cfg_fname = glob.glob(osp.join('/', '/'.join(trained_weights_fname.split('/')[:-2]), '*.yaml'))
    if len(model_cfg_fname) == 0:
        model_cfg_fname = glob.glob(osp.join('/'.join(trained_weights_fname.split('/')[:-2]), '*.yaml'))

    model_cfg_fname = model_cfg_fname[0]
    model_cfg = OmegaConf.load(model_cfg_fname)
    if model_cfg_override:
        override_cfg_dotlist = [f'{k}={v}' for k, v in model_cfg_override.items()]
        override_cfg = OmegaConf.from_dotlist(override_cfg_dotlist)
        model_cfg = OmegaConf.merge(model_cfg, override_cfg)

    return model_cfg, trained_weights_fname

def load_model(expr_dir, model_code=None,
               remove_words_in_model_weights: str = None,
               load_only_cfg: bool = False,
               disable_grad: bool = True,
               model_cfg_override: dict = None,
               comp_device='gpu'):
    """

    :param expr_dir:
    :param model_code: an imported module
    from supercap.train.supercap_smpl import SuperCap, then pass SuperCap to this function
    :param if True will load the model definition used for training, and not the one in current repository
    :return:
    """
    import torch

    model_cfg, trained_weights_fname = exprdir2model(expr_dir, model_cfg_override=model_cfg_override)

    if load_only_cfg: return model_cfg

    assert model_code is not None, ValueError('model_code should be provided')
    model_instance = model_code(model_cfg)
    if disable_grad:  # i had to do this. torch.no_grad() couldnt achieve what i was looking for
        for param in model_instance.parameters():
            param.requires_grad = False

    if comp_device=='cpu' or not torch.cuda.is_available():
        logger.info('No GPU detected. Loading on CPU!')
        state_dict = torch.load(trained_weights_fname, map_location=torch.device('cpu'))['state_dict']
    else:
        state_dict = torch.load(trained_weights_fname)['state_dict']
    if remove_words_in_model_weights is not None:
        words = '{}'.format(remove_words_in_model_weights)
        state_dict = {k.replace(words, '') if k.startswith(words) else k: v for k, v in state_dict.items()}

    ## keys that were in the model trained file and not in the current model
    instance_model_keys = list(model_instance.state_dict().keys())
    # trained_model_keys = list(state_dict.keys())
    # wts_in_model_not_in_file = set(instance_model_keys).difference(set(trained_model_keys))
    ## keys that are in the current model not in the training weights
    # wts_in_file_not_in_model = set(trained_model_keys).difference(set(instance_model_keys))
    # assert len(wts_in_model_not_in_file) == 0, ValueError('Some model weights are not present in the pretrained file. {}'.format(wts_in_model_not_in_file))

    state_dict = {k: v for k, v in state_dict.items() if k in instance_model_keys}
    model_instance.load_state_dict(state_dict,strict=False)
    # Todo fix the issues so that we can set the strict to true. The body model uses unnecessary registered buffers
    model_instance.eval()
    logger.info(f'Loaded model in eval mode with trained weights: {trained_weights_fname}')
    return model_instance,  model_cfg
