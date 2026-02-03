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
import os
import os.path as osp
import random
import sys
from pathlib import Path

import numpy as np
import torch


def copy2cpu(tensor):
    if isinstance(tensor, np.ndarray): return tensor
    return tensor.detach().cpu().numpy()


def create_list_chunks(list_, group_size, overlap_size, cut_smaller_batches=True):
    if cut_smaller_batches:
        return [list_[i:i + group_size] for i in range(0, len(list_), group_size - overlap_size) if
                len(list_[i:i + group_size]) == group_size]
    else:
        return [list_[i:i + group_size] for i in range(0, len(list_), group_size - overlap_size)]


def trainable_params_count(params):
    return sum([p.numel() for p in params if p.requires_grad])


def flatten_list(l):
    return [item for sublist in l for item in sublist]


SUPPORT_DATA_ENV = 'HUMAN_BODY_PRIOR_SUPPORT_DATA'


def get_support_data_dir(current_fname=__file__):
    """Return the absolute path to the packaged support_data directory.

    The lookup works for both editable installs (repo checkouts) and wheels
    installed into site-packages. Set the HUMAN_BODY_PRIOR_SUPPORT_DATA
    environment variable to override the auto detection (helpful on Windows
    when data lives outside the Python environment).
    """
    env_override = os.environ.get(SUPPORT_DATA_ENV)
    if env_override:
        env_path = Path(env_override).expanduser()
        if env_path.is_dir():
            return str(env_path.resolve())
        raise FileNotFoundError(
            f'{SUPPORT_DATA_ENV}={env_path} does not exist. '
            f'Update the environment variable or place support_data next to the package.'
        )

    current_path = Path(current_fname).resolve()
    for parent in current_path.parents:
        if parent.name == 'support_data' and parent.is_dir():
            return str(parent)
        candidate = parent / 'support_data'
        if candidate.is_dir():
            return str(candidate)

    raise FileNotFoundError(
        "Could not locate the 'support_data' directory. "
        "Place it next to the human_body_prior package or set "
        f'the {SUPPORT_DATA_ENV} environment variable.'
    )


def make_deterministic(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def id_generator(size=13):
    import string
    import random
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choice(chars) for _ in range(size))


def logger_sequencer(logger_list, prefix=None):
    def post_text(text):
        if prefix is not None: text = '{} -- '.format(prefix) + text
        for logger_call in logger_list: logger_call(text)

    return post_text


class log2file():
    def __init__(self, logpath=None, prefix='', auto_newline=True, write2file_only=False):
        if logpath is not None:
            makepath(logpath, isfile=True)
            self.fhandle = open(logpath, 'a+')
        else:
            self.fhandle = None

        self.prefix = prefix
        self.auto_newline = auto_newline
        self.write2file_only = write2file_only

    def __call__(self, text):
        if text is None: return
        if self.prefix != '': text = '{} -- '.format(self.prefix) + text
        # breakpoint()
        if self.auto_newline:
            if not text.endswith('\n'):
                text = text + '\n'
        if not self.write2file_only: sys.stderr.write(text)
        if self.fhandle is not None:
            self.fhandle.write(text)
            self.fhandle.flush()


def makepath(*args, **kwargs):
    '''
    if the path does not exist make it
    :param desired_path: can be path to a file or a folder name
    :return:
    '''
    isfile = kwargs.get('isfile', False)
    import os
    desired_path = os.path.join(*args)
    if isfile:
        if not os.path.exists(os.path.dirname(desired_path)): os.makedirs(os.path.dirname(desired_path))
    else:
        if not os.path.exists(desired_path): os.makedirs(desired_path)
    return desired_path


def matrot2axisangle(matrots):
    '''
    :param matrots: N*T*num_joints*9
    :return: N*T*num_joints*3
    '''
    import cv2
    N = matrots.shape[0]
    T = matrots.shape[1]
    n_joints = matrots.shape[2]
    out_axisangle = []
    for tIdx in range(T):
        T_axisangle = []
        for mIdx in range(N):
            cur_axisangle = []
            for jIdx in range(n_joints):
                cur_axisangle.append(cv2.Rodrigues(matrots[mIdx, tIdx, jIdx:jIdx + 1, :].reshape(3, 3))[0].T)
            T_axisangle.append(np.vstack(cur_axisangle)[np.newaxis])
        out_axisangle.append(np.vstack(T_axisangle).reshape([N, 1, -1, 3]))
    return np.concatenate(out_axisangle, axis=1)


def axisangle2matrots(axisangle):
    '''
    :param matrots: N*1*num_joints*3
    :return: N*num_joints*9
    '''
    import cv2
    batch_size = axisangle.shape[0]
    axisangle = axisangle.reshape([batch_size, 1, -1, 3])
    out_matrot = []
    for mIdx in range(axisangle.shape[0]):
        cur_axisangle = []
        for jIdx in range(axisangle.shape[2]):
            a = cv2.Rodrigues(axisangle[mIdx, 0, jIdx:jIdx + 1, :].reshape(1, 3))[0].T
            cur_axisangle.append(a)

        out_matrot.append(np.array(cur_axisangle).reshape([batch_size, 1, -1, 9]))
    return np.vstack(out_matrot)


def apply_mesh_tranfsormations_(meshes, transf):
    '''
    apply inplace translations to meshes
    :param meshes: list of trimesh meshes
    :param transf:
    :return:
    '''
    for i in range(len(meshes)):
        meshes[i] = meshes[i].apply_transform(transf)


def rm_spaces(in_text): return in_text.replace(' ', '_')
