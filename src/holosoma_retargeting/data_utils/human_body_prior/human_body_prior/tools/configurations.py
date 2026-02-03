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
from dotmap import DotMap
import os
import yaml

def load_config(default_ps_fname=None, **kwargs):
    if isinstance(default_ps_fname, str):
        assert os.path.exists(default_ps_fname), FileNotFoundError(default_ps_fname)
        assert default_ps_fname.lower().endswith('.yaml'), NotImplementedError('Only .yaml files are accepted.')
        default_ps = yaml.safe_load(open(default_ps_fname, 'r'))
    else:
        default_ps = {}

    default_ps.update(kwargs)

    return DotMap(default_ps, _dynamic=False)

def dump_config(data, fname):
    '''
    dump current configuration to an ini file
    :param fname:
    :return:
    '''
    with open(fname, 'w') as file:
        yaml.dump(data.toDict(), file)
    return fname
