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

import glob
import os.path as osp

from human_body_prior.tools.configurations import load_config
from human_body_prior.train.vposer_trainer import train_vposer_once

def main():
    expr_id = 'V02_05'

    default_ps_fname = glob.glob(osp.join(osp.dirname(__file__), '*.yaml'))[0]

    vp_ps = load_config(default_ps_fname)

    vp_ps.train_parms.batch_size = 128

    vp_ps.general.expr_id = expr_id

    total_jobs = []
    total_jobs.append(vp_ps.toDict().copy())

    print('#training_jobs to be done: {}'.format(len(total_jobs)))
    if len(total_jobs) == 0:
        print('No jobs to be done')
        return

    for job in total_jobs:
        train_vposer_once(job)


if __name__ == '__main__':
    main()