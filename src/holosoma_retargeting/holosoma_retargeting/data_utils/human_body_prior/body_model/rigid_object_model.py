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
from human_body_prior.body_model.lbs import batch_rodrigues

try:
    from psbody.mesh import Mesh as _PsbodyMesh
except ImportError:
    _PsbodyMesh = None

try:
    import trimesh
except ImportError:
    trimesh = None


def _load_rigid_mesh(path):
    if _PsbodyMesh is not None:
        mesh = _PsbodyMesh(filename=path)
        return mesh.v, mesh.f.astype(np.int32)
    if trimesh is not None:
        tri_mesh = trimesh.load(path, process=False)
        if isinstance(tri_mesh, list):
            tri_mesh = trimesh.util.concatenate(tri_mesh)
        if hasattr(tri_mesh, 'geometry'):
            tri_mesh = trimesh.util.concatenate(tuple(tri_mesh.geometry.values()))
        if not isinstance(tri_mesh, trimesh.Trimesh):
            raise TypeError(f'Unsupported mesh container returned by trimesh for {path}')
        faces = np.asarray(tri_mesh.faces, dtype=np.int32)
        verts = np.asarray(tri_mesh.vertices)
        return verts, faces
    raise ImportError(
        "RigidObjectModel requires either `psbody.mesh` (install via `human_body_prior[psbody]`, Linux only) "
        "or the `trimesh` fallback provided by the `vis` extra."
    )

class RigidObjectModel(nn.Module):

    def __init__(self, plpath, batch_size=1, dtype=torch.float32):
        super(RigidObjectModel, self).__init__()

        trans = torch.tensor(np.zeros((batch_size, 3)), dtype=dtype, requires_grad=True)
        self.register_parameter('trans', nn.Parameter(trans, requires_grad=True))

        root_orient = torch.tensor(np.zeros((batch_size, 3)), dtype=dtype, requires_grad=True)
        self.register_parameter('root_orient', nn.Parameter(root_orient, requires_grad=True))

        verts, faces = _load_rigid_mesh(plpath)

        self.rigid_v = torch.from_numpy(np.repeat(verts[np.newaxis], batch_size, axis=0)).type(dtype)
        self.f = torch.from_numpy(faces)

    def forward(self, root_orient, trans):
        if root_orient is None: root_orient = self.root_orient
        if trans is None: trans = self.trans
        verts = torch.bmm(self.rigid_v, batch_rodrigues(root_orient)) + trans.view(-1,1,3)

        res = {}
        res['v'] = verts
        res['f'] = self.f

        class result_meta(object): pass

        res_class = result_meta()
        for k, v in res.items():
            res_class.__setattr__(k, v)
        res = res_class

        return res
