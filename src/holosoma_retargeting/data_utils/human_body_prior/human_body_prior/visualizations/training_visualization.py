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

def pyrenderer(imw=2048, imh=2048):

    from body_visualizer.mesh.mesh_viewer import MeshViewer
    import cv2

    import numpy as np
    import trimesh

    try:
        mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
    except:
        import os
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
        os.environ['EGL_DEVICE_ID'] = os.environ['GPU_DEVICE_ORDINAL'].split(',')[0]

        mv = MeshViewer(width=imw, height=imh, use_offscreen=True)

    mv.set_cam_trans([0, -0.5, 2.])

    def render_an_image(meshes):
        n_all = len(meshes)
        nc = int(np.sqrt(n_all))

        out_image = np.zeros([1, 1, 1, mv.width, mv.height, 4])

        scale_percent = 100./nc
        width = int(mv.width * scale_percent / 100)
        height = int(mv.height * scale_percent / 100)
        dim = (width, height)

        for rId in range(nc):
            for cId in range(nc):
                i = (nc*rId) + cId
                if i>len(meshes): break

                mesh = meshes[i]

                # mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(-90), (1, 0, 0)))
                mesh.vertices -= np.median(np.array(mesh.vertices), axis=0)
                mv.set_dynamic_meshes([mesh])
                img = mv.render(render_wireframe=False, RGBA=True)
                img_resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

                out_image[0, 0, 0, (rId*width):((rId+1)*width), (cId*height):((cId+1)*height)] = cv2.cvtColor(img_resized, cv2.COLOR_BGRA2RGBA)

        return out_image.astype(np.uint8)

    return render_an_image

def vposer_trainer_renderer(bm, num_bodies_to_display=5):
    import numpy as np
    import trimesh
    import torch

    from body_visualizer.tools.vis_tools import imagearray2file, colors
    from human_body_prior.tools.omni_tools import copy2cpu as c2c
    from human_body_prior.tools.omni_tools import makepath
    from trimesh import Trimesh as Mesh
    from trimesh.util import concatenate as mesh_cat

    renderer = pyrenderer(1024, 1024)

    faces = c2c(bm.f)

    def render_once(body_parms, body_colors=[colors['grey'], colors['brown-light']], out_fname=None):
        '''

        :param body_parms: list of dictionaries of body parameters.
        :param body_colors: list of np arrays of color rgb values
        :param movie_outpath: a mp4 path
        :return:
        '''

        if out_fname is not None: makepath(out_fname, isfile=True)
        assert len(body_parms) <= len(body_colors), ValueError('Not enough colors provided for #{} body_parms'.format(len(body_parms)))

        bs = body_parms[0]['pose_body'].shape[0]

        body_ids = np.random.choice(bs, num_bodies_to_display)

        body_evals = [c2c(bm(root_orient=v['root_orient'].view(bs, -1) if 'root_orient' in v else torch.zeros(bs, 3).type_as(v['pose_body']),
                         pose_body=v['pose_body'].contiguous().view(bs, -1)).v) for v in body_parms]
        num_verts = body_evals[0].shape[1]

        render_meshes = []
        for bId in body_ids:
            concat_cur_meshes = None
            for body, body_color in zip(body_evals, body_colors):
                cur_body_mesh = Mesh(body[bId], faces, vertex_colors=np.ones([num_verts, 3]) * body_color)
                concat_cur_meshes = cur_body_mesh if concat_cur_meshes is None else mesh_cat(concat_cur_meshes, cur_body_mesh)
            render_meshes.append(concat_cur_meshes)

        img = renderer(render_meshes)

        if out_fname is not None: imagearray2file(img, out_fname, fps=10)


        return

    return render_once
