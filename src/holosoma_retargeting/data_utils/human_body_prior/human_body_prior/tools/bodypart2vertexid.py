from psbody.mesh.meshviewer import MeshViewer
from psbody.mesh import Mesh
import numpy as np





def find_handVertexIDs(blend_weights, all_partIds, interested_partIds):

    segm = np.argmax(blend_weights, axis=1)               # n_vertex
    num2part = {v: k for k, v in all_partIds.items()}    # n_joints
    print(num2part)
    vert2part = [num2part[i] for i in segm]                 # 6890

    # handPartIDs = np.arange(20, len(num2part))      # SMPL+HH --> B
   #handPartIDs = [20] + range(22, 36 + 1)          # SMPL+HH --> L
   #handPartIDs = [21] + range(37, len(num2part))   # SMPL+HH --> R
   #handPartIDs = [15]  # head things
    #
   #handPartIDs = [20, 21] + np.arange(25, len(num2part)).tolist()   # SMPL+HF --> B
   #handPartIDs = [20] + np.arange(25, 40).tolist()                  # SMPL+HF --> L
   #handPartIDs = [21] + np.arange(40, len(num2part)).tolist()       # SMPL+HF --> R
   #handPartIDs = [15, 22, 23, 24]  # head things

    PartLABELs = [num2part[ii] for ii in interested_partIds]
    VertexIDs = [ii for ii in range(len(vert2part)) if vert2part[ii] in PartLABELs]

    # test
    allOK = all(vert2part[VertexID] in PartLABELs for VertexID in VertexIDs)
    print('allOK =', allOK)
    print('isSorted =', all(VertexIDs[i] <= VertexIDs[i+1] for i in range(len(VertexIDs)-1)))

    return VertexIDs

def smplx_part_ids():
    from human_body_prior.tools.omni_tools import copy2cpu as c2c
    from body_visualizer.tools.vis_tools import colors
    import torch

    from human_body_prior.body_model.body_model import BodyModel
    # all_partids = np.load('/ps/project/common/moshpp/smplx/part2num.npy', allow_pickle=True).tolist()
    bm = BodyModel(bm_fname='/ps/project/common/moshpp/smplx/locked_head/model_6_merged_exp_hands_fixed_eyes/neutral/model.npz')
    # bm = BodyModel(bm_fname='/ps/scratch/soma/support_files/smplx_downsampled/328/female/model.npz')

    # joints = np.load('/ps/project/common/moshpp/smplx/locked_head/model_6_merged_exp_hands_fixed_eyes/neutral/model.npz')['joints']
    joints = np.load('/ps/project/supercap/support_files/smplx/smplx_downsampled/328/female/model.npz')['joints']
    joints = torch.from_numpy(joints)

    # bm = BodyModel(bm_fname='/ps/scratch/common/moshpp/smplx/locked_head/model_6_merged_exp_hands_fixed_eyes/female/model.npz')
    # all_partids = np.load('/ps/project/common/moshpp/smplx/part2num.npy', allow_pickle=True).tolist()
    # print(all_partids)
    #
    # from psbody.smpl.serialization import load_model
    # model = load_model(fname_or_dict='/ps/body/projects/faces/fullbody_hand_head_models/SMPL+HF/trained_models/init_low_res_fixed_neck/male/model_0.pkl')  # not 6 !!!
    # model.part2num = part2num_body_hand_face

    # smplx_partids = {'body': [0,1,2,3,4,5,6,7,8,9,10,11,13,14,16,17,18,19],
    #                 'face': [15, 12, 22],
    #                 'eyeballs': [23, 24],
    #                 'hand': list(range(20,22)),
    #                 'finger': list(range(25, 55)),
    #                 'handR': [21] + list(range(40, 55))
    #                 }
    smplx_partids = {
                    'body': [0,1,2,3,4,5,6,9,13,14,16,17,18,19],
                    'face': [12, 15, 22],
                    'eyeball': [23, 24],
                     'leg': [4, 5, 7, 8, 10, 11],
                     'arm': [18, 19, 20, 21],
                    'handl': [20] + list(range(25, 40)),
                    'handr': [21] + list(range(40, 55)),
                    'footl': [7,10],
                    'footr': [8,11],
                     'ftip': [27, 30, 33, 36, 39, 42, 45, 48, 51, 54]

                     }

    all_partids = {}
    for bk, jids in smplx_partids.items():
        for jid in jids:
            all_partids['%s_%02d'%(bk, jid)] = jid
            print('%s_%02d'%(bk, jid))

    body_part_vc = {'body': colors['yellow'], 'arm': colors['orange'],'face': colors['green'],'leg': colors['green'],
                    'ftip':colors['white'],
                    'footl': colors['blue'],  'footr': colors['blue'],
                    'handl': colors['red'],  'handr': colors['orange'],

                    }

    part2vids = {}
    for partname, partids in smplx_partids.items():
        vertex_ids = find_handVertexIDs(c2c(bm.weights), all_partids, partids)
        vertex_ids = np.array(sorted(vertex_ids))
        part2vids[partname] = vertex_ids

    # body_v = c2c(bm().v[0])
    body_v = c2c(bm(joints=joints).v[0])

    part2vids['all'] = np.arange(0,body_v.shape[0])
    # part2vids.pop('eyeballs')

    np.savez('/ps/project/common/moshpp/smplx/part2vids.npz', **part2vids)
    # np.savez('/ps/scratch/soma/support_files/smplx_downsampled/328/part2vids_v2v_errs.npz', **part2vids)


    from psbody.mesh.meshviewer import MeshViewer
    mv = MeshViewer(keepalive=True)
    meshes = [Mesh(v=body_v[part2vids[partname]], f=[], vc=part_vc) for partname, part_vc in body_part_vc.items()]
    mv.set_static_meshes(meshes)

#
#
def smplh_part_ids():
    from human_body_prior.tools.omni_tools import copy2cpu as c2c
    from body_visualizer.tools.vis_tools import colors

    from human_body_prior.body_model.body_model import BodyModel
    bm = BodyModel(bm_fname='/ps/scratch/common/moshpp/smplh/locked_head/female/model.npz')

    smplx_partids = {'body': [0, 1, 2, 3, 4, 5, 6, 9, 13, 14, 16, 17, 18, 19,22,23,24,],
                     'face': [15, 12],
                     'handl': [20] + list(range(22, 37)),
                     'handr': [21] + list(range(37, 52)),
                     # 'leg': [4,5,7,8],
                     # 'arm': [18,19,20,21],
                     'footl': [7, 10],
                     'footr': [8, 11],
                     # 'finger': list(range(22, 52)),
                     }
    all_partids = {}
    for bk, jids in smplx_partids.items():
        for jid in jids:
            all_partids['%s_%02d' % (bk, jid)] = jid
            print('%s_%02d' % (bk, jid))

    body_part_vc = {k:v for k,v in {'body': colors['yellow'],
                                    # 'arm': colors['orange'],
                                    'face': colors['green'],
                                    # 'leg': colors['green'],
                    # 'ftipl':colors['white'],
                    # 'ftipr':colors['pink'],
                    'footl': colors['brown'],  'footr': colors['blue'],
                    'handl': colors['pink'],  'handr': colors['orange'],

                    }.items() if k in smplx_partids}
    part2vids = {}
    for partname, partids in smplx_partids.items():
        vertex_ids = find_handVertexIDs(c2c(bm.weights), all_partids, partids)
        vertex_ids = np.array(sorted(vertex_ids))
        part2vids[partname] = vertex_ids

    body_v = c2c(bm().v[0])

    part2vids['all'] = np.arange(0, body_v.shape[0])
    np.savez('/ps/scratch/common/moshpp/smplh/part2vids.npz', **part2vids)

    from psbody.mesh.meshviewer import MeshViewer
    mv = MeshViewer(keepalive=True)
    meshes = [Mesh(v=body_v[part2vids[partname]], f=[], vc=part_vc) for partname, part_vc in body_part_vc.items()]
    mv.set_static_meshes(meshes)
    mv.save_snapshot('/ps/scratch/common/moshpp/smplh/part2vids.jpeg')
# #
# def mano_part_ids():
#     from human_body_prior.tools.omni_tools import copy2cpu as c2c
#     from human_body_prior.tools.omni_tools import colors
#
#     from human_body_prior.body_model.body_model import BodyModel
#     bm = BodyModel(bm_fname='/ps/scratch/common/moshpp/mano/MANO_LEFT.npz')
#
#     smplx_partids = {'hand': [0, 1],
#                      'finger': [15,3,6,12,9,14,2,5,11,8],
#                      }
#     smplx_partids['all_others'] = list(set(range(16)).difference(set([i for v in list(smplx_partids.values()) for i in v])))
#     all_partids = {}
#     for bk, jids in smplx_partids.items():
#         for jid in jids:
#             all_partids['%s_%02d' % (bk, jid)] = jid
#             print('%s_%02d' % (bk, jid))
#
#     body_part_vc = {'all_others': colors['orange'], 'finger': colors['blue'], 'hand': colors['red']}
#
#     part2vids = {}
#     for partname, partids in smplx_partids.items():
#         vertex_ids = find_handVertexIDs(c2c(bm.weights), all_partids, partids)
#         vertex_ids = np.array(sorted(vertex_ids))
#         part2vids[partname] = vertex_ids
#
#     body_v = c2c(bm().v[0])
#
#     part2vids['all'] = np.arange(0, body_v.shape[0])
#     np.savez('/ps/scratch/common/moshpp/mano/part2vids.npz', **part2vids)
#
#     from psbody.mesh.meshviewer import MeshViewer
#     mv = MeshViewer(keepalive=True)
#     meshes = [Mesh(v=body_v[part2vids[partname]], f=[], vc=part_vc) for partname, part_vc in body_part_vc.items()]
#     mv.set_static_meshes(meshes)

if __name__ == '__main__':
    # smplx_part_ids()
    smplh_part_ids()
    # mano_part_ids()

    