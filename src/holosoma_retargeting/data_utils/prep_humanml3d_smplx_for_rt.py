
from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "human_body_prior"))

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from human_body_prior.body_model.body_model import BodyModel  # type: ignore[import-not-found]


def load_ori_npz_file(npz_file_path, dest_fps=30):
    """
    >>> import numpy as np
    >>> data = np.load("Jog_1_stageii.npz")
    >>> data.files
    ['gender', 'surface_model_type', 'mocap_frame_rate', 'mocap_time_length',
    'markers_latent', 'latent_labels', 'markers_latent_vids',
    'trans', 'poses', 'betas', 'num_betas',
    'root_orient', 'pose_body', 'pose_hand', 'pose_jaw', 'pose_eye']
    """
    data = np.load(npz_file_path)
    ori_fps = data["mocap_frame_rate"]

    downsample_ratio = int(ori_fps / dest_fps)

    return {
        "gender": data["gender"],
        "fps": dest_fps,
        "trans": data["trans"][::downsample_ratio],
        "poses": data["poses"][::downsample_ratio],
        "betas": data["betas"],
        "num_betas": data["num_betas"],
        "root_orient": data["root_orient"][::downsample_ratio],
        "pose_body": data["pose_body"][::downsample_ratio],
        "pose_hand": data["pose_hand"][::downsample_ratio],
        "pose_jaw": data["pose_jaw"][::downsample_ratio],
        "pose_eye": data["pose_eye"][::downsample_ratio],
    }


def run_smplx_model(
    root_trans,
    aa_rot_rep,
    betas,
    gender,
    bm_dict,
):
    # root_trans: BS X T X 3
    # aa_rot_rep: BS X T X 22 X 3
    # betas: BS X 16
    # gender: BS
    bs, num_steps, num_joints, _ = aa_rot_rep.shape
    if num_joints != 52:
        padding_zeros_hand = torch.zeros(bs, num_steps, 30, 3).to(aa_rot_rep.device)  # BS X T X 30 X 3
        aa_rot_rep = torch.cat((aa_rot_rep, padding_zeros_hand), dim=2)  # BS X T X 52 X 3

    aa_rot_rep = aa_rot_rep.reshape(bs * num_steps, -1, 3)  # (BS*T) X n_joints X 3

    betas = betas[:, None, :].repeat(1, num_steps, 1).reshape(bs * num_steps, -1)  # (BS*T) X 16
    gender = np.asarray(gender)[:, np.newaxis].repeat(num_steps, axis=1)
    gender = gender.reshape(-1).tolist()  # (BS*T)

    smpl_trans = root_trans.reshape(-1, 3)  # (BS*T) X 3
    smpl_betas = betas  # (BS*T) X 16
    smpl_root_orient = aa_rot_rep[:, 0, :]  # (BS*T) X 3
    smpl_pose_body = aa_rot_rep[:, 1:22, :].reshape(-1, 63)  # (BS*T) X 63
    smpl_pose_hand = aa_rot_rep[:, 22:, :].reshape(-1, 90)  # (BS*T) X 90

    B = smpl_trans.shape[0]  # (BS*T)

    smpl_vals = [
        smpl_trans,
        smpl_root_orient,
        smpl_betas,
        smpl_pose_body,
        smpl_pose_hand,
    ]
    # batch may be a mix of genders, so need to carefully use the corresponding SMPL body model
    # gender_names = ["male", "female", "neutral"]
    gender_names = ["neutral"]  # We use neutral gender for all the data in G1 setting
    pred_joints = []
    pred_verts = []
    prev_nbidx = 0
    cat_idx_map = np.ones((B), dtype=np.int64) * -1
    for gender_name in gender_names:
        gender_idx = np.array(gender) == gender_name
        nbidx = np.sum(gender_idx)

        cat_idx_map[gender_idx] = np.arange(prev_nbidx, prev_nbidx + nbidx, dtype=np.int64)
        prev_nbidx += nbidx

        gender_smpl_vals = [val[gender_idx] for val in smpl_vals]

        if nbidx == 0:
            # skip if no frames for this gender
            continue

        # reconstruct SMPL
        (
            cur_pred_trans,
            cur_pred_orient,
            cur_betas,
            cur_pred_pose,
            cur_pred_pose_hand,
        ) = gender_smpl_vals
        bm = bm_dict[gender_name]

        pred_body = bm(
            pose_body=cur_pred_pose,
            pose_hand=cur_pred_pose_hand,
            betas=cur_betas,
            root_orient=cur_pred_orient,
            trans=cur_pred_trans,
        )

        pred_joints.append(pred_body.Jtr)
        pred_verts.append(pred_body.v)

    x_pred_smpl_joints = torch.cat(pred_joints, axis=0)  # () X 52 X 3

    x_pred_smpl_joints = x_pred_smpl_joints[cat_idx_map]  # (BS*T) X 22 X 3

    x_pred_smpl_verts = torch.cat(pred_verts, axis=0)
    x_pred_smpl_verts = x_pred_smpl_verts[cat_idx_map]  # (BS*T) X 6890 X 3

    x_pred_smpl_joints = x_pred_smpl_joints.reshape(bs, num_steps, -1, 3)  # BS X T X 22 X 3/BS X T X 24 X 3
    x_pred_smpl_verts = x_pred_smpl_verts.reshape(bs, num_steps, -1, 3)  # BS X T X 6890 X 3

    mesh_faces = pred_body.f

    return x_pred_smpl_joints, x_pred_smpl_verts, mesh_faces


def prep_smplx_model(model_root_folder):
    # Prepare SMPLX model
    support_base_dir = model_root_folder
    surface_model_type = "smplx"
    surface_model_male_fname = os.path.join(support_base_dir, surface_model_type, "SMPLX_MALE.npz")
    surface_model_female_fname = os.path.join(support_base_dir, surface_model_type, "SMPLX_FEMALE.npz")
    surface_model_neutral_fname = os.path.join(support_base_dir, surface_model_type, "SMPLX_NEUTRAL.npz")
    dmpl_fname = None
    num_dmpls = None
    num_expressions = None
    num_betas = 16

    male_bm = BodyModel(
        bm_fname=surface_model_male_fname,
        num_betas=num_betas,
        num_expressions=num_expressions,
        num_dmpls=num_dmpls,
        dmpl_fname=dmpl_fname,
    )
    female_bm = BodyModel(
        bm_fname=surface_model_female_fname,
        num_betas=num_betas,
        num_expressions=num_expressions,
        num_dmpls=num_dmpls,
        dmpl_fname=dmpl_fname,
    )
    neutral_bm = BodyModel(
        bm_fname=surface_model_neutral_fname,
        num_betas=num_betas,
        num_expressions=num_expressions,
        num_dmpls=num_dmpls,
        dmpl_fname=dmpl_fname,
    )
    return {
        "male": male_bm,
        "female": female_bm,
        "neutral": neutral_bm,
    }


def compute_height(bm_dict, betas, gender):
    """
    Compute height by running SMPLX model in T-pose and measuring vertex height.

    Args:
        bm_dict: Dictionary of BodyModel instances
        betas: Shape parameters (1, 16) or (16,)
        gender: Gender string ('male' or 'female')

    Returns:
        float: Height in meters (max_z - min_z of vertices)
    """
    rest_root_trans = torch.zeros(1, 1, 3)
    rest_poses = torch.zeros(1, 1, 52, 3)
    rest_jnts, rest_verts, mesh_faces = run_smplx_model(
        root_trans=rest_root_trans, aa_rot_rep=rest_poses, betas=betas, gender=gender, bm_dict=bm_dict
    )

    rest_jnts = rest_jnts.squeeze(0).squeeze(0).detach().cpu().numpy()
    rest_verts = rest_verts.squeeze(0).squeeze(0).detach().cpu().numpy()

    # Compute height as max_y - min_y
    # (Use y because it seems when root orientation is 0, the SMPL model is not standing along z axis)
    min_z = np.min(rest_verts[:, 1])
    max_z = np.max(rest_verts[:, 1])

    return max_z - min_z


def get_npz_files(amass_root_folder, subdataset_folder=None):
    """
    Get all npz files from the amass root folder.

    Args:
        amass_root_folder: Root folder containing AMASS SMPLX npz files
        subdataset_folder: Optional subdataset folder name. If specified, only loads
            npz files from amass_root_folder/subdataset_folder/*/*.npz.
            If None, loads from amass_root_folder/**/*.npz (recursive).

    Returns:
        List of npz file paths
    """
    amass_path = Path(amass_root_folder)
    if subdataset_folder is not None:
        # Load from amass_root_folder/subdataset_folder/*/*.npz
        search_path = amass_path / subdataset_folder
        npz_files = [str(p) for p in search_path.rglob("*_stageii.npz")]
    else:
        # Load from amass_root_folder/**/*.npz (recursive)
        npz_files = [str(p) for p in amass_path.rglob("*_stageii.npz")]
    return npz_files


@dataclass
class Config:
    """Configuration for processing AMASS SMPLX data."""

    amass_root_folder: str = "/home/ubuntu/datasets/rt_ori_human_data/amass-smplx"
    """Root folder containing AMASS SMPLX npz files."""

    output_folder: str = "/home/ubuntu/datasets/rt_processed_data/amass-smplx-processed"
    """Output folder for processed data."""

    model_root_folder: str = "/home/ubuntu/datasets/rt_ori_human_data/smpl_all_models"
    """Root folder containing SMPLX model files."""

    subdataset_folder: str | None = None
    """Optional subdataset folder name. If specified, only loads npz files from
    amass_root_folder/subdataset_folder/*/*.npz. If None, loads from
    amass_root_folder/**/*.npz (recursive)."""


def main(cfg: Config):
    # Get all the npz file paths in the amass-smplx folder
    npz_file_paths = get_npz_files(cfg.amass_root_folder, cfg.subdataset_folder)

    # Prepare desired output folder
    os.makedirs(cfg.output_folder, exist_ok=True)

    bm_dict = prep_smplx_model(cfg.model_root_folder)

    num_body_joints = 22
    for npz_file_path in npz_file_paths:
        data = load_ori_npz_file(npz_file_path)
        gender = data["gender"]
        betas = data["betas"]  # 16
        root_trans = data["trans"]  # T X 3
        aa_rot_rep = data["poses"]  # T X 165 (55*3)
        aa_rot_52 = aa_rot_rep.reshape(-1, 55, 3)[:, :52, :]  # T X 52 X 3

        # Convert numpy to tensor
        root_trans = torch.from_numpy(root_trans).float()[None]  # 1 X T X 3
        aa_rot_52 = torch.from_numpy(aa_rot_52).float()[None]  # 1 X T X 52 X 3
        betas = torch.from_numpy(betas).float()[None]  # 1 X 16

        # Run FK to obtain global joint positions and global joint rotations
        global_joint_positions, global_joint_verts, mesh_faces = run_smplx_model(
            root_trans=root_trans, aa_rot_rep=aa_rot_52, betas=betas, gender=[gender], bm_dict=bm_dict
        )

        global_joint_positions = (
            global_joint_positions.squeeze(0).detach().cpu().numpy()[:, :num_body_joints, :]
        )  # T X 55 X 3

        # Compute height based on min_z and max_z value of all the vertices
        height = compute_height(bm_dict, betas, gender=[gender])
        print(f"Height: {height}")

        # Save the processed data to the output folder
        npz_path = Path(npz_file_path)
        subset_data_name = npz_path.parts[-3]
        sub_name = npz_path.parts[-2]
        output_file_path = os.path.join(cfg.output_folder, subset_data_name + "_" + sub_name + "_" + npz_path.name)
        np.savez(output_file_path, global_joint_positions=global_joint_positions, height=height)
        print(f"Saved processed data to {output_file_path}")

        # break

    print("All data processed successfully")


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
