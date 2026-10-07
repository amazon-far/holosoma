from pathlib import Path

import holosoma
from holosoma.sensors.warp.camera_config.base_depth_camera_config import BaseDepthCameraConfig

_G1_MESH_ROOT = Path(holosoma.__file__).parent / "data" / "robots" / "g1" / "meshes"

G1_ASSET_MESHES_ROOT = str(_G1_MESH_ROOT)


class RsD435iConfig(BaseDepthCameraConfig):
    num_sensors = 1  # number of sensors of this type

    # camera params VFOV is calcuated from the aspect ratio and HFOV
    # VFOV = 2 * atan(tan(HFOV/2) / aspect_ratio)
    width = 106  # 80
    height = 60  # 45
    horizontal_fov_deg = 89.5  # 101
    max_range = 3.0
    min_range = 0.3
    crop_top = 2
    crop_left = 4
    crop_right = 4

    dynamic_meshes = True

    # NOTE: [DELTA] randomize position and rotation of the sensor
    randomize_placement = True
    min_translation = {
        "cam_front_depth": [-0.025, -0.025, -0.025],
    }
    max_translation = {
        "cam_front_depth": [0.025, 0.025, 0.025],
    }
    min_euler_rotation_deg = {
        "cam_front_depth": [-2.5, -3.0, -2.5],
    }
    max_euler_rotation_deg = {
        "cam_front_depth": [2.5, 3.0, 2.5],
    }

    offset_rot_base = [-90.0, 0, -90.0]  # roll, pitch, yaw [deg]


class G1FlatRsD435iConfig(RsD435iConfig):
    asset_meshes_root = G1_ASSET_MESHES_ROOT

    # transform from sensor element coordinate frame to sensor_base_link frame
    offset = {
        "cam_front_depth": {
            "offset_pos": (0.01, 0.01, 0.44),  # (0.15,0.06,0.02),
            "offset_rot": (1.0, 27.0, 1.0),
        }
    }

    # camera base link frame - can support multiple cameras
    base_link_frame = {
        "cam_front_depth": "torso_link",
    }

    ray_cast_bodies = {
        "pelvis": "combined_pelvis.STL",
        # 'pelvis_contour_link': 'pelvis_contour_link.STL',
        "left_hip_pitch_link": "left_hip_pitch_link.STL",
        "left_hip_roll_link": "left_hip_roll_link.STL",
        "left_hip_yaw_link": "left_hip_yaw_link.STL",
        "left_knee_link": "left_knee_link.STL",
        "left_ankle_pitch_link": "left_ankle_pitch_link.STL",
        "left_ankle_roll_link": "left_ankle_roll_link.STL",
        "right_hip_pitch_link": "right_hip_pitch_link.STL",
        "right_hip_roll_link": "right_hip_roll_link.STL",
        "right_hip_yaw_link": "right_hip_yaw_link.STL",
        "right_knee_link": "right_knee_link.STL",
        "right_ankle_pitch_link": "right_ankle_pitch_link.STL",
        "right_ankle_roll_link": "right_ankle_roll_link.STL",
        "waist_yaw_link": "waist_yaw_link_rev_1_0.STL",
        "waist_roll_link": "waist_roll_link_rev_1_0.STL",
        # 'torso_link': 'combined_torso_head.STL',
        # 'logo_link': 'logo_link.STL',
        # 'head_link': 'head_link.STL',
        "left_shoulder_pitch_link": "left_shoulder_pitch_link.STL",
        "left_shoulder_roll_link": "left_shoulder_roll_link.STL",
        "left_shoulder_yaw_link": "left_shoulder_yaw_link.STL",
        "left_elbow_link": "left_elbow_link.STL",
        "left_wrist_roll_link": "left_wrist_roll_link.STL",
        "left_wrist_pitch_link": "left_wrist_pitch_link.STL",
        "left_wrist_yaw_link": "combined_left_wrist_spherehand.STL",
        # 'left_sphere_hand_link': 'half_sphere.obj',
        "right_shoulder_pitch_link": "right_shoulder_pitch_link.STL",
        "right_shoulder_roll_link": "right_shoulder_roll_link.STL",
        "right_shoulder_yaw_link": "right_shoulder_yaw_link.STL",
        "right_elbow_link": "right_elbow_link.STL",
        "right_wrist_roll_link": "right_wrist_roll_link.STL",
        "right_wrist_pitch_link": "right_wrist_pitch_link.STL",
        "right_wrist_yaw_link": "combined_right_wrist_spherehand.STL",
        # 'right_sphere_hand_link': 'half_sphere.obj',
    }

    add_offpath_obstacle = False
    offpath_obstacle_meshes_root = None
    offpath_obstacle_bodies = {}
