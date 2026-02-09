"""Whole Body Tracking termination presets for the Statue robot."""

from holosoma.config_types.termination import TerminationManagerCfg, TerminationTermCfg

statue_v0_1_wbt_termination = TerminationManagerCfg(
    terms={
        "timeout": TerminationTermCfg(
            func="holosoma.managers.termination.terms.common:timeout_exceeded",
            is_timeout=True,
        ),
        "motion_ends": TerminationTermCfg(
            func="holosoma.managers.termination.terms.wbt:motion_ends",
        ),
        "bad_tracking": TerminationTermCfg(
            func="holosoma.managers.termination.terms.wbt:BadTracking",
            params={
                # robot tracking
                "bad_ref_pos_threshold": 0.5,
                "bad_ref_ori_threshold": 0.8,
                "bad_motion_body_pos_threshold": 0.25,
                # NOTE: body_names_to_track is shared with command_manager
                # Updated for Statue robot body names
                "body_names_to_track": [
                    "pelvis_link",  # Changed from "pelvis"
                    "left_hip_roll_link",
                    "left_knee_pitch_link",  # Changed from "left_knee_link"
                    "left_ankle_roll_link",
                    "right_hip_roll_link",
                    "right_knee_pitch_link",  # Changed from "right_knee_link"
                    "right_ankle_roll_link",
                    "waist_pitch_link",  # Changed from "torso_link"
                    "left_shoulder_roll_link",
                    "left_elbow_pitch_link",  # Changed from "left_elbow_link"
                    "left_wrist_yaw_link",
                    "right_shoulder_roll_link",
                    "right_elbow_pitch_link",  # Changed from "right_elbow_link"
                    "right_wrist_yaw_link",
                ],
                "bad_motion_body_pos_body_names": [
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                    "left_wrist_yaw_link",
                    "right_wrist_yaw_link",
                ],
                # object tracking
                # only triggered when has_object=True
                "bad_object_pos_threshold": 0.25,
                "bad_object_ori_threshold": 0.8,
            },
        ),
    }
)

__all__ = ["statue_v0_1_wbt_termination"]
