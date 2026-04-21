"""Whole Body Tracking termination presets for the Statue v1 RMD robot."""

from holosoma.config_types.termination import TerminationManagerCfg, TerminationTermCfg

_STATUE_V1_RMD_BODY_NAMES_TO_TRACK = [
    "pelvis_link",
    "left_hip_roll_link",
    "left_knee_pitch_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_pitch_link",
    "right_ankle_roll_link",
    "waist_pitch_link",
    "left_shoulder_roll_link",
    "left_elbow_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_pitch_link",
    "right_wrist_yaw_link",
]

_STATUE_V1_RMD_BAD_MOTION_BODY_POS_BODY_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]

statue_v1_rmd_wbt_termination = TerminationManagerCfg(
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
                "bad_ref_pos_threshold": 0.8,
                "bad_ref_ori_threshold": 1.2,
                "bad_motion_body_pos_threshold": 0.4,
                "body_names_to_track": _STATUE_V1_RMD_BODY_NAMES_TO_TRACK,
                "bad_motion_body_pos_body_names": _STATUE_V1_RMD_BAD_MOTION_BODY_POS_BODY_NAMES,
                "bad_object_pos_threshold": 0.25,
                "bad_object_ori_threshold": 0.8,
            },
        ),
    }
)

statue_v1_rmd_wbt_termination_relaxed = TerminationManagerCfg(
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
                "bad_ref_pos_threshold": 3.0,
                "bad_ref_ori_threshold": 1.8,
                "bad_motion_body_pos_threshold": 1.5,
                "body_names_to_track": _STATUE_V1_RMD_BODY_NAMES_TO_TRACK,
                "bad_motion_body_pos_body_names": _STATUE_V1_RMD_BAD_MOTION_BODY_POS_BODY_NAMES,
                "bad_object_pos_threshold": 0.5,
                "bad_object_ori_threshold": 1.2,
            },
        ),
    }
)

__all__ = ["statue_v1_rmd_wbt_termination", "statue_v1_rmd_wbt_termination_relaxed"]
