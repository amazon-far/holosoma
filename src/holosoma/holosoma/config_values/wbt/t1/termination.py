"""Whole Body Tracking termination presets for the T1-23dof robot."""

from holosoma.config_types.termination import TerminationManagerCfg, TerminationTermCfg

# ===================================================================
# Termination Configuration for T1-23DOF Whole Body Tracking
# ===================================================================

t1_23dof_wbt_termination = TerminationManagerCfg(
    terms={
        # Timeout termination - ends episode after fixed duration
        "timeout": TerminationTermCfg(
            func="holosoma.managers.termination.terms.common:timeout_exceeded",
            is_timeout=True,  # Marks this as a timeout termination
        ),
        
        # Motion end termination - ends when reference motion completes
        "motion_ends": TerminationTermCfg(
            func="holosoma.managers.termination.terms.wbt:motion_ends",
        ),
        
        # Bad tracking termination - triggers when tracking error exceeds thresholds
        "bad_tracking": TerminationTermCfg(
            func="holosoma.managers.termination.terms.wbt:BadTracking",
            params={
                # Global reference tracking thresholds
                "bad_ref_pos_threshold": 0.5,    # Meters - maximum allowed position error
                "bad_ref_ori_threshold": 0.8,    # Radians - maximum allowed orientation error
                
                # Body position tracking threshold
                "bad_motion_body_pos_threshold": 0.25,  # Meters - per-body position error limit
                
                # Body names to monitor for tracking performance
                # NOTE: These are specific to T1-23dof robot model
                "body_names_to_track": [
                    "Trunk",          # Main torso body
                    "Waist",          # Waist joint body
                    
                    # Left leg tracking bodies
                    "Shank_Left",     # Left shank
                    "left_foot_link", # Left foot
                    
                    # Right leg tracking bodies  
                    "Shank_Right",    # Right shank
                    "right_foot_link", # Right foot
                    
                    # Left arm tracking bodies
                    "AL3",            # Left arm link 3 (elbow/wrist area)
                    "left_hand_link", # Left hand
                    
                    # Right arm tracking bodies
                    "AR3",            # Right arm link 3 (elbow/wrist area)
                    "right_hand_link", # Right hand
                ],
                
                # Specific bodies with stricter position thresholds
                # Used for foot and hand placement accuracy
                "bad_motion_body_pos_body_names": [
                    "left_foot_link",   # Critical for stability
                    "right_foot_link",  # Critical for stability
                    "left_hand_link",   # Important for manipulation tasks
                    "right_hand_link",  # Important for manipulation tasks
                ],
                
                # Object tracking thresholds (only used when object is present)
                "bad_object_pos_threshold": 0.25,    # Meters - object position error limit
                "bad_object_ori_threshold": 0.8,     # Radians - object orientation error limit
            },
        ),
    }
)

__all__ = ["t1_23dof_wbt_termination"]

"""
Note: This termination configuration is specifically designed for the T1-23dof robot.
Key thresholds have been tuned based on the robot's kinematics and typical motion tracking
performance. Adjustments may be needed for different motion types or environmental conditions.
"""
