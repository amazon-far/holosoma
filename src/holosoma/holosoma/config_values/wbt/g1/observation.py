"""Whole Body Tracking observation presets for the G1 robot."""

from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg

actor_obs_shared = ObsGroupCfg(
    concatenate=True,
    enable_noise=True,
    history_length=1,
    terms={
        "motion_command": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:motion_command",
            scale=1.0,
            noise=0.0,
        ),
        "motion_ref_ori_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:motion_ref_ori_b",
            scale=1.0,
            noise=0.05,
        ),
        "base_ang_vel": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:base_ang_vel",
            scale=1.0,
            noise=0.2,
        ),
        "dof_pos": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:dof_pos",
            scale=1.0,
            noise=0.01,
        ),
        "dof_vel": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:dof_vel",
            scale=1.0,
            noise=0.5,
        ),
        "actions": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:actions",
            scale=1.0,
            noise=0.0,
        ),
    },
)

critic_obs_shared_terms = {
    "motion_command": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:motion_command",
        scale=1.0,
        noise=0.0,
    ),
    "motion_ref_pos_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:motion_ref_pos_b",
        scale=1.0,
        noise=0.25,
    ),
    "motion_ref_ori_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:motion_ref_ori_b",
        scale=1.0,
        noise=0.05,
    ),
    "robot_body_pos_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:robot_body_pos_b",
        scale=1.0,
        noise=0.0,
    ),
    "robot_body_ori_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:robot_body_ori_b",
        scale=1.0,
        noise=0.0,
    ),
    "base_lin_vel": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:base_lin_vel",
        scale=1.0,
        noise=0.0,
    ),
    "base_ang_vel": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:base_ang_vel",
        scale=1.0,
        noise=0.2,
    ),
    "dof_pos": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:dof_pos",
        scale=1.0,
        noise=0.01,
    ),
    "dof_vel": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:dof_vel",
        scale=1.0,
        noise=0.5,
    ),
    "actions": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:actions",
        scale=1.0,
        noise=0.0,
    ),
}

critic_obs_w_object_terms = critic_obs_shared_terms.copy()
critic_obs_w_object_terms.update(
    {
        "obj_pos_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_pos_b",
            scale=1.0,
            noise=0.0,
        ),
        "obj_ori_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_ori_b",
            scale=1.0,
            noise=0.0,
        ),
        "obj_lin_vel_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_lin_vel_b",
            scale=1.0,
            noise=0.0,
        ),
    }
)

g1_29dof_wbt_observation = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_shared,
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_shared_terms,
        ),
    },
)

g1_29dof_wbt_observation_w_object = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_shared,
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_w_object_terms,
        ),
    },
)

# Future motion observation config
# Includes future_motion_targets for motion encoder
future_motion_obs_group = ObsGroupCfg(
    concatenate=True,
    enable_noise=False,
    history_length=1,
    terms={
        "future_motion_targets": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:future_motion_targets",
            scale=1.0,
            noise=0.0,
            params={"future_num_steps": 20, "future_max_steps": 95},
        ),
    },
)

g1_29dof_wbt_observation_future_motion = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_shared,
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_shared_terms,
        ),
        "future_motion_targets": future_motion_obs_group,
    },
)

# Future motion without local key body positions (ablation)
future_motion_obs_group_no_key_body = ObsGroupCfg(
    concatenate=True,
    enable_noise=False,
    history_length=1,
    terms={
        "future_motion_targets": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:future_motion_targets",
            scale=1.0,
            noise=0.0,
            params={
                "future_num_steps": 20,
                "future_max_steps": 95,
                "include_key_body_pos": False,
            },
        ),
    },
)

g1_29dof_wbt_observation_future_motion_no_key_body = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_shared,
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_shared_terms,
        ),
        "future_motion_targets": future_motion_obs_group_no_key_body,
    },
)

# Future motion + height map observation config
# Includes future_motion_targets for motion encoder AND height_map_obs for terrain perception
height_map_obs_group = ObsGroupCfg(
    concatenate=True,
    enable_noise=False,
    history_length=1,
    terms={
        "height_map_obs": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:height_map_obs",
            scale=1.0,
            noise=0.0,
            params={"height_scan_offset": 0.5, "min_height": -5.0},
        ),
    },
)

g1_29dof_wbt_observation_future_motion_heightmap = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_shared,
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_shared_terms,
        ),
        "future_motion_targets": future_motion_obs_group,
        "height_map_obs": height_map_obs_group,
    },
)

# Critic-only privileged foot-contact observations. Actor does not see these
# (keeps the asymmetric actor-critic setup); critic gets both the actual robot
# contact and the motion-target contact so it can learn a tight value baseline
# around the foot_contact_match reward.
critic_obs_with_foot_contact_terms = dict(critic_obs_shared_terms)
critic_obs_with_foot_contact_terms["actual_foot_contact"] = ObsTermCfg(
    func="holosoma.managers.observation.terms.wbt:actual_foot_contact",
    scale=1.0,
    noise=0.0,
    params={"threshold": 1.0},
)
critic_obs_with_foot_contact_terms["target_foot_contact"] = ObsTermCfg(
    func="holosoma.managers.observation.terms.wbt:target_foot_contact",
    scale=1.0,
    noise=0.0,
)

g1_29dof_wbt_observation_future_motion_heightmap_foot_contact = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_shared,
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_with_foot_contact_terms,
        ),
        "future_motion_targets": future_motion_obs_group,
        "height_map_obs": height_map_obs_group,
    },
)

# VideoMimic-style heightmap obs group.
# Uses the 11x11 `videomimic_height_map_scanner` sensor and returns only the
# height channel (no x_local/y_local) — matching VideoMimic HeightfieldSensor
# `depth_map` exactly. Output dim = 121 (vs 561 for the 3-channel version).
videomimic_height_map_obs_group = ObsGroupCfg(
    concatenate=True,
    enable_noise=False,
    history_length=1,
    terms={
        "videomimic_height_map_obs": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:videomimic_height_map_obs",
            scale=1.0,
            noise=0.0,
            params={"height_scan_offset": 0.5, "min_height": -5.0},
        ),
    },
)

g1_29dof_wbt_observation_future_motion_heightmap_videomimic = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_shared,
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_shared_terms,
        ),
        "future_motion_targets": future_motion_obs_group,
        "height_map_obs": videomimic_height_map_obs_group,
    },
)

g1_29dof_wbt_observation_future_motion_heightmap_videomimic_foot_contact = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_shared,
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_with_foot_contact_terms,
        ),
        "future_motion_targets": future_motion_obs_group,
        "height_map_obs": videomimic_height_map_obs_group,
    },
)

# ──────────────────────────────────────────────────────────────────────────────
# DAgger student obs (VideoMimic-style)
#
# Mirrors VideoMimic's `G1DeepMimicCfgRootHeightfieldNoHistoryDagger` actor
# obs: proprio (current step) + root pos/yaw error in body frame + heightmap.
# The student does NOT see future motion targets (no `motion_command`,
# no `future_motion_targets` group on the actor side).
#
# - `motion_ref_pos_b` ↔ VideoMimic's `torso_xy_rel` (3D pos in body frame)
# - `motion_ref_ori_b` ↔ VideoMimic's `torso_yaw_rel` (6D ori → includes yaw)
# - heightmap is the same VideoMimic-style 11×17 xyz scanner the teacher uses
# - `teacher_actor_obs` re-publishes the teacher's full actor obs slice so the
#   DAgger algo can query the frozen teacher each step; it is NOT seen by the
#   student.
# ──────────────────────────────────────────────────────────────────────────────
actor_obs_videomimic_student_terms = {
    # current-step root pos error in body frame (~ torso_xy_rel + z)
    "motion_ref_pos_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:motion_ref_pos_b",
        scale=1.0,
        noise=0.05,
    ),
    # current-step root orientation error (6D rep — yaw/roll/pitch via 2 cols)
    "motion_ref_ori_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:motion_ref_ori_b",
        scale=1.0,
        noise=0.05,
    ),
    "base_ang_vel": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:base_ang_vel",
        scale=1.0,
        noise=0.2,
    ),
    "dof_pos": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:dof_pos",
        scale=1.0,
        noise=0.01,
    ),
    "dof_vel": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:dof_vel",
        scale=1.0,
        noise=0.5,
    ),
    "actions": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:actions",
        scale=1.0,
        noise=0.0,
    ),
}

actor_obs_videomimic_student = ObsGroupCfg(
    concatenate=True,
    enable_noise=True,
    history_length=1,
    terms=actor_obs_videomimic_student_terms,
)

g1_29dof_wbt_observation_videomimic_student = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_videomimic_student,
        # teacher's actor obs (= the original `actor_obs_shared` content).
        # Re-published under a separate key so the DAgger algo can route it to
        # the frozen teacher without colliding with the student's actor_obs.
        "teacher_actor_obs": actor_obs_shared,
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_shared_terms,
        ),
        "future_motion_targets": future_motion_obs_group,
        "height_map_obs": videomimic_height_map_obs_group,
    },
)


# ──────────────────────────────────────────────────────────────────────────────
# VideoMimic-narrowed actor obs (xy + yaw command only)
#
# Same as `actor_obs_videomimic_student_terms` but the motion-derived
# command is collapsed to (xy, yaw) in body frame, matching VideoMimic's
# `torso_xy_rel` (2D) + `torso_yaw_rel` (1D) interface. Lets us deploy the
# actor with a 3-DoF command (gamepad / joystick) by feeding the same
# slots from a user input source instead of a motion clip — and keeps the
# step-function pelvis-z signal out of the actor entirely.
# ──────────────────────────────────────────────────────────────────────────────
actor_obs_videomimic_student_xy_yaw_terms = {
    # 2D body-frame xy of the next ref pose (drops z; was 3D in
    # `motion_ref_pos_b`).
    "motion_ref_pos_xy_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:motion_ref_pos_xy_b",
        scale=1.0,
        noise=0.05,
    ),
    # 2D (cos, sin) yaw error (collapses the 6D `motion_ref_ori_b` to yaw
    # only and avoids the ±π wrap discontinuity).
    "motion_ref_yaw_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:motion_ref_yaw_b",
        scale=1.0,
        noise=0.05,
    ),
    "base_ang_vel": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:base_ang_vel",
        scale=1.0,
        noise=0.2,
    ),
    "dof_pos": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:dof_pos",
        scale=1.0,
        noise=0.01,
    ),
    "dof_vel": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:dof_vel",
        scale=1.0,
        noise=0.5,
    ),
    "actions": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:actions",
        scale=1.0,
        noise=0.0,
    ),
}

actor_obs_videomimic_student_xy_yaw = ObsGroupCfg(
    concatenate=True,
    enable_noise=True,
    history_length=1,
    terms=actor_obs_videomimic_student_xy_yaw_terms,
)

g1_29dof_wbt_observation_videomimic_student_xy_yaw = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_videomimic_student_xy_yaw,
        # Teacher still receives the full motion-mimic obs so DAgger
        # distillation from the existing teacher works unchanged.
        "teacher_actor_obs": actor_obs_shared,
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_shared_terms,
        ),
        "future_motion_targets": future_motion_obs_group,
        "height_map_obs": videomimic_height_map_obs_group,
    },
)


__all__ = [
    "g1_29dof_wbt_observation",
    "g1_29dof_wbt_observation_w_object",
    "g1_29dof_wbt_observation_future_motion",
    "g1_29dof_wbt_observation_future_motion_no_key_body",
    "g1_29dof_wbt_observation_future_motion_heightmap",
    "g1_29dof_wbt_observation_future_motion_heightmap_foot_contact",
    "g1_29dof_wbt_observation_future_motion_heightmap_videomimic",
    "g1_29dof_wbt_observation_future_motion_heightmap_videomimic_foot_contact",
    "g1_29dof_wbt_observation_videomimic_student",
    "g1_29dof_wbt_observation_videomimic_student_xy_yaw",
]
