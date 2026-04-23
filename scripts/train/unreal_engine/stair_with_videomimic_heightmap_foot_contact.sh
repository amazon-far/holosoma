source scripts/source_isaacsim_setup.sh

# ----------------------------------------------------------------------------
# VideoMimic-style heightmap + foot_contact_match reward + critic-only privileged
# foot-contact observations (actual + target). Built on top of the existing
# videomimic heightmap exp — the original exp is untouched.
#
# Additions over exp:g1-29dof-wbt-future-motion-heightmap-videomimic
#   * Observation (critic only, actor unchanged):
#       actual_foot_contact : (N, 2) binary, from contact_forces on
#                             left/right_foot_contact_point (threshold=1.0 N).
#       target_foot_contact : (N, 2) binary, from the npz `foot_contact` field.
#   * Reward:
#       foot_contact_match = Σ (actual == target)   # range {0,1,2} per env
#       weight = 0.5        (VideoMimic feet_contact_matching style)
#
# Other rewards / weights / sigmas (g1_29dof_wbt_reward, unchanged):
#     motion_global_ref_position_error_exp            sigma=0.3,  weight=0.5
#     motion_global_ref_orientation_error_exp         sigma=0.4,  weight=0.5
#     motion_relative_body_position_error_exp         sigma=0.3,  weight=1.0
#     motion_relative_body_orientation_error_exp      sigma=0.4,  weight=1.0
#     motion_global_body_lin_vel                      sigma=1.0,  weight=1.0
#     motion_global_body_ang_vel                      sigma=3.14, weight=1.0
#     action_rate_l2                                              weight=-0.1
#     limits_dof_pos            soft_dof_pos_limit=0.9            weight=-100.0
#     penalty_dof_acc                                             weight=0.0 (off)
#     undesired_contacts        threshold=1.0                     weight=-0.5
#
# Motion uses the newly produced npz containing the `foot_contact` (T,2) key:
#     g1_stair_30depth_20260420_foot_contact.npz
# ----------------------------------------------------------------------------
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt-future-motion-heightmap-videomimic-foot-contact \
    terrain:terrain-load-obj \
    logger:wandb \
    --logger.entity draftrec \
    --logger.project MotionTracking \
    --logger.group=statue_wbt \
    --logger.name=g1_stair_30depth_with_videomimic_heightmap_foot_contact \
    --terrain.terrain-term.obj-file-path="holosoma/data/motions/g1_29dof/whole_body_tracking/unreal_engine/g1_stair_30depth_20260420_foot_contact.npz" \
    --terrain.terrain-term.env-origin-in-tile="[0.0, 0.0, 0.05]" \
    --terrain.terrain-term.tile-mesh False \
    --command.setup_terms.motion_command.params.motion_config.motion_file="holosoma/data/motions/g1_29dof/whole_body_tracking/unreal_engine/g1_stair_30depth_20260420_foot_contact.npz" \
    --command.setup_terms.motion_command.params.motion_config.enable_default_pose_prepend False \
    --command.setup_terms.motion_command.params.motion_config.enable_default_pose_append False \
    --termination.terms.bad_tracking.params.bad_ref_pos_threshold 0.5 \
    --termination.terms.bad_tracking.params.bad_ref_ori_threshold 0.8 \
    --termination.terms.bad_tracking.params.bad_motion_body_pos_threshold 0.5 \
    --simulator.config.scene.env_spacing=0.0 \
    --logger.video.enabled False \
    --algo.config.save_interval 1000 \
    robot:g1-29dof-diverse-action-scale
