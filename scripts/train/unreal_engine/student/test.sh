source scripts/source_isaacsim_setup.sh

MOTION_DIR="src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/Unreal_Engine_stair_subset200_aligned"
TEACHER_CKPT="logs/MotionTracking/20260426_g1_ue_multi_terrain_stair_200_future_motion_heightmap_xyz/model_22000.pt"

python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-multi-terrain-videomimic-student-dagger \
    logger:disabled \
    --algo.config.policy_to_clone="$TEACHER_CKPT" \
    --simulator.config.videomimic-height-map.map-height 11 \
    --simulator.config.videomimic-height-map.map-width 17 \
    --simulator.config.videomimic-height-map.resolution 0.1 \
    --simulator.config.videomimic-height-map.num-channels 3 \
    --terrain.terrain-term.obj-dir="$MOTION_DIR" \
    --terrain.terrain-term.num-rows 1 \
    --terrain.terrain-term.obj-max-faces-per-tile 3000 \
    --terrain.terrain-term.obj-tile-gap 1.0 \
    --terrain.terrain-term.env-origin-in-tile="[0.0, 0.0, 0.0]" \
    --command.setup_terms.motion_command.params.motion_config.motion_dir="$MOTION_DIR" \
    --command.setup_terms.motion_command.params.motion_config.max_motions 5 \
    --command.setup_terms.motion_command.params.motion_config.noise_to_initial_pose.init_root_offset="[0.0, 0.0, 0.1]" \
    --training.num_envs 10 \
    --training.headless False \
    --simulator.config.scene.env_spacing=0.0 \
    --algo.config.save_interval 1000 \
    --algo.config.eval_interval 0
