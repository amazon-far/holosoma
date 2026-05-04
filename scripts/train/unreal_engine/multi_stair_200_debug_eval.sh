source scripts/source_isaacsim_setup.sh

# Debug script: triggers SuccessRateCallback eval at iter 10 to validate OOM fix
# (success_rate_callback.py: pool resize capped to min(num_envs, num_motions)).
# Same num_envs / max_motions as multi_stair_200.sh so we hit the same memory
# pressure as the production run, just much earlier.

# Reduce CUDA fragmentation (recommended by the OOM error message).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MOTION_DIR="src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/Unreal_Engine_stair_subset200_aligned"

python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-multi-terrain-future-motion-heightmap-videomimic \
    logger:wandb \
    --logger.entity draftrec \
    --logger.project MotionTracking \
    --logger.group=Unreal_Engine \
    --logger.name=g1_ue_multi_terrain_stair_200_future_motion_heightmap_xyz \
    --logger.video.enabled False \
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
    --command.setup_terms.motion_command.params.motion_config.max_motions 200 \
    --command.setup_terms.motion_command.params.motion_config.noise_to_initial_pose.init_root_offset="[0.0, 0.0, 0.1]" \
    --training.num_envs 4096 \
    --simulator.config.scene.env_spacing=0.0 \
    --algo.config.save_interval 1000 \
    --algo.config.eval_interval 5
