source scripts/source_isaacsim_setup.sh

MOTION_DIR="src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/videomimic_captures_29dof"

python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-multi-terrain-future-motion \
    logger:wandb \
    --logger.entity draftrec \
    --logger.project MotionTracking \
    --logger.group=statue_wbt \
    --logger.name=g1_videomimic_multi_terrain_future_motion \
    --terrain.terrain-term.obj-dir="$MOTION_DIR" \
    --terrain.terrain-term.num-rows 1 \
    --terrain.terrain-term.obj-max-faces-per-tile 3000 \
    --terrain.terrain-term.env-origin-in-tile="[0.0, 0.0, 0.0]" \
    --command.setup_terms.motion_command.params.motion_config.motion_dir="$MOTION_DIR" \
    --command.setup_terms.motion_command.params.motion_config.max_motions 130 \
    --training.num_envs 4096 \
    --simulator.config.scene.env_spacing=0.0 \
    --algo.config.save_interval 1000 \
    --algo.config.eval_interval 0
