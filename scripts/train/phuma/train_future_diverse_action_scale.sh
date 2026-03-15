source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-phuma-future-motion \
    logger:wandb \
    --logger.entity draftrec \
    --logger.project holosoma \
    --logger.name=g1_29dof_phuma_future_diverse_action_scale \
    --logger.video.enabled False \
    --command.setup_terms.motion_command.params.motion_config.motion_dir="./g1_npz" \
    --command.setup_terms.motion_command.params.motion_config.split_file="./split/phuma_train.txt" \
    robot:g1-29dof-diverse-action-scale
