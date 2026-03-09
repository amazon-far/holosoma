source scripts/source_isaacsim_setup.sh
LOCAL_RANK=${LOCAL_RANK:-1} python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-phuma \
    logger:wandb \
    --logger.project holosoma \
    --logger.name=g1_29dof_phuma \
    --logger.video.enabled False \
    --command.setup_terms.motion_command.params.motion_config.motion_dir="/home/nas4_user/kyungminlee/work/holosoma/g1_npz" \
    --command.setup_terms.motion_command.params.motion_config.split_file="/home/nas4_user/kyungminlee/work/holosoma/split/phuma_train.txt"
