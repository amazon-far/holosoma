"""Multi-motion experiment configs for G1 robot.

Two base variants:
    * `g1_29dof_multi_motion` — many motions on a flat plane (was `g1_29dof_phuma`).
    * `g1_29dof_multi_terrain` — many motions, each paired to its own scene mesh tile
      (for VideoMimic-style training).

Each variant has a `_future_motion` sibling that swaps in the motion encoder
actor/critic architecture.
"""

from dataclasses import replace

from holosoma.config_types.algo import (
    DAggerAlgoConfig,
    DAggerConfig,
    DAggerModuleDictConfig,
    HeightMapEncoderConfig,
    LayerConfig,
    ModuleConfig,
    MotionEncoderConfig,
    OptimizerConfig,
)
from holosoma.config_types.command import CommandManagerCfg, CommandTermCfg, MultiMotionConfig, NoiseToInitialPoseConfig
from holosoma.config_types.experiment import ExperimentConfig, TrainingConfig
from holosoma.config_values import (
    action,
    algo,
    curriculum,
    observation,
    randomization,
    reward,
    robot,
    simulator,
    termination,
    terrain,
)
from pydantic.dataclasses import dataclass as pydantic_dataclass


# ── Shared hyperparameters ──
NUM_ENVS = 8192
POOL_SIZE = NUM_ENVS
RESAMPLE_INTERVAL = 5000           # resample pool from disk every N env steps (~208 iterations)
MIN_MOTION_LENGTH = 30             # skip motions shorter than this (frames)
NUM_LEARNING_ITERATIONS = 500000
SAVE_INTERVAL = 1000
EVAL_INTERVAL = SAVE_INTERVAL

_init_pose_config = NoiseToInitialPoseConfig(
    overall_noise_scale=1.0,
    dof_pos=0.1,
    root_pos=[0.05, 0.05, 0.01],
    root_rot=[0.1, 0.1, 0.2],
    root_lin_vel=[0.1, 0.1, 0.05],
    root_ang_vel=[0.1, 0.1, 0.1],
    object_pos=[0.0, 0.0, 0.0],
)

_BODY_NAMES_TO_TRACK = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]


def _build_motion_config(**overrides) -> MultiMotionConfig:
    base = dict(
        motion_dir="/path/to/g1_npz/",
        split_file="",
        pool_size=POOL_SIZE,
        resample_interval=RESAMPLE_INTERVAL,
        min_motion_length=MIN_MOTION_LENGTH,
        body_names_to_track=_BODY_NAMES_TO_TRACK,
        body_name_ref=["torso_link"],
        start_at_timestep_zero_prob=0.2,
        freeze_at_timestep_zero_prob=0.95,
        enable_default_pose_prepend=False,
        enable_default_pose_append=False,
        noise_to_initial_pose=_init_pose_config,
    )
    base.update(overrides)
    return MultiMotionConfig(**base)


def _build_command_cfg(motion_config: MultiMotionConfig) -> CommandManagerCfg:
    return CommandManagerCfg(
        params={},
        setup_terms={
            "motion_command": CommandTermCfg(
                func="holosoma.managers.command.terms.wbt:MultiMotionCommand",
                params={"motion_config": motion_config},
            ),
        },
        reset_terms={
            "motion_command": CommandTermCfg(
                func="holosoma.managers.command.terms.wbt:MultiMotionCommand",
            )
        },
        step_terms={
            "motion_command": CommandTermCfg(
                func="holosoma.managers.command.terms.wbt:MultiMotionCommand",
            )
        },
    )


@pydantic_dataclass(frozen=True)
class _SuccessRateCbConfig:
    _target_: str = "holosoma.agents.callbacks.success_rate_callback.SuccessRateCallback"


_eval_callbacks = {"success_rate": _SuccessRateCbConfig()}


def _future_motion_module_dict(ppo_cfg):
    """Swap the actor/critic for MLPWithMotionEncoder and attach the motion encoder."""
    return replace(
        ppo_cfg.module_dict,
        actor=replace(
            ppo_cfg.module_dict.actor,
            type="MLPWithMotionEncoder",
            input_dim=["actor_obs"],
            layer_config=replace(
                ppo_cfg.module_dict.actor.layer_config,
                hidden_dims=[768, 512, 256],
                activation="SiLU",
            ),
        ),
        critic=replace(
            ppo_cfg.module_dict.critic,
            type="MLPWithMotionEncoder",
            input_dim=["critic_obs"],
            layer_config=replace(
                ppo_cfg.module_dict.critic.layer_config,
                hidden_dims=[768, 512, 256],
                activation="SiLU",
            ),
        ),
        motion_encoder=MotionEncoderConfig(
            input_dim=0,  # Auto-calculated
            hidden_dim=60,
            output_dim=128,
            num_timesteps=20,
            activation="SiLU",
        ),
    )


def _base_algo():
    return replace(
        algo.ppo,
        config=replace(
            algo.ppo.config,
            num_learning_iterations=NUM_LEARNING_ITERATIONS,
            save_interval=SAVE_INTERVAL,
            entropy_coef=0.005,
            init_noise_std=1.0,
            init_at_random_ep_len=False,
            use_symmetry=False,
            eval_interval=EVAL_INTERVAL,
            eval_callbacks=_eval_callbacks,
            actor_optimizer=replace(algo.ppo.config.actor_optimizer, weight_decay=0.000),
            critic_optimizer=replace(algo.ppo.config.critic_optimizer, weight_decay=0.000),
        ),
    )


def _future_motion_algo():
    cfg = _base_algo()
    return replace(cfg, config=replace(cfg.config, module_dict=_future_motion_module_dict(cfg.config)))


def _future_motion_heightmap_videomimic_module_dict(ppo_cfg):
    """Future-motion module_dict + VideoMimic-style heightmap encoder."""
    base = _future_motion_module_dict(ppo_cfg)
    return replace(
        base,
        actor=replace(base.actor, type="MLPWithHeightMapVideoMimic"),
        critic=replace(base.critic, type="MLPWithHeightMapVideoMimic"),
        height_map_encoder=HeightMapEncoderConfig(
            latent_dim=415,
            num_heads=16,
            map_height=11,
            map_width=11,
            num_channels=1,
            conv_hidden_channels=16,
        ),
    )


def _future_motion_heightmap_videomimic_algo():
    cfg = _base_algo()
    return replace(
        cfg,
        config=replace(cfg.config, module_dict=_future_motion_heightmap_videomimic_module_dict(cfg.config)),
    )


def _future_motion_heightmap_videomimic_add_module_dict(ppo_cfg):
    """Future-motion module_dict + VideoMimic-style heightmap encoder
    that *adds* the heightmap latent to the activation right after the actor's
    first Linear layer (mirrors VideoMimic's
    ``flatten_then_embed_with_attention_to_hidden``).

    Designed for finetuning a non-heightmap pretrained checkpoint: the MLP
    input dim is unchanged, so all pretrained MLP weights load shape-compatibly,
    and the new heightmap branch is gated to zero at init via the encoder's
    learnable attention parameter.
    """
    base = _future_motion_module_dict(ppo_cfg)
    first_hidden_dim = base.actor.layer_config.hidden_dims[0]
    return replace(
        base,
        actor=replace(base.actor, type="MLPWithHeightMapVideoMimic"),
        critic=replace(base.critic, type="MLPWithHeightMapVideoMimic"),
        height_map_encoder=HeightMapEncoderConfig(
            latent_dim=first_hidden_dim,
            num_heads=16,
            map_height=11,
            map_width=17,
            num_channels=3,
            conv_hidden_channels=16,
            combine_mode="add_to_hidden",
        ),
    )


def _future_motion_heightmap_videomimic_add_algo():
    cfg = _base_algo()
    return replace(
        cfg,
        config=replace(cfg.config, module_dict=_future_motion_heightmap_videomimic_add_module_dict(cfg.config)),
    )


# ──────────────────────────────────────────────────────────────────────────────
# g1_29dof_multi_motion — many motions on a plane (ex-PhUMA)
# ──────────────────────────────────────────────────────────────────────────────
_multi_motion_motion_cfg = _build_motion_config()

g1_29dof_multi_motion = ExperimentConfig(
    training=TrainingConfig(
        project="MultiMotion",
        name="g1_29dof_multi_motion",
        num_envs=NUM_ENVS,
    ),
    env_class="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
    algo=_base_algo(),
    simulator=replace(
        simulator.isaacsim,
        config=replace(
            simulator.isaacsim.config,
            sim=replace(
                simulator.isaacsim.config.sim,
                max_episode_length_s=10.0,
            ),
        ),
    ),
    robot=replace(
        robot.g1_29dof,
        control=replace(robot.g1_29dof.control, action_scale=1.0),
        asset=replace(robot.g1_29dof.asset, enable_self_collisions=True),
        init_state=replace(robot.g1_29dof.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    terrain=terrain.terrain_locomotion_plane,
    observation=observation.g1_29dof_wbt_observation,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_wbt_termination,
    randomization=randomization.g1_29dof_wbt_randomization,
    command=_build_command_cfg(_multi_motion_motion_cfg),
    curriculum=curriculum.g1_29dof_wbt_curriculum,
    reward=reward.g1_29dof_wbt_reward,
)

g1_29dof_multi_motion_future_motion = replace(
    g1_29dof_multi_motion,
    training=replace(g1_29dof_multi_motion.training, name="g1_29dof_multi_motion_future_motion"),
    algo=_future_motion_algo(),
    observation=observation.g1_29dof_wbt_observation_future_motion,
)


# ──────────────────────────────────────────────────────────────────────────────
# g1_29dof_multi_terrain — many motions, each paired with its own scene mesh tile
# ──────────────────────────────────────────────────────────────────────────────
# Defaults bundle the multi-mesh terrain wiring so the training script only has
# to supply the motion_dir, pool_size, and optional caps.
_multi_terrain_motion_cfg = _build_motion_config(
    pool_ordered=True,
    bind_terrain_tiles=True,
    resample_interval=0,
)

g1_29dof_multi_terrain = replace(
    g1_29dof_multi_motion,
    training=replace(g1_29dof_multi_motion.training, name="g1_29dof_multi_terrain"),
    terrain=terrain.terrain_load_obj,
    command=_build_command_cfg(_multi_terrain_motion_cfg),
)

g1_29dof_multi_terrain_future_motion = replace(
    g1_29dof_multi_terrain,
    training=replace(g1_29dof_multi_terrain.training, name="g1_29dof_multi_terrain_future_motion"),
    algo=_future_motion_algo(),
    observation=observation.g1_29dof_wbt_observation_future_motion,
)

g1_29dof_multi_terrain_future_motion_heightmap_videomimic = replace(
    g1_29dof_multi_terrain_future_motion,
    training=replace(
        g1_29dof_multi_terrain_future_motion.training,
        name="g1_29dof_multi_terrain_future_motion_heightmap_videomimic",
    ),
    algo=_future_motion_heightmap_videomimic_algo(),
    observation=observation.g1_29dof_wbt_observation_future_motion_heightmap_videomimic,
)

g1_29dof_multi_terrain_future_motion_heightmap_videomimic_add = replace(
    g1_29dof_multi_terrain_future_motion,
    training=replace(
        g1_29dof_multi_terrain_future_motion.training,
        name="g1_29dof_multi_terrain_future_motion_heightmap_videomimic_add",
    ),
    algo=_future_motion_heightmap_videomimic_add_algo(),
    observation=observation.g1_29dof_wbt_observation_future_motion_heightmap_videomimic,
)


# ──────────────────────────────────────────────────────────────────────────────
# DAgger student experiment — distill the videomimic teacher to a smaller-obs
# student that does not see future motion targets.
# ──────────────────────────────────────────────────────────────────────────────
def _dagger_student_algo() -> DAggerAlgoConfig:
    """Pure DAgger student. Heightmap + small actor MLP, no future motion."""

    actor = ModuleConfig(
        type="MLPWithHeightMapVideoMimic",
        input_dim=["actor_obs"],
        output_dim=["robot_action_dim"],
        # Match VideoMimic's `G1DeepMimicCfgRootHeightfieldNoHistoryDagger`
        # actor: [1024, 512, 256, 128] with ELU.
        layer_config=LayerConfig(hidden_dims=[1024, 512, 256, 128], activation="ELU"),
    )

    height_map_encoder = HeightMapEncoderConfig(
        # latent_dim must match actor's first hidden dim under add_to_hidden.
        latent_dim=1024,
        num_heads=16,
        map_height=11,
        map_width=17,
        num_channels=3,
        conv_hidden_channels=16,
        combine_mode="add_to_hidden",
    )

    return DAggerAlgoConfig(
        _target_="holosoma.agents.dagger.dagger.DAgger",
        _recursive_=False,
        config=DAggerConfig(
            module_dict=DAggerModuleDictConfig(
                actor=actor,
                height_map_encoder=height_map_encoder,
            ),
            policy_to_clone="",  # supplied via CLI
            actor_learning_rate=1e-3,
            actor_optimizer=OptimizerConfig(_target_="torch.optim.AdamW", weight_decay=0.0),
            max_grad_norm=1.0,
            num_steps_per_env=24,
            num_learning_epochs=5,
            num_mini_batches=4,
            init_noise_std=0.8,
            save_interval=SAVE_INTERVAL,
            num_learning_iterations=NUM_LEARNING_ITERATIONS,
            bc_sigma_loss_coef=1.0,
            clip_teacher_actions=True,
            clip_actions_threshold=8.0,
            take_teacher_actions=False,
        ),
    )


g1_29dof_multi_terrain_videomimic_student_dagger = replace(
    g1_29dof_multi_terrain_future_motion,
    training=replace(
        g1_29dof_multi_terrain_future_motion.training,
        name="g1_29dof_multi_terrain_videomimic_student_dagger",
    ),
    algo=_dagger_student_algo(),
    observation=observation.g1_29dof_wbt_observation_videomimic_student,
)


# Backward-compat aliases for older train scripts that still refer to the
# legacy PhUMA names.
g1_29dof_phuma = g1_29dof_multi_motion
g1_29dof_phuma_future_motion = g1_29dof_multi_motion_future_motion


__all__ = [
    "g1_29dof_multi_motion",
    "g1_29dof_multi_motion_future_motion",
    "g1_29dof_multi_terrain",
    "g1_29dof_multi_terrain_future_motion",
    "g1_29dof_multi_terrain_future_motion_heightmap_videomimic",
    "g1_29dof_multi_terrain_future_motion_heightmap_videomimic_add",
    "g1_29dof_multi_terrain_videomimic_student_dagger",
    "g1_29dof_phuma",
    "g1_29dof_phuma_future_motion",
]
