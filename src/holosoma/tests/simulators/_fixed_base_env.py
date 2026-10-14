"""Shared ``ExperimentConfig`` for the fixed-base arm env tests.

Used by the CPU e2e test (``holosoma/envs/tests/test_fixed_base_arm_e2e_mujoco.py``) and the cross-backend
harness (``fixed_base_env_assert.py``) so both run the same env: the ``test_arm_fixed`` preset (3-DOF arm,
``asset.fix_base_link=True``) through ``LeggedRobotLocomotionManager`` with robot-agnostic terms only (dof_pos /
dof_vel / actions observations, action-rate + DOF-limit penalties, timeout termination, the g1 gait/velocity
command terms, the g1 joint-position action term) on the plane terrain, and PPO for two iterations.

The default plane terrain is used as shipped: its spawn XY jitter applies to floating bases only, so a reset
writes terrain origin + init pose. That is the spawn pose on MuJoCo/IsaacGym (a same-pose no-op under the
simulator's fixed-base rule); on IsaacSim, which spawns on IsaacLab's env grid, the first reset is a teleport.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from holosoma.config_types.curriculum import CurriculumManagerCfg, CurriculumTermCfg
from holosoma.config_types.experiment import ExperimentConfig, TrainingConfig
from holosoma.config_types.logger import DisabledLoggerConfig
from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg
from holosoma.config_types.randomization import RandomizationManagerCfg, RandomizationTermCfg
from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg
from holosoma.config_types.termination import TerminationManagerCfg, TerminationTermCfg
from holosoma.config_values import algo, command, simulator, terrain
from tests.simulators._robot_presets import test_arm_action, test_arm_fixed

LEARNING_ITERATIONS = 2

_OBS_TERMS = {
    "dof_pos": ObsTermCfg(func="holosoma.managers.observation.terms.locomotion:dof_pos"),
    "dof_vel": ObsTermCfg(func="holosoma.managers.observation.terms.locomotion:dof_vel", scale=0.05),
    "actions": ObsTermCfg(func="holosoma.managers.observation.terms.locomotion:actions"),
}
OBS_DIM = 3 * len(test_arm_fixed.dof_names)

# --simulator choice -> simulator preset (the CLI subcommand names).
SIMULATORS = {
    "mujoco": simulator.mujoco,
    "mjwarp": simulator.mjwarp,
    "isaacgym": simulator.isaacgym,
    "isaacsim": simulator.isaacsim,
}


def fixed_base_arm_experiment(backend: str, num_envs: int, log_root: Path) -> ExperimentConfig:
    """The fixed-base arm experiment for ``backend`` (a ``SIMULATORS`` key) with ``num_envs`` envs."""
    return ExperimentConfig(
        env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
        training=TrainingConfig(num_envs=num_envs, headless=True, seed=0, project="fixed_base_arm", name=backend),
        algo=dataclasses.replace(
            algo.ppo,
            config=dataclasses.replace(
                algo.ppo.config, use_symmetry=False, num_learning_iterations=LEARNING_ITERATIONS
            ),
        ),
        simulator=SIMULATORS[backend],
        terrain=terrain.terrain_locomotion_plane,
        robot=test_arm_fixed,
        observation=ObservationManagerCfg(
            groups={
                "actor_obs": ObsGroupCfg(terms=dict(_OBS_TERMS)),
                "critic_obs": ObsGroupCfg(terms=dict(_OBS_TERMS)),
            }
        ),
        action=test_arm_action,
        reward=RewardManagerCfg(
            terms={
                "penalty_action_rate": RewardTermCfg(
                    func="holosoma.managers.reward.terms.locomotion:penalty_action_rate", weight=-0.01
                ),
                "limits_dof_pos": RewardTermCfg(
                    func="holosoma.managers.reward.terms.locomotion:limits_dof_pos", weight=-1.0
                ),
            }
        ),
        termination=TerminationManagerCfg(
            terms={
                "timeout": TerminationTermCfg(
                    func="holosoma.managers.termination.terms.common:timeout_exceeded", is_timeout=True
                )
            }
        ),
        randomization=RandomizationManagerCfg(
            reset_terms={
                "randomize_pd_gains": RandomizationTermCfg(
                    func="holosoma.managers.randomization.terms.locomotion:randomize_pd_gains",
                    params={"kp_range": [0.9, 1.1], "kd_range": [0.9, 1.1]},
                ),
            },
        ),
        command=command.g1_29dof_command,
        curriculum=CurriculumManagerCfg(
            setup_terms={
                "average_episode_tracker": CurriculumTermCfg(
                    func="holosoma.managers.curriculum.terms.locomotion:AverageEpisodeLengthTracker"
                ),
            }
        ),
        logger=DisabledLoggerConfig(base_dir=str(log_root)),
    )
