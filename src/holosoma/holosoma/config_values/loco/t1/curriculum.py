"""Locomotion curriculum presets for the T1 robot."""

from holosoma.config_types.curriculum import CurriculumManagerCfg, CurriculumTermCfg

t1_29dof_curriculum = CurriculumManagerCfg(
    setup_terms={
        "average_episode_tracker": CurriculumTermCfg(
            func="holosoma.managers.curriculum.terms.locomotion:AverageEpisodeLengthTracker",
            params={"base_num_envs": 4096},
        ),
        "penalty_curriculum": CurriculumTermCfg(
            func="holosoma.managers.curriculum.terms.locomotion:PenaltyCurriculum",
            params={
                "enabled": True,
                "tag": "penalty_curriculum",
                "initial_scale": 0.1,
                "min_scale": 0.0,
                "max_scale": 1.0,
                "level_down_threshold": 150.0,
                "level_up_threshold": 750.0,
                "degree": 0.00025,
            },
        ),
    },
    reset_terms={},
    step_terms={},
)

t1_29dof_curriculum_fast_sac = CurriculumManagerCfg(
    setup_terms={
        "average_episode_tracker": CurriculumTermCfg(
            func="holosoma.managers.curriculum.terms.locomotion:AverageEpisodeLengthTracker",
            params={"base_num_envs": 4096},
        ),
        "penalty_curriculum": CurriculumTermCfg(
            func="holosoma.managers.curriculum.terms.locomotion:PenaltyCurriculum",
            params={
                "enabled": True,
                "tag": "penalty_curriculum",
                "initial_scale": 0.5,
                "min_scale": 0.5,
                "max_scale": 1.0,
                "level_down_threshold": 150.0,
                "level_up_threshold": 750.0,
                "degree": 0.001,
            },
        ),
    },
    reset_terms={},
    step_terms={},
)

__all__ = ["t1_29dof_curriculum", "t1_29dof_curriculum_fast_sac"]
