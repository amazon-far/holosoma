"""Locomotion curriculum presets for the H1 robot."""

from holosoma.config_types.curriculum import CurriculumManagerCfg, CurriculumTermCfg

h1_19dof_curriculum = CurriculumManagerCfg(
    params={
        "num_compute_average_epl": 1000,
    },
    setup_terms={
        "average_episode_tracker": CurriculumTermCfg(
            func="holosoma.managers.curriculum.terms.locomotion:AverageEpisodeLengthTracker",
            params={},
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

h1_19dof_curriculum_fast_sac = CurriculumManagerCfg(
    params={
        "num_compute_average_epl": 1000,
    },
    setup_terms={
        "average_episode_tracker": CurriculumTermCfg(
            func="holosoma.managers.curriculum.terms.locomotion:AverageEpisodeLengthTracker",
            params={},
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

__all__ = ["h1_19dof_curriculum", "h1_19dof_curriculum_fast_sac"]

