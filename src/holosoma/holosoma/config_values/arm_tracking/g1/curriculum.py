from holosoma.config_types.curriculum import CurriculumManagerCfg, CurriculumTermCfg


g1_arm_tracking_curriculum = CurriculumManagerCfg(
    params={
        "num_compute_average_epl": 1000,
    },
    setup_terms={
        "average_episode_tracker": CurriculumTermCfg(
            func="holosoma.managers.curriculum.terms.locomotion:AverageEpisodeLengthTracker",
            params={},
        )
    }
)

