"""Curriculum manager config for the loco-arm-tracking environment (G1 robot).

Unlike the locomotion environment, loco-arm-tracking does NOT use the adaptive
PenaltyCurriculum (which scales penalty weights based on episode length).
Penalties are applied at full weight from the start; difficulty progression is
handled entirely by the multi-phase pipeline curriculum (tracking sigma,
velocity ranges, workspace bounds).
"""

from holosoma.config_types.curriculum import CurriculumManagerCfg, CurriculumTermCfg


g1_loco_arm_tracking_curriculum = CurriculumManagerCfg(
    params={
        "num_compute_average_epl": 1000,
    },
    setup_terms={
        "average_episode_tracker": CurriculumTermCfg(
            func="holosoma.managers.curriculum.terms.locomotion:AverageEpisodeLengthTracker",
            params={},
        ),
    },
    reset_terms={},
    step_terms={},
)

