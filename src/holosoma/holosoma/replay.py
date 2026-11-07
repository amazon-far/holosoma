from __future__ import annotations

from typing import cast

from holosoma.config_types.experiment import ExperimentConfig
from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager


def replay(tyro_config: ExperimentConfig) -> None:
    with tyro_config.create_task() as task:
        env = cast("WholeBodyTrackingManager", task)
        done = False
        while not done:
            env.simulator.sim.step()  # type: ignore[attr-defined]
            done = env.step_visualize_motion(None)


def main() -> None:
    from holosoma.config_values.experiment import get_annotated_experiment_config
    from holosoma.utils.config_registry import parse_config

    # Pass the factory uncalled so parse_config builds it after plugins load.
    tyro_cfg = parse_config(get_annotated_experiment_config)
    replay(tyro_cfg)


if __name__ == "__main__":
    main()
