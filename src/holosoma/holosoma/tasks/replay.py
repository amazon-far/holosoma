from __future__ import annotations

from typing import TYPE_CHECKING, cast

from holosoma.config_types.experiment import ExperimentConfig
from holosoma.tasks.base import Task

if TYPE_CHECKING:
    from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager


def replay(tyro_config: ExperimentConfig) -> None:
    with tyro_config.create_task() as task:
        env = cast("WholeBodyTrackingManager", task)
        done = False
        while not done:
            env.simulator.sim.step()  # type: ignore[attr-defined]
            done = env.step_visualize_motion(None)


class ReplayTask(Task):
    """Play a whole-body-tracking experiment's reference motion back in the simulator."""

    name = "replay"

    @classmethod
    def from_args(cls, args: list[str]) -> ReplayTask:
        from holosoma.config_values.experiment import get_annotated_experiment_config
        from holosoma.utils.config_registry import parse_config

        # Pass the factory uncalled so parse_config builds it after plugins load.
        return cls(parse_config(get_annotated_experiment_config, args=cls.with_selection(args)))

    def run(self) -> None:
        replay(self.config)
