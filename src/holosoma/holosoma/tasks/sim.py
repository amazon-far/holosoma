"""The ``sim`` task: run a ``SimulationTask`` until the simulator requests shutdown."""

from __future__ import annotations

import os
import sys
import traceback

from loguru import logger

from holosoma.config_types.experiment import HolosomaConfig
from holosoma.config_values import experiment as experiment_values
from holosoma.tasks.base import Task
from holosoma.utils.eval_utils import init_eval_logging

_THREAD_LIMIT_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def limit_cpu_math_threads(config: HolosomaConfig, device: str | None = None) -> None:
    """Serialize Torch's CPU intra-op pool for classic MuJoCo on cpu, unless the operator set a limit.

    Classic MuJoCo runs one env on the host, where every tensor is small enough that a machine-sized
    pool costs more per elementwise op in fork/join than the op itself. A no-op for Warp and
    IsaacSim, which keep this work on the GPU, and for the other tasks, which do not route through
    here. ``device`` is the resolved simulation device; it defaults to ``config.run.device``.
    """
    from holosoma.config_types.simulator import MujocoBackend
    from holosoma.utils.safe_torch_import import torch

    # Both halves are needed: ``mujoco_backend`` lives on the shared SimulatorInitConfig and
    # defaults to CLASSIC, so every backend's preset reports CLASSIC unless it overrides it. The
    # name is the same discriminator ``resolve_simulation_device`` uses.
    sim_config = config.simulator.config
    is_classic = sim_config.name == "mujoco" and getattr(sim_config, "mujoco_backend", None) == MujocoBackend.CLASSIC
    if device is None:
        device = config.run.device
    if not is_classic or device is None or not device.startswith("cpu"):
        return
    # Truthiness, not presence: OpenMP and MKL ignore an empty value, so an empty one is not an
    # operator's choice and must not suppress the cap.
    preset = [name for name in _THREAD_LIMIT_VARIABLES if os.environ.get(name)]
    if preset:
        logger.info(f"{', '.join(preset)} already set - leaving Torch at {torch.get_num_threads()} CPU thread(s)")
        return
    torch.set_num_threads(1)
    logger.info("Classic MuJoCo on cpu - limiting Torch to 1 CPU thread")


def run_simulation(config: HolosomaConfig) -> None:
    """Build the configured simulation task and run it until shutdown."""
    from holosoma.config_types.env import get_tyro_env_config
    from holosoma.simulator.session import get_local_rank, resolve_simulation_device

    # Resolved as the task session will, so the CPU thread cap applies before the task is built.
    limit_cpu_math_threads(config, resolve_simulation_device(get_tyro_env_config(config), get_local_rank()))

    try:
        with config.create_task() as task:
            task.run()
    except Exception as e:
        logger.error(f"Error during simulation: {e}")
        traceback.print_exc()
        sys.exit(1)


class SimTask(Task):
    """Run the simulation loop with no policy; an external controller drives the robot."""

    name = "sim"

    @classmethod
    def from_args(cls, args: list[str]) -> SimTask:
        from holosoma.utils.config_registry import parse_config

        init_eval_logging()
        config = parse_config(
            HolosomaConfig,
            default=experiment_values.sim,
            args=cls.with_selection(args),
            description="Run simulation with direct simulator control and bridge support.\n\n"
            "Usage: holosoma simulator:<sim> robot:<robot> terrain:<terrain>\n"
            "Examples:\n"
            "  holosoma # defaults \n"
            "  holosoma simulator:sim-mujoco robot:t1_29dof_waist_wrist "
            "terrain:terrain_locomotion_plane\n"
            "  holosoma simulator:sim-isaacsim robot:g1_29dof terrain:terrain_locomotion_mix\n"
            "  holosoma --config run.yaml [--config-path a.b] [overrides...]\n"
            "  holosoma task:train exp:g1_29dof simulator:isaacsim  # also task:eval, task:replay",
        )
        return cls(config)

    def run(self) -> None:
        run_simulation(self.config)
