"""Simulation-only frames avoid data-dependent reset selection when no terms exist."""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace
from typing import Any, cast

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from holosoma.config_types.robot import RobotConfig
from holosoma.config_types.termination import TerminationManagerCfg, TerminationTermCfg
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.envs.simulation_task import SimulationTask
from holosoma.managers.termination import TerminationManager


class _Operators(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.calls: Counter[str] = Counter()

    def __torch_dispatch__(
        self, func: Any, types: tuple[type, ...], args: tuple[Any, ...] = (), kwargs: dict[str, Any] | None = None
    ) -> Any:
        self.calls[str(func)] += 1
        return func(*args, **({} if kwargs is None else kwargs))


def _failures(env: BaseTask) -> torch.Tensor:
    return torch.tensor([False, True, False], device=env.device)


def _timeouts(env: BaseTask) -> torch.Tensor:
    return torch.tensor([False, False, True], device=env.device)


def _task(task_type: type[BaseTask], cfg: TerminationManagerCfg) -> BaseTask:
    task = object.__new__(task_type)
    task.num_envs = 3
    task.device = "cpu"
    task.robot_config = cast("RobotConfig", SimpleNamespace(init_state=SimpleNamespace(default_joint_angles={})))
    task.dof_names = []
    task._init_buffers()
    task.termination_manager = TerminationManager(cfg, task, task.device)
    return task


def test_without_terms_frames_clear_masks_without_reset_selection() -> None:
    task = _task(SimulationTask, TerminationManagerCfg())
    task.reset_buf.fill_(1)
    task.time_out_buf.fill_(True)
    with _Operators() as operations:
        task._check_termination()
        ids = task._get_reset_env_ids()

    # nonzero() is a host sync on CUDA; without terms the empty result is known.
    assert operations.calls["aten.nonzero.default"] == 0
    assert ids.dtype == torch.long and ids.numel() == 0
    assert task.reset_buf.tolist() == [0, 0, 0]
    assert task.time_out_buf.tolist() == [False, False, False]


def test_configured_terms_preserve_failure_timeout_masks_and_reset_selection() -> None:
    cfg = TerminationManagerCfg(
        terms={
            "failure": TerminationTermCfg(func=f"{__name__}:_failures"),
            "timeout": TerminationTermCfg(func=f"{__name__}:_timeouts", is_timeout=True),
        }
    )
    task = _task(SimulationTask, cfg)

    task._check_termination()
    ids = task._get_reset_env_ids()

    assert ids.tolist() == [1, 2]
    assert task.reset_buf.tolist() == [0, 1, 1]
    assert task.time_out_buf.tolist() == [False, False, True]
