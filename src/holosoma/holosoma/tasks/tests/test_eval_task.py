from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from holosoma.config_types.experiment import HolosomaConfig
from holosoma.config_types.run import RunConfig
from holosoma.config_values import experiment
from holosoma.tasks import eval as eval_task
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim


@pytest.mark.parametrize("flag", ["--checkpoint", "--task.checkpoint"])
def test_eval_starts_from_the_checkpoint_config_with_overrides_on_top(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flag: str
) -> None:
    monkeypatch.setattr(eval_task, "init_eval_logging", lambda: None)
    saved = dataclasses.replace(HolosomaConfig.from_experiment(experiment.g1_29dof), run=RunConfig(seed=11))
    checkpoint = tmp_path / "model.pt"
    torch.save({"experiment_config": saved.to_serializable_dict()}, checkpoint)

    task = eval_task.EvalTask.from_args([f"{flag}={checkpoint}", "--run.num_envs", "3", "--push.config.enabled=True"])

    assert task.config.task.checkpoint == str(checkpoint)  # type: ignore[attr-defined]
    assert task.config.task.callbacks.push.config.enabled  # type: ignore[attr-defined]
    assert task.config.run.seed == 11
    assert task.config.run.num_envs == 3
