"""PolicyTask copies the metrics class terms publish into log_dict."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from holosoma.envs.policy_task import PolicyTask
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim


def _task(*managers: dict[str, Any]) -> SimpleNamespace:
    wrapped = [SimpleNamespace(iter_terms=terms.items) for terms in managers]
    return SimpleNamespace(log_dict={}, _metric_managers=lambda: wrapped)


def test_metrics_are_copied_from_every_manager_each_frame() -> None:
    clip = SimpleNamespace(metrics={"clip": torch.tensor(0.5)})
    task = _task(
        {"clip": clip}, {"silent": SimpleNamespace(), "motion": SimpleNamespace(metrics={"err": torch.tensor(1.0)})}
    )

    PolicyTask._collect_metrics(task)  # type: ignore[arg-type]
    clip.metrics["clip"] = torch.tensor(0.25)
    PolicyTask._collect_metrics(task)  # type: ignore[arg-type]

    assert task.log_dict == {"clip": torch.tensor(0.25), "err": torch.tensor(1.0)}


def test_a_key_published_by_two_terms_is_rejected() -> None:
    task = _task(
        {"a": SimpleNamespace(metrics={"k": torch.tensor(1.0)})},
        {"b": SimpleNamespace(metrics={"k": torch.tensor(2.0)})},
    )

    with pytest.raises(ValueError, match="'k' is published by both 'a' and 'b'"):
        PolicyTask._collect_metrics(task)  # type: ignore[arg-type]
