"""Checkpoint, multi-GPU sync and hook wiring for curriculum term state (no sim).

Uses the real CurriculumManager, curriculum terms and BaseAlgo checkpoint methods on a minimal task, so no
simulator is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.config_types.curriculum import CurriculumManagerCfg, CurriculumTermCfg
from holosoma.managers.curriculum.manager import CurriculumManager
from holosoma.managers.curriculum.terms.locomotion import AverageEpisodeLengthTracker, PenaltyCurriculum
from holosoma.simulator.base_simulator.hooks import HookRegistry, Phase

pytestmark = pytest.mark.no_sim

INITIAL_SCALE = 0.1
LEVEL_UP_THRESHOLD = 750.0
DEGREE = 0.00025
PENALTY_WEIGHTS = {"penalty_action_rate": -3.0, "penalty_feet_ori": -1.0}
UNTAGGED_WEIGHTS = {"tracking_lin_vel": 1.0}


@dataclass(frozen=True)
class _TermCfg:
    weight: float
    tags: tuple[str, ...] = ()


class _RewardManager:
    def __init__(self) -> None:
        self._terms = {name: _TermCfg(w, ("penalty_curriculum",)) for name, w in PENALTY_WEIGHTS.items()}
        self._terms.update({name: _TermCfg(w) for name, w in UNTAGGED_WEIGHTS.items()})

    @property
    def active_terms(self) -> list[str]:
        return list(self._terms)

    def get_term_cfg(self, name: str) -> _TermCfg:
        return self._terms[name]

    def set_term_cfg(self, name: str, cfg: _TermCfg) -> None:
        self._terms[name] = cfg


def _curriculum_cfg(
    *, penalty_key: str | None = "penalty_curriculum", enabled: bool = True, num_compute_average_epl: int = 1000
) -> CurriculumManagerCfg:
    terms = {
        "average_episode_tracker": CurriculumTermCfg(
            func="holosoma.managers.curriculum.terms.locomotion:AverageEpisodeLengthTracker",
            params={"num_compute_average_epl": num_compute_average_epl},
        )
    }
    if penalty_key is not None:
        terms[penalty_key] = CurriculumTermCfg(
            func="holosoma.managers.curriculum.terms.locomotion:PenaltyCurriculum",
            params={
                "enabled": enabled,
                "tag": "penalty_curriculum",
                "initial_scale": INITIAL_SCALE,
                "min_scale": 0.0,
                "max_scale": 1.0,
                "level_down_threshold": 150.0,
                "level_up_threshold": LEVEL_UP_THRESHOLD,
                "degree": DEGREE,
            },
        )
    return CurriculumManagerCfg(setup_terms=terms)


class _Task:
    """Just enough of a policy task for a curriculum: reward weights, pending episode lengths and hooks."""

    def __init__(self) -> None:
        self.num_envs = 4
        self.device = "cpu"
        self.reward_manager = _RewardManager()
        self._pending_episode_lengths = torch.zeros(self.num_envs, dtype=torch.long)
        self._pending_episode_update_mask = torch.zeros(self.num_envs, dtype=torch.bool)
        self.hooks = HookRegistry()
        self.metric_sources: list[Any] = []

    def add_metric_source(self, source: Any) -> None:
        self.metric_sources.append(source)

    def get_checkpoint_state(self) -> dict[str, Any]:
        return {"task": 1}

    def load_checkpoint_state(self, state: dict[str, Any] | None) -> None:
        self.loaded = state


class _Curriculum:
    def __init__(self, *, install: bool = False, **cfg_kwargs: Any) -> None:
        self.task = _Task()
        self.manager = CurriculumManager(_curriculum_cfg(**cfg_kwargs), self.task, "cpu")
        if install:
            self.manager.install()
        else:
            self.manager.setup()

    @property
    def penalty(self) -> PenaltyCurriculum:
        return next(t for _, t in self.manager.iter_terms() if isinstance(t, PenaltyCurriculum))

    @property
    def tracker(self) -> AverageEpisodeLengthTracker:
        return cast("AverageEpisodeLengthTracker", self.manager.get_term("average_episode_tracker"))

    def weight(self, name: str) -> float:
        return self.task.reward_manager.get_term_cfg(name).weight

    def algo(self) -> BaseAlgo:
        """BaseAlgo's checkpoint methods, bound to this curriculum and task."""
        return cast("BaseAlgo", SimpleNamespace(curriculum=self.manager, _unwrap_env=lambda: self.task))


def _trained(scale: float = 0.8, avg_length: float = 900.0) -> _Curriculum:
    curriculum = _Curriculum()
    curriculum.tracker.set_average(avg_length, suppress_update=False)
    curriculum.penalty.load_state_dict({"current_scale": scale})
    return curriculum


def _assert_scale_applied(curriculum: _Curriculum, scale: float) -> None:
    assert curriculum.penalty.current_scale == pytest.approx(scale)
    for name, weight in PENALTY_WEIGHTS.items():
        assert curriculum.weight(name) == pytest.approx(weight * scale)
    for name, weight in UNTAGGED_WEIGHTS.items():
        assert curriculum.weight(name) == pytest.approx(weight)


def test_checkpoint_round_trip_keeps_the_env_state_layout() -> None:
    trained = _trained()
    state = BaseAlgo._collect_env_state(trained.algo())
    assert state["task"] == 1
    assert set(state["curriculum"]) == {"average_episode_tracker", "penalty_curriculum"}

    resumed = _Curriculum()
    _assert_scale_applied(resumed, INITIAL_SCALE)
    BaseAlgo._restore_env_state(resumed.algo(), state)
    _assert_scale_applied(resumed, 0.8)
    assert float(resumed.tracker.get_average()) == pytest.approx(900.0)
    assert resumed.task.loaded == state

    # The first reset after a restore steps the curriculum from the restored scale, not from initial_scale.
    resumed.penalty.reset(None)
    _assert_scale_applied(resumed, 0.8 * (1 + DEGREE))


def test_penalty_term_found_under_any_key() -> None:
    state = _trained().manager.state_dict()
    resumed = _Curriculum(penalty_key="penalties")
    resumed.manager.load_checkpoint_state({"curriculum": {"penalties": state["penalty_curriculum"]}})
    _assert_scale_applied(resumed, 0.8)


@pytest.mark.parametrize("scale", [0.873, torch.tensor(0.873)])
def test_legacy_checkpoint_layout_is_converted(scale: float | torch.Tensor) -> None:
    legacy = {
        "average_episode_tracker": {"average_episode_length": torch.tensor(900.0), "suppress_next_update": False},
        "reward_penalty_scale": scale,
    }
    resumed = _Curriculum(penalty_key="penalties")
    with pytest.warns(FutureWarning, match="old env-state layout"):
        resumed.manager.load_checkpoint_state(legacy)
    _assert_scale_applied(resumed, 0.873)
    assert float(resumed.tracker.get_average()) == pytest.approx(900.0)
    assert resumed.tracker._suppress_next_update


def test_unknown_term_is_ignored_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    from loguru import logger

    handler = logger.add(caplog.handler, level="WARNING", format="{message}")
    try:
        resumed = _Curriculum()
        resumed.manager.load_checkpoint_state({"curriculum": {"removed_term": {"x": 1}}})
    finally:
        logger.remove(handler)
    assert "removed_term" in caplog.text
    _assert_scale_applied(resumed, INITIAL_SCALE)


def test_disabled_penalty_curriculum_saves_nothing() -> None:
    disabled = _Curriculum(enabled=False)
    assert disabled.penalty.state_dict() == {}
    assert "penalty_curriculum" not in disabled.manager.state_dict()


@pytest.mark.parametrize("penalty_key", ["penalty_curriculum", None], ids=["with-penalty", "tracker-only"])
def test_first_reset_after_restore_keeps_tracker_average(penalty_key: str | None) -> None:
    saved = _Curriculum(penalty_key=penalty_key, num_compute_average_epl=1)
    saved.tracker.set_average(300.0, suppress_update=False)
    state = saved.manager.state_dict()

    resumed = _Curriculum(penalty_key=penalty_key, num_compute_average_epl=1)
    # As after a full reset: every env flagged for an update with zero-length episodes.
    resumed.task._pending_episode_update_mask[:] = True
    resumed.manager.load_checkpoint_state({"curriculum": state})
    resumed.tracker.reset(torch.arange(resumed.task.num_envs))
    assert float(resumed.tracker.get_average()) == pytest.approx(300.0)


def test_installed_curriculum_updates_on_every_reset_call_and_steps_each_frame() -> None:
    curriculum = _Curriculum(install=True)
    assert curriculum.task.metric_sources == [curriculum.manager]

    # A frame with no resets still advances the penalty curriculum, as an explicit full reset does.
    curriculum.task.hooks.emit(Phase.RESET, torch.zeros(0, dtype=torch.long))
    curriculum.task.hooks.emit(Phase.RESET, torch.arange(curriculum.task.num_envs))
    _assert_scale_applied(curriculum, INITIAL_SCALE * (1 - DEGREE) ** 2)

    curriculum.penalty.current_scale = 2.0
    curriculum.task.hooks.emit(Phase.FRAME_BEGIN)
    _assert_scale_applied(curriculum, 1.0)


def _sync_worker(rank: int, init_file: str, results: Any) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2, timeout=timedelta(seconds=60)
    )
    try:
        trained = _trained(scale=0.8 if rank == 0 else 0.3, avg_length=900.0 if rank == 0 else 200.0)
        trained.manager.synchronize(device="cpu", world_size=2)
        results[rank] = (
            trained.penalty.current_scale,
            trained.weight("penalty_action_rate"),
            float(trained.tracker.get_average()),
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="needs torch.distributed gloo")
def test_multi_gpu_sync_matches_rank_zero(tmp_path: Path) -> None:
    with mp.Manager() as manager:
        results = manager.dict()
        mp.start_processes(_sync_worker, args=(str(tmp_path / "pg"), results), nprocs=2, start_method="fork")
        got = dict(results)
    for rank in (0, 1):
        scale, weight, avg = got[rank]
        assert scale == pytest.approx(0.8)
        assert weight == pytest.approx(PENALTY_WEIGHTS["penalty_action_rate"] * 0.8)
        assert avg == pytest.approx(900.0)
