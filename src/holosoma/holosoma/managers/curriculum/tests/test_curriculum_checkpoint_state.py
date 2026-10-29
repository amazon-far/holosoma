"""Checkpoint and multi-GPU sync tests for curriculum term state (no sim).

Uses the real CurriculumManager, curriculum terms and env checkpoint methods on a minimal env, so no simulator is
needed.
"""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from typing import Any, cast

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from holosoma.config_types.curriculum import CurriculumManagerCfg, CurriculumTermCfg
from holosoma.envs.locomotion.locomotion_manager import LeggedRobotLocomotionManager
from holosoma.managers.curriculum.manager import CurriculumManager
from holosoma.managers.curriculum.terms.locomotion import AverageEpisodeLengthTracker, PenaltyCurriculum

pytestmark = pytest.mark.no_sim

INITIAL_SCALE = 0.1
LEVEL_UP_THRESHOLD = 750.0
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


def _curriculum_cfg(*, penalty_key: str = "penalty_curriculum", enabled: bool = True) -> CurriculumManagerCfg:
    return CurriculumManagerCfg(
        setup_terms={
            "average_episode_tracker": CurriculumTermCfg(
                func="holosoma.managers.curriculum.terms.locomotion:AverageEpisodeLengthTracker"
            ),
            penalty_key: CurriculumTermCfg(
                func="holosoma.managers.curriculum.terms.locomotion:PenaltyCurriculum",
                params={
                    "enabled": enabled,
                    "tag": "penalty_curriculum",
                    "initial_scale": INITIAL_SCALE,
                    "min_scale": 0.0,
                    "max_scale": 1.0,
                    "level_down_threshold": 150.0,
                    "level_up_threshold": LEVEL_UP_THRESHOLD,
                    "degree": 0.00025,
                },
            ),
        }
    )


class _Env(LeggedRobotLocomotionManager):
    """Just enough of a locomotion env for its curriculum manager and checkpoint methods."""

    def __init__(self, **cfg_kwargs: Any) -> None:
        # Deliberately skips BaseTask.__init__, which needs a simulator.
        self.num_envs = 4
        self.device = "cpu"
        self.log_dict: dict[str, Any] = {}
        self.reward_manager = cast("Any", _RewardManager())
        self.curriculum_manager = CurriculumManager(_curriculum_cfg(**cfg_kwargs), self, self.device)
        self.curriculum_manager.setup()

    @property
    def penalty(self) -> PenaltyCurriculum:
        return next(t for _, t in self.curriculum_manager.iter_terms() if isinstance(t, PenaltyCurriculum))

    @property
    def tracker(self) -> AverageEpisodeLengthTracker:
        return self._get_average_episode_tracker()

    def weight(self, name: str) -> float:
        return self.reward_manager.get_term_cfg(name).weight


def _trained_env(scale: float = 0.8, avg_length: float = 900.0) -> _Env:
    env = _Env()
    env.tracker.set_average(avg_length, suppress_update=False)
    env.penalty.load_state_dict({"current_scale": scale})
    return env


def _assert_scale_applied(env: _Env, scale: float) -> None:
    assert env.penalty.current_scale == pytest.approx(scale)
    assert getattr(env, "reward_penalty_scale") == pytest.approx(scale)  # noqa: B009 - set by the term at runtime
    for name, weight in PENALTY_WEIGHTS.items():
        assert env.weight(name) == pytest.approx(weight * scale)
    for name, weight in UNTAGGED_WEIGHTS.items():
        assert env.weight(name) == pytest.approx(weight)


def test_checkpoint_round_trip_restores_all_terms() -> None:
    state = _trained_env().get_checkpoint_state()
    assert set(state["curriculum"]) == {"average_episode_tracker", "penalty_curriculum"}

    resumed = _Env()
    _assert_scale_applied(resumed, INITIAL_SCALE)
    resumed.load_checkpoint_state(state)
    _assert_scale_applied(resumed, 0.8)
    assert resumed.average_episode_length == pytest.approx(900.0)

    # The first reset after a restore steps the curriculum from the restored scale, not from initial_scale.
    resumed.penalty.reset(None)
    _assert_scale_applied(resumed, 0.8 * 1.00025)


def test_penalty_term_found_under_any_key() -> None:
    state = _trained_env().get_checkpoint_state()
    resumed = _Env(penalty_key="penalties")
    resumed.load_checkpoint_state({"curriculum": {"penalties": state["curriculum"]["penalty_curriculum"]}})
    _assert_scale_applied(resumed, 0.8)


@pytest.mark.parametrize("scale", [0.873, torch.tensor(0.873)])
def test_legacy_checkpoint_layout_is_converted(scale: float | torch.Tensor) -> None:
    legacy = {
        "average_episode_tracker": {"average_episode_length": torch.tensor(900.0), "suppress_next_update": False},
        "reward_penalty_scale": scale,
    }
    resumed = _Env(penalty_key="penalties")
    resumed.load_checkpoint_state(legacy)
    _assert_scale_applied(resumed, 0.873)
    assert resumed.average_episode_length == pytest.approx(900.0)
    assert resumed.tracker._suppress_next_update


def test_unknown_term_is_ignored_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    from loguru import logger

    handler = logger.add(caplog.handler, level="WARNING", format="{message}")
    try:
        resumed = _Env()
        resumed.load_checkpoint_state({"curriculum": {"removed_term": {"x": 1}}})
    finally:
        logger.remove(handler)
    assert "removed_term" in caplog.text
    _assert_scale_applied(resumed, INITIAL_SCALE)


def test_disabled_penalty_curriculum_saves_nothing() -> None:
    disabled = _Env(enabled=False)
    assert disabled.penalty.state_dict() == {}
    assert "penalty_curriculum" not in disabled.get_checkpoint_state()["curriculum"]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _sync_worker(rank: int, port: int, results: Any) -> None:
    os.environ.update({"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port)})
    dist.init_process_group("gloo", rank=rank, world_size=2)
    try:
        env = _trained_env(scale=0.8 if rank == 0 else 0.3, avg_length=900.0 if rank == 0 else 200.0)
        env.synchronize_curriculum_state(device="cpu", world_size=2)
        results[rank] = (env.penalty.current_scale, env.weight("penalty_action_rate"), env.average_episode_length)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="needs torch.distributed gloo")
def test_multi_gpu_sync_matches_rank_zero() -> None:
    with mp.Manager() as manager:
        results = manager.dict()
        mp.start_processes(_sync_worker, args=(_free_port(), results), nprocs=2, start_method="fork")
        got = dict(results)
    for rank in (0, 1):
        scale, weight, avg = got[rank]
        assert scale == pytest.approx(0.8)
        assert weight == pytest.approx(PENALTY_WEIGHTS["penalty_action_rate"] * 0.8)
        assert avg == pytest.approx(900.0)
