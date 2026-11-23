"""Fixed-base behavior of the environment layer (no simulator).

A fixed robot resets to its mount pose without spawn jitter or velocity randomization.
Both push entry points reject enabled pushes for fixed robots. These tests exercise
the environment methods without a simulator backend.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from holosoma.config_types.randomization import RandomizationTermCfg
from holosoma.envs.locomotion.locomotion_manager import LeggedRobotLocomotionManager
from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager
from holosoma.managers.randomization.terms.locomotion import (
    PushRandomizerState,
    apply_pushes,
    randomize_push_schedule,
)
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.sampler import STAGE_RESET, TermSampler

pytestmark = pytest.mark.no_sim

_NUM_ENVS = 4
_ENV_IDS = torch.tensor([0, 2])
_OTHER_IDS = torch.tensor([1, 3])
_MANAGERS = [
    pytest.param(LeggedRobotLocomotionManager, 2, id="locomotion"),
    pytest.param(WholeBodyTrackingManager, 6, id="wbt"),
]


def _root_states() -> torch.Tensor:
    """Distinct value per cell so any write is visible."""
    return torch.arange(_NUM_ENVS * 13, dtype=torch.float).view(_NUM_ENVS, 13)


def _env_stub(*, floating_base: bool, push_dim: int) -> Any:
    """The attributes ``_reset_root_states`` / ``_push_robots`` read on either manager."""
    stub: Any = object.__new__(LeggedRobotLocomotionManager)
    stub.robot_config = SimpleNamespace(asset=SimpleNamespace(fix_base_link=not floating_base))
    stub.device = "cpu"
    stub.num_envs = _NUM_ENVS
    stub.simulator = MagicMock()
    stub.simulator.robot_root_states = _root_states()
    stub.simulator.dof_pos = torch.zeros(_NUM_ENVS, 2)
    stub.simulator.dof_vel = torch.ones(_NUM_ENVS, 2)
    stub.default_dof_pos = torch.ones(_NUM_ENVS, 2)
    stub.need_to_refresh_envs = torch.zeros(_NUM_ENVS, dtype=torch.bool)
    stub.randomization_manager = None
    stub._max_push_vel = torch.ones(push_dim)
    stub.push_robot_vel_buf = torch.zeros(_NUM_ENVS, push_dim)
    stub.record_push_robot_vel_buf = torch.zeros(_NUM_ENVS, push_dim)
    stub.base_init_state = torch.tensor([0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0] + [0.0] * 6)
    terrain = SimpleNamespace(env_origins=torch.arange(_NUM_ENVS * 3, dtype=torch.float).view(_NUM_ENVS, 3))
    terrain.custom_origins = False
    spawn = SimpleNamespace(xy_offset_range=1.0, query_terrain_height=False)
    stub.terrain_manager = SimpleNamespace(
        get_state=lambda _name: terrain, cfg=SimpleNamespace(terrain_term=SimpleNamespace(spawn=spawn))
    )
    stub._get_env_origins = lambda: terrain.env_origins
    return stub


# --------------------------------------------------------------------------- _reset_root_states


@pytest.mark.parametrize("floating_base", [True, False], ids=["floating", "fixed"])
def test_reset_root_states_writes_spawn_pose(floating_base: bool) -> None:
    """The buffer write is the same for both bases; the simulator decides what a fixed-base flush does."""
    torch.manual_seed(0)
    env = _env_stub(floating_base=floating_base, push_dim=2)
    before = env.simulator.robot_root_states.clone()
    origins = env.terrain_manager.get_state("locomotion_terrain").env_origins

    env._reset_robot_states_callback(_ENV_IDS)

    states = env.simulator.robot_root_states
    assert torch.equal(states[_ENV_IDS, :3], env.base_init_state[:3] + origins[_ENV_IDS])
    assert torch.equal(states[_ENV_IDS, 3:7], env.base_init_state[3:7].expand(len(_ENV_IDS), 4))
    vel = states[_ENV_IDS, 7:13]
    assert (vel.abs() <= 0.5).all()
    if floating_base:
        assert vel.abs().sum() > 0
    else:
        assert torch.equal(vel, torch.zeros_like(vel))
    assert torch.equal(states[_OTHER_IDS], before[_OTHER_IDS])
    assert env.simulator.mock_calls == []  # the flush happens in _refresh_envs_after_reset


def test_reset_root_states_spawn_jitter_is_floating_base_only() -> None:
    """On custom origins the XY spawn offset randomizes a floating base; a welded base keeps its mount pose."""
    torch.manual_seed(0)
    for floating_base in (True, False):
        env = _env_stub(floating_base=floating_base, push_dim=2)
        env.terrain_manager.get_state("locomotion_terrain").custom_origins = True
        spawn_xy = (env.base_init_state[:3] + env.terrain_manager.get_state("locomotion_terrain").env_origins)[
            _ENV_IDS, :2
        ]

        env._reset_robot_states_callback(_ENV_IDS)

        offset = env.simulator.robot_root_states[_ENV_IDS, :2] - spawn_xy
        if floating_base:
            assert (offset.abs() <= 1.0).all() and offset.abs().sum() > 0
        else:
            assert torch.equal(offset, torch.zeros_like(offset))


@pytest.mark.parametrize("floating_base", [True, False], ids=["floating", "fixed"])
def test_reset_root_states_target_adds_origins(floating_base: bool) -> None:
    env = _env_stub(floating_base=floating_base, push_dim=2)
    origins = env.terrain_manager.get_state("locomotion_terrain").env_origins
    target = torch.ones(len(_ENV_IDS), 13)

    env._reset_robot_states_callback(_ENV_IDS, {"root_states": target, "dof_states": torch.zeros(len(_ENV_IDS), 2, 2)})

    states = env.simulator.robot_root_states
    assert torch.equal(states[_ENV_IDS, :3], target[:, :3] + origins[_ENV_IDS])
    assert torch.equal(states[_ENV_IDS, 3:], target[:, 3:])


# --------------------------------------------------------------------------- _push_robots


@pytest.mark.parametrize(("manager", "push_dim"), _MANAGERS)
def test_push_robots_floating_base_updates_selected_velocities(manager: type[Any], push_dim: int) -> None:
    env = _env_stub(floating_base=True, push_dim=push_dim)
    before = env.simulator.robot_root_states.clone()

    manager._push_robots(env, _ENV_IDS)

    env.simulator.set_actor_root_state_tensor_robots.assert_not_called()
    pushed = env.push_robot_vel_buf[_ENV_IDS]
    assert (pushed.abs() <= env._max_push_vel).all()
    states = env.simulator.robot_root_states
    if manager is LeggedRobotLocomotionManager:
        assert torch.equal(states[_ENV_IDS, 7:9], pushed)  # planar velocity is overwritten
    else:
        assert torch.equal(states[_ENV_IDS, 7:13], before[_ENV_IDS, 7:13] + pushed)  # 6-dof additive push
    assert torch.equal(states[_OTHER_IDS], before[_OTHER_IDS])


# --------------------------------------------------------------------------- push guard (both entry points)


def _sampler() -> TermSampler:
    return TermSampler.bind(0, "test_fixed_base_env_guards", STAGE_RESET)


def _push_env(*, floating_base: bool, enabled: bool) -> tuple[Any, PushRandomizerState]:
    """A stub env carrying a real, set-up ``PushRandomizerState`` behind ``randomization_manager``."""
    env = SimpleNamespace(
        device="cpu",
        num_envs=_NUM_ENVS,
        dt=0.02,
        is_evaluating=False,
        robot_config=SimpleNamespace(asset=SimpleNamespace(fix_base_link=not floating_base)),
        _push_robots=MagicMock(),
    )
    cfg = RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion.PushRandomizerState",
        params={"max_push_vel": [1.0, 1.0], "push_interval_s": [5, 16], "enabled": enabled},
    )
    state = PushRandomizerState(cfg, env)
    state.setup(_sampler())
    env.randomization_manager = SimpleNamespace(get_state=lambda _name: state)
    return env, state


def _run_push_entry(entry: Any, env: Any, **params: Any) -> None:
    """Call either push entry point with its own positional shape."""
    if entry is randomize_push_schedule:
        entry(env, None, sampler=_sampler(), **params)
    else:
        entry(env, sampler=_sampler(), **params)


# entry point -> the env flag it publishes once the guard passes
_PUSH_ENTRY_FLAGS = {apply_pushes: "_push_robots_enabled"}
_PUSH_ENTRIES = [
    pytest.param(randomize_push_schedule, id="randomize_push_schedule"),
    pytest.param(apply_pushes, id="apply_pushes"),
]


@pytest.mark.parametrize("entry", _PUSH_ENTRIES)
@pytest.mark.parametrize("via_param", [False, True], ids=["configured", "term-param"])
def test_push_enabled_on_fixed_base_raises(entry: Any, via_param: bool) -> None:
    env, state = _push_env(floating_base=False, enabled=not via_param)
    params = {"enabled": True} if via_param else {}
    with pytest.raises(ValueError, match="requires a floating base"):
        _run_push_entry(entry, env, **params)
    assert state.enabled
    if entry in _PUSH_ENTRY_FLAGS:
        assert not hasattr(env, _PUSH_ENTRY_FLAGS[entry])
    assert not hasattr(env, "_max_push_vel")
    env._push_robots.assert_not_called()


@pytest.mark.parametrize("entry", _PUSH_ENTRIES)
def test_push_disabled_on_fixed_base_does_not_raise(entry: Any) -> None:
    env, state = _push_env(floating_base=False, enabled=False)
    _run_push_entry(entry, env)
    if entry in _PUSH_ENTRY_FLAGS:
        assert getattr(env, _PUSH_ENTRY_FLAGS[entry]) is False
    assert not state.enabled
    env._push_robots.assert_not_called()


def test_apply_pushes_enabled_on_floating_base_pushes_the_due_envs() -> None:
    env, state = _push_env(floating_base=True, enabled=True)
    assert state.push_robot_counter is not None and state.push_interval_s is not None
    interval_steps = int(5.0 / env.dt)
    state.push_interval_s[:] = 5.0
    state.push_robot_counter[:] = interval_steps - 1  # one step short
    state.push_robot_counter[0] = interval_steps  # due

    apply_pushes(env, sampler=_sampler())

    assert env._push_robots_enabled is True
    env._push_robots.assert_called_once()
    assert env._push_robots.call_args.args[0].tolist() == [0]
    assert torch.equal(env._max_push_vel, state.max_push_vel)
    assert state.push_robot_counter[0] == 0
    assert (state.push_robot_counter[1:] == interval_steps - 1).all()
    assert 5.0 <= float(state.push_interval_s[0]) <= 16.0  # env 0's interval was redrawn
    assert (state.push_interval_s[1:] == 5.0).all()


def test_push_schedule_enabled_on_floating_base_resamples() -> None:
    env, state = _push_env(floating_base=True, enabled=True)
    assert state.push_robot_counter is not None and state.push_interval_s is not None
    state.push_robot_counter[:] = 7

    randomize_push_schedule(env, _ENV_IDS, sampler=_sampler())

    assert state.enabled
    assert torch.equal(env._max_push_vel, state.max_push_vel)
    assert not state.push_robot_counter[_ENV_IDS].any()
    assert (state.push_robot_counter[_OTHER_IDS] == 7).all()
    assert ((state.push_interval_s >= 5.0) & (state.push_interval_s <= 16.0)).all()
