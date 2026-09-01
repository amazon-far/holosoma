"""Tests for non-mutating reads of stateful observation terms."""

from __future__ import annotations

import types

import pytest
from holosoma.config_types.observation import ObsTermCfg
from holosoma.managers.observation.manager import ObservationManager
from holosoma.managers.observation.terms.depth import WarpDepthImageObsTerm
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim


def test_compute_group_forwards_modify_history():
    calls = []

    def compute_term(group_name, term_name, term_cfg, *, modify_history=True):
        calls.append((group_name, term_name, modify_history))
        return torch.ones(2, 1)

    term_cfg = types.SimpleNamespace(noise=0.0, scale=1.0, clip=None)
    group_cfg = types.SimpleNamespace(
        terms={"image": term_cfg},
        concatenate=False,
        enable_noise=False,
        history_length=1,
    )
    stub = types.SimpleNamespace(
        cfg=types.SimpleNamespace(groups={"depth": group_cfg}),
        _compute_term=compute_term,
    )

    output = ObservationManager.compute_group(stub, "depth", modify_history=False)

    assert torch.equal(output["image"], torch.ones(2, 1))
    assert calls == [("depth", "image", False)]


def test_compute_term_forwards_flag_to_stateful_instance():
    calls = []

    class StatefulTerm:
        def __call__(self, env, **kwargs):
            calls.append(kwargs.copy())
            return torch.ones(2, 1)

    stub = types.SimpleNamespace(
        env=object(),
        _term_instances={"depth": {"image": StatefulTerm()}},
        _term_funcs={"depth": {}},
        _term_supports_modify_history={"depth": {"image": True}},
    )
    cfg = ObsTermCfg(func="unused", params={"configured": 7})

    ObservationManager._compute_term(stub, "depth", "image", cfg, modify_history=False)

    assert calls == [{"modify_history": False, "configured": 7}]


def test_legacy_stateful_term_without_flag_remains_compatible():
    calls = []

    class LegacyTerm:
        def __call__(self, env, *, field):
            calls.append(field)
            return torch.ones(1, 1)

    term = LegacyTerm()
    stub = types.SimpleNamespace(
        env=object(),
        _term_instances={"legacy": {"field": term}},
        _term_funcs={"legacy": {}},
        _term_supports_modify_history={"legacy": {"field": False}},
    )
    cfg = ObsTermCfg(func="unused", params={"field": "cached"})

    ObservationManager._compute_term(stub, "legacy", "field", cfg, modify_history=False)

    assert calls == ["cached"]
    assert not ObservationManager._accepts_modify_history(term)


def test_shape_inference_is_non_mutating():
    calls = []

    def compute_term(group_name, term_name, term_cfg, *, modify_history=True):
        calls.append((group_name, term_name, modify_history))
        return torch.zeros(2, 3, 4)

    group_cfg = types.SimpleNamespace(
        concatenate=False,
        terms={"image": ObsTermCfg(func="unused")},
    )
    stub = types.SimpleNamespace(
        cfg=types.SimpleNamespace(groups={"depth": group_cfg}),
        _compute_term=compute_term,
    )

    dims = ObservationManager.get_obs_dims(stub)

    assert dims == {"depth": {"image": (3, 4)}}
    assert calls == [("depth", "image", False)]


def _depth_term_for_buffer_test(*, latency_frame):
    term = object.__new__(WarpDepthImageObsTerm)
    term.num_envs = 2
    term.device = "cpu"
    term.buffer_len = 3
    term.latency_frame_range = latency_frame if isinstance(latency_frame, tuple) else None
    term.latency_frame = None if term.latency_frame_range is not None else latency_frame
    term.depth_buffer = torch.zeros(2, 3, 1, 2)
    term._cached_output = torch.zeros(2, 1, 2)
    term.reset_episodes = torch.tensor([False, True])
    term.command = types.SimpleNamespace(time_steps=torch.tensor([5, 1]))
    term._prev_time_step = torch.tensor([4, 2])
    term._ensure_command_resolved = lambda _env: None
    frame = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])
    term._get_depth_images = lambda _env: frame
    term._process_depth_images = lambda images, env_ids: images
    return term, frame


def test_depth_normal_read_advances_and_reset_fills_all_slots():
    term, frame = _depth_term_for_buffer_test(latency_frame=1)
    env = types.SimpleNamespace(device="cpu")

    output = term(env, modify_history=True)

    assert torch.equal(term.depth_buffer[1, 0], frame[1])
    assert torch.equal(term.depth_buffer[1, 1], frame[1])
    assert torch.equal(term.depth_buffer[1, 2], frame[1])
    assert torch.equal(output[1], frame[1])
    assert not term.reset_episodes.any()


def test_depth_non_mutating_read_preserves_buffer_cache_and_rng():
    term, _ = _depth_term_for_buffer_test(latency_frame=1)
    env = types.SimpleNamespace(device="cpu")
    term._cached_output = torch.arange(4, dtype=torch.float32).reshape(2, 1, 2)
    term._get_depth_images = lambda _env: pytest.fail("non-mutating read rendered depth")
    frames_before = term.depth_buffer.clone()
    reset_before = term.reset_episodes.clone()
    rng_before = torch.random.get_rng_state().clone()

    first = term(env, modify_history=False)
    second = term(env, modify_history=False)

    assert torch.equal(first, term._cached_output)
    assert torch.equal(first, second)
    assert torch.equal(term.depth_buffer, frames_before)
    assert torch.equal(term.reset_episodes, reset_before)
    assert torch.equal(torch.random.get_rng_state(), rng_before)


def test_depth_ranged_latency_selects_requested_slots(monkeypatch):
    term, frame = _depth_term_for_buffer_test(latency_frame=(1, 2))
    env = types.SimpleNamespace(device="cpu")

    def fixed_latency(*args, **kwargs):
        return torch.tensor([1, 2])

    monkeypatch.setattr(torch, "randint", fixed_latency)
    output = term(env, modify_history=True)

    # Env 0 is not reset: latency 1 selects its previous zero slot. Env 1 is
    # reset: every slot contains the new frame, so latency 2 also returns it.
    assert torch.equal(output[0], torch.zeros_like(frame[0]))
    assert torch.equal(output[1], frame[1])


def test_final_read_then_reset_advances_non_reset_env_only_once():
    term, frame = _depth_term_for_buffer_test(latency_frame=1)
    env = types.SimpleNamespace(device="cpu")
    term.reset_episodes[:] = False
    term.command.time_steps = torch.tensor([5, 3])
    term._prev_time_step = torch.tensor([4, 2])
    term.depth_buffer[0, :, 0, 0] = torch.tensor([10.0, 20.0, 30.0])
    term.depth_buffer[0, :, 0, 1] = torch.tensor([11.0, 21.0, 31.0])
    term._cached_output = term.depth_buffer[:, -1].clone()

    final_depth = term(env, modify_history=False)
    term.reset(torch.tensor([1]))
    post_reset_depth = term(env, modify_history=True)

    # The final read is cached and does not shift env 0. The post-reset normal
    # read performs its single shift: [old1, old2, new].
    assert torch.equal(final_depth[0], torch.tensor([[30.0, 31.0]]))
    assert torch.equal(term.depth_buffer[0, 0], torch.tensor([[20.0, 21.0]]))
    assert torch.equal(term.depth_buffer[0, 1], torch.tensor([[30.0, 31.0]]))
    assert torch.equal(term.depth_buffer[0, 2], frame[0])
    # Reset env 1 is initialized entirely from its first post-reset frame.
    assert all(torch.equal(term.depth_buffer[1, i], frame[1]) for i in range(term.buffer_len))
    assert torch.equal(post_reset_depth[1], frame[1])
