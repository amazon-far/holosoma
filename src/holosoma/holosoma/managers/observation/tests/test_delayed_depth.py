"""Delayed depth term: preprocessing, latency history, refills, row-subset reads, and augmentation toggles."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from holosoma.config_types.depth import (
    DepthAugmentationConfig,
    DepthEdgeNoiseConfig,
    DepthGaussianNoiseConfig,
    DepthHolesConfig,
    DepthOffsetConfig,
    DepthProcessingConfig,
)
from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg
from holosoma.managers.observation.manager import ObservationManager
from holosoma.managers.observation.terms.depth import DelayedDepthImageObsTerm
from holosoma.utils.depth_processing import DepthAugmenter, preprocess_depth, resize_depth
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim

NUM_ENVS = 2
HEIGHT, WIDTH = 6, 8


class _Camera:
    """Serves a constant depth frame per step: frame k is filled with 0.5 + 0.1 * k meters."""

    def __init__(self) -> None:
        self.step = 0
        self.sensor_config = {"depth": SimpleNamespace(height=HEIGHT, width=WIDTH)}

    def depth(self, step: int) -> float:
        return 0.5 + 0.1 * step

    def get_camera_data(self, name: str, data_type: str) -> torch.Tensor:
        assert (name, data_type) == ("depth", "depth")
        return torch.full((NUM_ENVS, HEIGHT, WIDTH, 1), self.depth(self.step))

    def rerender_sensors(self, env_ids: torch.Tensor) -> None:
        pass


def _term(**params: Any) -> tuple[DelayedDepthImageObsTerm, Any]:
    camera = _Camera()
    env = SimpleNamespace(num_envs=NUM_ENVS, device="cpu", simulator=camera)
    params = {
        "sensor": "depth",
        "latency_frames": 2,
        "buffer_len": 4,
        "processing": {"near": 0.0, "far": 2.0},
        **params,
    }
    return DelayedDepthImageObsTerm(SimpleNamespace(params=params, func="depth"), env), env


def _normalized(depth_m: float, near: float = 0.0, far: float = 2.0) -> float:
    return (depth_m - near) / (far - near) - 0.5


def _served(term: DelayedDepthImageObsTerm, env: Any) -> list[float]:
    out = term.read(env)
    env.simulator.step += 1
    return [round(v, 6) for v in out[:, 0, 0].tolist()]


# --------------------------------------------------------------- preprocessing


def test_preprocessing_clamps_crops_resizes_and_normalizes() -> None:
    cfg = DepthProcessingConfig(near=0.3, far=3.0, crop_top=1, crop_left=2, resize=[4, 5], empty_threshold=0.15)
    depth = torch.full((1, HEIGHT, WIDTH), 1.5)
    depth[0, 0, 0] = float("inf")  # cropped away
    depth[0, -1, -1] = float("nan")  # no return -> far
    depth[0, 2, 3] = 0.1  # closer than near -> near

    out = preprocess_depth(depth, cfg)

    assert out.shape == (1, 4, 5)
    assert out.max().item() <= 0.5 + 1e-6
    metric = resize_depth(depth, cfg)
    assert torch.all(metric >= cfg.empty_threshold) and torch.all(metric <= cfg.far)
    assert torch.isfinite(out).all()


def test_values_below_the_empty_threshold_report_far() -> None:
    cfg = DepthProcessingConfig(near=0.0, far=3.0, empty_threshold=0.15)
    depth = torch.tensor([[[0.05, 1.0]]])

    assert resize_depth(depth, cfg).tolist() == [[[3.0, 1.0]]]


# ------------------------------------------------------------------- history


def test_fixed_latency_serves_the_frame_from_n_steps_ago_after_an_initial_refill() -> None:
    term, env = _term()

    served = [_served(term, env)[0] for _ in range(5)]

    # The first read refills the whole history with frame 0, so the delay ramps in from it.
    expected = [_normalized(env.simulator.depth(max(0, k - 2))) for k in range(5)]
    assert served == pytest.approx(expected, abs=1e-6)


def test_reset_envs_are_refilled_with_the_current_frame_and_others_keep_their_delay() -> None:
    term, env = _term()
    for _ in range(4):
        _served(term, env)

    term.reset(torch.tensor([0]))
    env0, env1 = _served(term, env)

    assert env0 == pytest.approx(_normalized(env.simulator.depth(4)), abs=1e-6)
    assert env1 == pytest.approx(_normalized(env.simulator.depth(2)), abs=1e-6)


def test_an_empty_reset_after_reset_all_keeps_the_pending_refill() -> None:
    # reset_all resets every env and then steps once, which resets an empty set of envs: the
    # refill requested by the first must survive the second.
    term, env = _term()
    for _ in range(4):
        _served(term, env)

    term.reset(None)
    term.reset(torch.tensor([], dtype=torch.long))

    assert _served(term, env) == pytest.approx([_normalized(env.simulator.depth(4))] * 2, abs=1e-6)


def test_ranged_latency_is_sampled_per_env_within_bounds() -> None:
    term, env = _term(latency_frames=[1, 3], buffer_len=4)
    for _ in range(8):
        _served(term, env)
        assert torch.all((term.latency >= 1) & (term.latency <= 3))


def test_a_row_subset_read_matches_a_full_read_and_advances_only_those_rows() -> None:
    def warmed_up() -> tuple[DelayedDepthImageObsTerm, Any]:
        torch.manual_seed(0)
        term, env = _term(augmentation={"gaussian_noise": {"enabled": True}})
        for _ in range(3):
            _served(term, env)
        return term, env

    term, env = warmed_up()
    twin, twin_env = warmed_up()
    other_row = term.buffer[1].clone()
    rng = torch.get_rng_state()

    subset = term.read(env, torch.tensor([0]))

    # With a latency of two frames the served frame is already buffered (augmented).
    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.equal(term.buffer[1], other_row)
    torch.testing.assert_close(subset, twin.read(twin_env)[:1])


def test_a_subset_read_appends_the_unaugmented_current_frame_for_its_rows() -> None:
    term, env = _term(latency_frames=0, augmentation={"gaussian_noise": {"enabled": True}})
    _served(term, env)

    subset = term.read(env, torch.tensor([1]))

    assert subset[:, 0, 0].tolist() == pytest.approx([_normalized(env.simulator.depth(1))], abs=1e-6)


def test_shape_is_the_processed_frame_shape_and_reads_no_camera() -> None:
    term, env = _term(processing={"near": 0.0, "far": 2.0, "crop_top": 1, "crop_right": 2})
    env.simulator = None  # sizing must not touch the camera
    assert term.shape(env) == (HEIGHT - 1, WIDTH - 2)
    term, env = _term(processing={"near": 0.0, "far": 2.0, "resize": [3, 4]})
    assert term.shape(env) == (3, 4)


def test_the_observation_manager_sizes_steps_and_refills_the_term() -> None:
    camera = _Camera()
    env = SimpleNamespace(num_envs=NUM_ENVS, device="cpu", simulator=camera)
    term = ObsTermCfg(
        func="holosoma.managers.observation.terms.depth:DelayedDepthImageObsTerm",
        params={"sensor": "depth", "latency_frames": 2, "buffer_len": 4, "processing": {"near": 0.0, "far": 2.0}},
        global_clip=False,
    )
    cfg = ObservationManagerCfg(groups={"depth": ObsGroupCfg(terms={"image": term}, concatenate=False)})
    manager = ObservationManager(cfg, env, device="cpu")
    assert manager.get_obs_dims()["depth"] == {"image": (HEIGHT, WIDTH)}

    def step() -> list[float]:
        manager.new_frame()
        out = manager.step()["depth"]["image"]
        camera.step += 1
        return [round(v, 6) for v in out[:, 0, 0].tolist()]

    for _ in range(4):
        step()
    manager.new_frame()
    terminal = manager.step_terminal(torch.tensor([0]))["depth"]["image"]
    manager.reset(torch.tensor([0]))
    env0, env1 = [round(v, 6) for v in manager.step()["depth"]["image"][:, 0, 0].tolist()]

    assert terminal[0, 0, 0].item() == pytest.approx(_normalized(camera.depth(2)), abs=1e-6)
    assert env0 == pytest.approx(_normalized(camera.depth(4)), abs=1e-6)  # refilled at its new state
    assert env1 == pytest.approx(_normalized(camera.depth(2)), abs=1e-6)  # keeps its delay


def test_latency_must_fit_in_the_buffer() -> None:
    with pytest.raises(ValueError, match="buffer_len"):
        _term(latency_frames=4, buffer_len=4)


# -------------------------------------------------------------- augmentation


def _scene() -> torch.Tensor:
    """Two envs of a near box in front of a far wall, giving sharp depth edges."""
    depth = torch.full((NUM_ENVS, 24, 32), 2.8)
    depth[:, 8:16, 10:22] = 0.8
    return depth


def test_disabled_augmentation_is_the_identity() -> None:
    augmenter = DepthAugmenter(DepthAugmentationConfig(), NUM_ENVS, far=3.0, device="cpu")
    depth = _scene()
    rng = torch.get_rng_state()

    assert torch.equal(augmenter(depth), depth)
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("effect", ["edge_noise", "holes", "gaussian_noise", "per_env_offset"])
def test_each_effect_can_be_enabled_on_its_own(effect: str) -> None:
    torch.manual_seed(0)
    enabled = {
        "edge_noise": DepthEdgeNoiseConfig(enabled=True),
        "holes": DepthHolesConfig(enabled=True),
        "gaussian_noise": DepthGaussianNoiseConfig(enabled=True),
        "per_env_offset": DepthOffsetConfig(enabled=True),
    }
    cfg = DepthAugmentationConfig(**{effect: enabled[effect]})  # type: ignore[arg-type]
    augmenter = DepthAugmenter(cfg, NUM_ENVS, far=3.0, device="cpu")
    # Holes only open at mid-range depths, so give them a scene where every pixel qualifies.
    depth = torch.full((NUM_ENVS, 24, 32), 1.0) if effect == "holes" else _scene()

    out = augmenter(depth)

    assert not torch.equal(out, depth)
    if effect == "per_env_offset":
        # A constant shift per env, identical on every frame.
        shift = out - depth
        assert torch.allclose(shift, shift[:, :1, :1].expand_as(shift))
        torch.testing.assert_close(augmenter(depth) - depth, shift)
    if effect in ("edge_noise", "holes"):
        # These only ever drop pixels to far or copy existing depths.
        assert set(out.unique().tolist()) <= {*depth.unique().tolist(), 3.0}


def test_hole_seed_changes_the_hole_pattern() -> None:
    depth = torch.full((NUM_ENVS, 24, 32), 1.0)

    def holes(seed: int | None) -> torch.Tensor:
        cfg = DepthAugmentationConfig(holes=DepthHolesConfig(enabled=True, seed=seed))
        return DepthAugmenter(cfg, NUM_ENVS, far=3.0, device="cpu")(depth) == 3.0

    assert torch.equal(holes(None), holes(None))
    assert not torch.equal(holes(None), holes(7))
