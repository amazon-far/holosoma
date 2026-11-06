"""Frame freshness through SimulationTask.run and the shared BaseTask control-frame schedule.

A control step's FRAME_END falls after the substep the backend renders on, so a camera consumer
egresses the frame drawn for the state written on that same control step.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch

from holosoma.config_types.env import EnvConfig
from holosoma.config_types.plugin import MotionPlaybackPluginConfig
from holosoma.envs.simulation_task import SimulationTask
from holosoma.managers.randomization import RandomizationManager
from holosoma.simulator.base_simulator.base_simulator import BaseSimulator
from holosoma.simulator.base_simulator.hooks import HookRegistry, Phase
from holosoma.simulator.shared.scene_types import EnvOriginsScene

FPS = 200
DECIMATION = 4
"""IsaacSim's registered defaults (holosoma.config_values.simulator): 200 Hz physics, 50 Hz control."""

SUBSTEP = "physics_substep"
"""Marker the fake records where the physics substep runs, so hooks are placed relative to it."""


class _LoopOverranError(AssertionError):
    """The loop outlived the fake's substep budget."""


class _FakeSim:
    """Stand-in simulator that draws every ``render_interval`` substeps and records the loop's phases.

    A playback-style plugin samples on FRAME_BEGIN, writes on PRE_STEP and egresses the last draw on
    FRAME_END. A substep budget makes a loop that never finishes fail instead of hanging.
    """

    def __init__(self, *, frames: int, render_interval: int = DECIMATION, decimation: int = DECIMATION) -> None:
        self.hooks = HookRegistry()
        self.decimation = decimation
        self.render_interval = render_interval
        self.frames = frames
        self._budget = (frames + 2) * decimation
        self.substeps = 0
        self._shutdown = False
        self._sampled: int | None = None
        self._written: int | None = None
        self._rendered: object = None
        self.rendered_at: int | None = None
        self.control_steps = 0
        self.captured: list[object] = []
        self.drawn_at_egress: list[int | None] = []
        self.phases: list[Phase | str] = []
        self.viewer_renders: list[int] = []
        self.video_recorder = None

        self.hooks.add(Phase.FRAME_BEGIN, self._sample, name="fake.sample")
        self.hooks.add(Phase.PRE_STEP, self._write, name="fake.write")
        self.hooks.add(Phase.POST_STEP, self._post_step, name="fake.post_step")
        self.hooks.add(Phase.FRAME_END, self._egress, name="fake.egress")

    @property
    def shutdown_requested(self) -> bool:
        return self._shutdown or len(self.captured) >= self.frames

    def request_shutdown(self, reason: str = "") -> None:
        self._shutdown = True

    def requires_step_render(self) -> bool:
        return False

    def render(self, sync_frame_time: bool = True) -> None:
        self.viewer_renders.append(self.substeps)

    def simulate_at_each_physics_step(self) -> None:
        self.substeps += 1
        if self.substeps > self._budget:
            raise _LoopOverranError(
                f"the loop ran {self.substeps} substeps without reaching {self.frames} frame(s) "
                f"(budget {self._budget}); egressed {len(self.captured)}"
            )
        self.phases.append(SUBSTEP)
        if self.substeps % self.render_interval == 0:
            self._rendered = self._scene_state()
            self.rendered_at = self.substeps

    def _scene_state(self) -> object:
        """What a draw captures: the value PRE_STEP wrote."""
        return self._written

    def _sample(self) -> None:
        self.phases.append(Phase.FRAME_BEGIN)
        self._sampled = self.control_steps
        self.control_steps += 1

    def _write(self) -> None:
        self.phases.append(Phase.PRE_STEP)
        self._written = self._sampled

    def _post_step(self) -> None:
        self.phases.append(Phase.POST_STEP)

    def _egress(self) -> None:
        self.phases.append(Phase.FRAME_END)
        self.captured.append(self._rendered)
        self.drawn_at_egress.append(self.rendered_at)


class _LoopTask(SimulationTask):
    """The production scheduler and run loop over a fake simulator, without backend construction."""

    def __init__(self, sim: _FakeSim, *, headless: bool = True, viewer_dt: float = 1.0) -> None:
        self.simulator = cast("BaseSimulator", sim)
        self.hooks = sim.hooks
        self.config = cast(
            "EnvConfig",
            SimpleNamespace(
                viewer_dt=viewer_dt,
                robot=SimpleNamespace(asset=SimpleNamespace(robot_type="test")),
                simulator=SimpleNamespace(
                    _target_="test.Simulator",
                    config=SimpleNamespace(
                        sim=SimpleNamespace(
                            fps=FPS,
                            control_decimation_steps=sim.decimation,
                            render_interval_steps=sim.render_interval,
                        )
                    ),
                ),
                terrain=SimpleNamespace(terrain_term=SimpleNamespace(mesh_type="plane", func="test.terrain")),
            ),
        )
        self.device = "cpu"
        self.num_envs = 1
        self.viewer = None if headless else object()
        self.sim_dt = 1 / FPS
        self._physics_step_count = 0
        self._next_viewer_time = 0.0
        self._rate_limiter = None
        self._shutdown_requested = False
        self.randomization_manager = cast(
            "RandomizationManager", SimpleNamespace(step=lambda: None, reset=lambda _env_ids: None)
        )
        self._episode_active = torch.zeros(self.num_envs, dtype=torch.bool)
        self._init_buffers()

    def reset_all(self) -> None:
        self._start_episodes(torch.arange(self.num_envs))


def _run(sim: _FakeSim, *, headless: bool = True, viewer_dt: float = 1.0) -> None:
    """Drive the real SimulationTask loop over ``sim``, unthrottled."""
    _LoopTask(sim, headless=headless, viewer_dt=viewer_dt).run()


def _expected_phases(groups: int, decimation: int = DECIMATION) -> list[Phase | str]:
    """The schedule ``BaseTask._physics_step`` emits: FRAME_BEGIN, every substep, then FRAME_END."""
    substeps: list[Phase | str] = [Phase.PRE_STEP, SUBSTEP, Phase.POST_STEP] * decimation
    return [Phase.FRAME_BEGIN, *substeps, Phase.FRAME_END] * groups


# ----- frame freshness -----


@pytest.mark.parametrize(
    ("render_interval", "decimation", "drawn_at_egress"),
    [
        (1, DECIMATION, [4, 8, 12]),
        (2, DECIMATION, [4, 8, 12]),
        (DECIMATION, DECIMATION, [4, 8, 12]),
        # An interval that does not divide the decimation draws before the frame's last substep.
        (3, DECIMATION, [3, 6, 12]),
        (1, 1, [1, 2, 3]),
    ],
)
def test_each_frame_egresses_the_draw_of_its_own_control_step(
    render_interval: int, decimation: int, drawn_at_egress: list[int]
) -> None:
    """FRAME_END egresses the last draw of the state written on the same control step."""
    sim = _FakeSim(frames=3, render_interval=render_interval, decimation=decimation)

    _run(sim)

    assert sim.captured == [0, 1, 2]
    assert sim.drawn_at_egress == drawn_at_egress
    assert sim.phases == _expected_phases(groups=3, decimation=decimation)


def test_every_control_step_egresses_exactly_one_frame() -> None:
    """One FRAME_BEGIN and one FRAME_END per control step."""
    sim = _FakeSim(frames=5)

    _run(sim)

    assert len(sim.captured) == 5
    assert sim.control_steps == 5
    assert sim.phases.count(Phase.FRAME_BEGIN) == 5
    assert sim.phases.count(Phase.FRAME_END) == 5
    assert sim.substeps == 5 * DECIMATION


# ----- shutdown and the viewer -----


def test_shutdown_asked_for_inside_a_frame_still_egresses_that_frame() -> None:
    """request_shutdown stops the run after the current frame, which still egresses."""
    sim = _FakeSim(frames=99)

    def ask_mid_frame() -> None:
        if sim.substeps == DECIMATION + 1:  # first substep of the second frame
            sim.request_shutdown("test")

    sim.hooks.add(Phase.POST_STEP, ask_mid_frame, name="test.ask")

    _run(sim)

    assert sim.captured == [0, 1]
    assert sim.substeps == 2 * DECIMATION


def test_a_frame_that_opens_always_closes() -> None:
    """Shutdown asked for from FRAME_BEGIN still closes that frame, so the phases stay paired."""
    sim = _FakeSim(frames=99)

    def ask_on_frame_begin() -> None:
        if sim.control_steps == 3:  # the third frame asks to stop while being opened
            sim.request_shutdown("test")

    sim.hooks.add(Phase.FRAME_BEGIN, ask_on_frame_begin, name="test.ask")

    _run(sim)

    assert sim.captured == [0, 1, 2]
    assert sim.phases == _expected_phases(groups=3)
    assert sim.substeps == 3 * DECIMATION


def test_viewer_renders_on_its_own_schedule_when_not_headless() -> None:
    """A windowed run draws the viewport every ``viewer_dt``, here on each frame's first substep."""
    sim = _FakeSim(frames=3)

    _run(sim, headless=False, viewer_dt=DECIMATION / FPS)

    assert sim.viewer_renders == [1, 5, 9]


def test_headless_never_renders_the_viewport() -> None:
    """Headless leaves the viewport alone even at a viewer_dt that would draw every step."""
    sim = _FakeSim(frames=3)

    _run(sim, headless=True, viewer_dt=1.0 / FPS)

    assert sim.viewer_renders == []


class _OffscreenSim(_FakeSim):
    """A backend with offscreen outputs (RTX cameras, video), so BaseTask schedules its renders."""

    def requires_step_render(self) -> bool:
        return True


@pytest.mark.parametrize("render_interval", [1, 2, DECIMATION])
def test_scheduled_renders_land_on_the_last_substep_of_each_interval(render_interval: int) -> None:
    """BaseTask renders offscreen outputs after every ``render_interval``-th substep (1-based)."""
    sim = _OffscreenSim(frames=3, render_interval=render_interval)

    _run(sim)

    assert sim.viewer_renders == list(range(render_interval, 3 * DECIMATION + 1, render_interval))


# ----- integration: the real playback plugin driven by the real loop -----


CLIP_FRAMES = 5
CLIP_JOINT0 = [10.0 + i for i in range(CLIP_FRAMES)]
"""``joint0`` per clip frame."""


@pytest.fixture
def local_path_loaders(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register the production local path loaders directly instead of through entry points."""
    from holosoma.utils import core_path_loaders, path

    monkeypatch.setattr(
        path,
        "_entrypoint_loaders",
        lambda: (
            (path._RegisteredLoader(core_path_loaders.local_path_loader, "local"),),
            (path._RegisteredLoader(core_path_loaders.local_paths_loader, "local"),),
        ),
    )


def _clip(tmp_path: Path, name: str) -> str:
    """A ``CLIP_FRAMES``-row floating-base clip whose ``joint0`` counts 10, 11, 12, ... per frame."""
    frames = np.arange(CLIP_FRAMES, dtype=np.float32)
    root = np.zeros((CLIP_FRAMES, 7), dtype=np.float32)
    root[:, 3] = 1.0  # identity quat, wxyz
    joints = np.stack([10 + frames, 20 + frames], axis=1).astype(np.float32)
    target = tmp_path / name
    np.save(target, np.concatenate([root, joints], axis=1))
    return str(target)


class _PlaybackSim(_FakeSim):
    """``_FakeSim`` plus the surface MotionPlaybackPlugin writes through; a draw captures ``joint0``."""

    def __init__(self, *, decimation: int, budget_frames: int) -> None:
        super().__init__(frames=budget_frames, render_interval=decimation, decimation=decimation)
        self.num_envs = 1
        self.num_dof = 2
        self.sim_device = "cpu"
        self.dof_pos = torch.zeros(1, self.num_dof)
        self.dof_vel = torch.zeros(1, self.num_dof)
        self.robot_root_states = torch.zeros(1, 13)
        self.scene = EnvOriginsScene(torch.zeros(1, 3), self.sim_device)
        self.simulator_config = SimpleNamespace(
            sim=SimpleNamespace(
                fps=FPS, control_decimation_steps=self.decimation, render_interval_steps=self.render_interval
            )
        )

    @property
    def shutdown_requested(self) -> bool:
        return self._shutdown

    def set_actor_states(self, names: list[str], env_ids: torch.Tensor, states: torch.Tensor) -> None:
        pass

    def _scene_state(self) -> object:
        return round(float(self.dof_pos[0, 0]), 4)


@pytest.mark.parametrize("clips", [1, 2])
@pytest.mark.parametrize("decimation", [1, 2, DECIMATION])
def test_playback_egresses_every_clip_frame_exactly_once(
    tmp_path: Path, local_path_loaders: None, decimation: int, clips: int
) -> None:
    """A finite playlist egresses each clip frame once: no stale first frame, no doubled last one."""
    sim = _PlaybackSim(decimation=decimation, budget_frames=(CLIP_FRAMES + 2) * clips)
    cfg = MotionPlaybackPluginConfig(
        motion_files=[_clip(tmp_path, f"clip{index}.npy") for index in range(clips)],
        fps=FPS / decimation,  # one clip frame per control step
        n_run=1,
    )
    cfg.get_cls()(cfg, sim)

    _run(sim)

    assert sim.captured == CLIP_JOINT0 * clips
