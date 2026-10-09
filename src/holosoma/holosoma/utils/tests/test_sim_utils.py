"""Unit tests for the direct-simulation helpers (pure, no simulator backend).

Covers the performance formatter, ``_calculate_viewer_steps``, and the phase schedule
:meth:`DirectSimulation.run` drives — above all that a control step's FRAME_END falls after the
substep the backend renders on, so a camera consumer egresses the frame drawn for the state written
on that same control step rather than the previous one's.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import torch

from holosoma.config_types.plugin import MotionPlaybackPluginConfig
from holosoma.simulator.base_simulator.hooks import HookRegistry, Phase
from holosoma.utils.sim_utils import DirectSimulation, _format_simulation_performance
from holosoma.utils.simulator_config import SimulatorType

if TYPE_CHECKING:
    from holosoma.config_types.run_sim import RunSimConfig

FPS = 200
DECIMATION = 4
"""IsaacSim's registered defaults (holosoma.config_values.simulator): 200 Hz physics, 50 Hz control."""

SUBSTEP = "physics_substep"
"""Marker the fake records where the physics substep runs, so hooks are placed relative to it.

Without it a phase log cannot tell PRE_STEP/POST_STEP bracketing the substep from both firing on
the same side of it, and POST_STEP consumers are documented to read post-step state.
"""


def test_format_simulation_performance_names_rates_and_time_ratio() -> None:
    message = _format_simulation_performance(
        physics_steps=1000,
        control_steps=250,
        sim_dt=0.005,
        wall_time=2.5,
    )

    assert message == (
        "Simulation performance: 2.00x real time (5.00 sim s / 2.50 wall s) | "
        "400.0 physics steps/s | 100.0 control steps/s"
    )


def test_format_simulation_performance_handles_empty_measurement() -> None:
    assert (
        _format_simulation_performance(
            physics_steps=0,
            control_steps=0,
            sim_dt=0.005,
            wall_time=0.0,
        )
        == "Simulation performance: unavailable (no wall time elapsed)"
    )


class _LoopOverranError(AssertionError):
    """The loop outlived the fake's substep budget: a stalled run must fail, not spin forever."""


class _FakeSim:
    """Stand-in simulator that reproduces a backend's substep render gate.

    A backend draws once its own substep counter reaches ``render_interval`` (IsaacSim's
    ``forward_kinematics``/``_step_dynamics``, whose counter runs over substeps and is never reset),
    and a camera consumer on FRAME_END reads whatever that last draw left in the buffers. Carrying
    written/rendered/egressed state makes the loop's schedule observable without a real backend: a
    playback-style plugin samples on FRAME_BEGIN, writes on PRE_STEP, and egresses on FRAME_END.

    The substep budget is enforced from ``simulate_at_each_physics_step`` rather than
    ``shutdown_requested``, because that is the one call every iteration of the loop must make: a
    regression that stops egress, or drops the shutdown check altogether, then fails an assertion
    instead of hanging the suite.
    """

    def __init__(self, *, frames: int, render_interval: int = DECIMATION, decimation: int = DECIMATION) -> None:
        self.hooks = HookRegistry()
        self.decimation = decimation
        self.render_interval = render_interval
        self.frames = frames
        self.sensor_config: dict[str, object] = {}
        self._budget = (frames + 2) * decimation
        self.substeps = 0
        self.refreshes = 0
        self._shutdown = False
        self._sampled: int | None = None
        self._written: int | None = None
        self._rendered: object = None
        self.rendered_at: int | None = None
        self.tensors_at: int | None = None
        self.control_steps = 0
        self.captured: list[object] = []
        self.alignment: list[tuple[int | None, int | None]] = []
        self.phases: list[Phase | str] = []
        self.viewer_renders: list[int] = []

        self.hooks.add(Phase.FRAME_BEGIN, self._sample, name="fake.sample")
        self.hooks.add(Phase.PRE_STEP, self._write, name="fake.write")
        self.hooks.add(Phase.POST_STEP, self._post_step, name="fake.post_step")
        self.egress_hook = self.hooks.add(Phase.FRAME_END, self._egress, name="fake.egress")

    @property
    def shutdown_requested(self) -> bool:
        """End on the wanted frame count, or when a hook asked — the behavior under test either way."""
        return self._shutdown or len(self.captured) >= self.frames

    def request_shutdown(self, reason: str = "") -> None:
        self._shutdown = True

    def refresh_sim_tensors(self) -> None:
        """Stand in for resyncing the state tensors: record which substep they now describe."""
        self.refreshes += 1
        self.tensors_at = self.substeps

    def render(self) -> None:
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
        """What a draw would capture — here the value PRE_STEP wrote, as a subclass may redefine."""
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
        self.alignment.append((self.tensors_at, self.rendered_at))


def _run_sim_config(
    *, decimation: int, render_interval: int = DECIMATION, headless: bool = True, viewer_dt: float = 1.0
) -> RunSimConfig:
    """The few RunSimConfig fields the direct loop reads, cast to satisfy the constructor.

    Building a real RunSimConfig would drag in a robot, simulator and experiment tree the loop never
    touches, and would bury which settings actually drive it.
    """
    return cast(
        "RunSimConfig",
        SimpleNamespace(
            simulator=SimpleNamespace(
                config=SimpleNamespace(
                    sim=SimpleNamespace(
                        fps=FPS, control_decimation_steps=decimation, render_interval_steps=render_interval
                    )
                )
            ),
            training=SimpleNamespace(headless=headless),
            time_scale=0.0,
            viewer_dt=viewer_dt,
        ),
    )


def _run(
    sim: _FakeSim,
    monkeypatch: pytest.MonkeyPatch,
    *,
    backend: SimulatorType = SimulatorType.ISAACSIM,
    headless: bool = True,
    viewer_dt: float = 1.0,
) -> None:
    """Drive the real loop over ``sim``: unthrottled, and headless unless a viewer test says not.

    The backend is reported by patching the loader the module resolved, leaving the process-wide
    simulator-type singleton alone so ordering against other tests cannot matter.
    """
    monkeypatch.setattr("holosoma.utils.sim_utils.get_simulator_type", lambda: backend)
    config = _run_sim_config(
        decimation=sim.decimation, render_interval=sim.render_interval, headless=headless, viewer_dt=viewer_dt
    )
    DirectSimulation(config, SimpleNamespace(sim=sim), device="cpu", simulation_app=None).run()


def _expected_phases(groups: int, decimation: int = DECIMATION) -> list[Phase | str]:
    """The schedule ``BaseTask._physics_step`` emits: FRAME_BEGIN, every substep, then FRAME_END."""
    substeps: list[Phase | str] = [Phase.PRE_STEP, SUBSTEP, Phase.POST_STEP] * decimation
    return [Phase.FRAME_BEGIN, *substeps, Phase.FRAME_END] * groups


# ----- frame freshness: the invariant the direct loop exists to preserve -----


@pytest.mark.parametrize("render_interval", [1, 2, 3, 4])
def test_frame_end_egresses_the_frame_rendered_for_that_control_step(
    monkeypatch: pytest.MonkeyPatch, render_interval: int
) -> None:
    """Each egressed frame is the one drawn for the state written on its own control step.

    Swept over every ``render_interval`` at or below the decimation, the range IsaacSim supports
    (``isaacsim.py`` warns below it and asks for equality): the last draw of a control step always
    happens after that step's PRE_STEP write, so the frame is never one step stale. The regression
    this guards: FRAME_END used to fire on the FIRST substep of the control step while the draw
    lands on the last, so every frame egressed was one control step behind and an episode
    boundary's first frame showed the previous episode.
    """
    sim = _FakeSim(frames=4, render_interval=render_interval)

    _run(sim, monkeypatch)

    assert sim.captured == [0, 1, 2, 3]


def test_frame_freshness_holds_without_decimation(monkeypatch: pytest.MonkeyPatch) -> None:
    """With one physics step per control step the frame opens and closes on the same substep."""
    sim = _FakeSim(frames=3, render_interval=1, decimation=1)

    _run(sim, monkeypatch)

    assert sim.captured == [0, 1, 2]
    assert sim.phases == _expected_phases(groups=3, decimation=1)


def test_every_control_step_egresses_exactly_one_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """No frame is dropped or emitted twice: one FRAME_BEGIN and one FRAME_END per control step."""
    sim = _FakeSim(frames=5)

    _run(sim, monkeypatch)

    assert len(sim.captured) == 5
    assert sim.control_steps == 5
    assert sim.phases.count(Phase.FRAME_BEGIN) == 5
    assert sim.phases.count(Phase.FRAME_END) == 5
    assert sim.substeps == 5 * DECIMATION


def test_direct_loop_phase_schedule_matches_base_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """PRE_STEP/POST_STEP bracket every substep inside one FRAME_BEGIN..FRAME_END frame.

    Asserted over the whole recorded sequence, so a stray or misplaced emission anywhere fails.
    """
    sim = _FakeSim(frames=3)

    _run(sim, monkeypatch)

    assert sim.phases == _expected_phases(groups=3)


def test_a_loop_that_never_egresses_fails_instead_of_hanging(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fake's own guard: with FRAME_END never reached, the run raises rather than spinning.

    This is what keeps the tests above honest under mutation — a regression that silences egress, or
    removes the shutdown check, cannot stall the suite into a timeout that reads as an error.
    """
    sim = _FakeSim(frames=3)
    sim.egress_hook.remove()

    with pytest.raises(_LoopOverranError, match="without reaching 3 frame"):
        _run(sim, monkeypatch)


# ----- the render cadence a coherent frame needs -----


def test_a_beating_render_cadence_misaligns_state_and_image(monkeypatch: pytest.MonkeyPatch) -> None:
    """A render_interval that does not divide the decimation draws before the frame's last substep.

    Interval 3 against decimation 4 draws on substeps 3, 6, 9, 12 while frames close on 4, 8, 12, so
    the skew between image and state changes frame to frame. Pinned here as the reason ``IsaacSim``
    rejects this pairing at construction when cameras are configured, rather than letting it ship
    frames a consumer cannot trust.
    """
    sim = _FakeSim(frames=3, render_interval=3)

    _run(sim, monkeypatch)

    assert sim.alignment == [(4, 3), (8, 6), (12, 12)]


# ----- the loop's other per-backend and viewer branches -----


@pytest.mark.parametrize("backend", [SimulatorType.ISAACSIM, SimulatorType.ISAACGYM])
def test_tensor_backends_refresh_every_substep_and_again_before_egress(
    monkeypatch: pytest.MonkeyPatch, backend: SimulatorType
) -> None:
    """IsaacGym/IsaacSim resync before every substep, plus once more before each frame egresses.

    The count is the observable here, not an implementation detail: it is what distinguishes a refresh
    per substep (PRE_STEP writers see current state) from one per control step, which the alignment
    assertion above cannot see because it only reads the last refresh before FRAME_END.
    """
    sim = _FakeSim(frames=3)

    _run(sim, monkeypatch, backend=backend)

    assert sim.refreshes == sim.substeps + len(sim.captured)


def test_mujoco_never_refreshes_tensors(monkeypatch: pytest.MonkeyPatch) -> None:
    """MuJoCo runs no env/task needing those tensors, so the direct loop leaves them alone."""
    sim = _FakeSim(frames=3)

    _run(sim, monkeypatch, backend=SimulatorType.MUJOCO)

    assert sim.refreshes == 0


@pytest.mark.parametrize("render_interval", [1, 2, DECIMATION])
def test_state_tensors_and_image_describe_the_same_substep(
    monkeypatch: pytest.MonkeyPatch, render_interval: int
) -> None:
    """At egress the state tensors describe the substep the image was drawn in, not the one before.

    A FRAME_END consumer may publish both — camera frames next to odometry, simulator state, or IMU —
    so a skew between them puts a stale pose beside a current image. Swept over the render intervals
    that divide the decimation, where the frame's last draw coincides with its last substep.
    """
    sim = _FakeSim(frames=3, render_interval=render_interval)

    _run(sim, monkeypatch)

    # Frame g's last draw and its pre-egress refresh both land on substep (g + 1) * DECIMATION.
    assert sim.alignment == [(4, 4), (8, 8), (12, 12)]


def test_shutdown_asked_for_inside_a_frame_still_egresses_that_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """request_shutdown stops the run after the current frame, so that frame keeps its egress.

    A plugin asking from PRE_STEP or POST_STEP must not lose the frame's rendering, recording, and
    telemetry to a mid-frame exit.
    """
    sim = _FakeSim(frames=99)  # the ask below ends the run, not the frame budget

    def ask_mid_frame() -> None:
        if sim.substeps == DECIMATION + 1:  # first substep of the second frame
            sim.request_shutdown("test")

    sim.hooks.add(Phase.POST_STEP, ask_mid_frame, name="test.ask")

    _run(sim, monkeypatch)

    assert sim.captured == [0, 1]
    assert sim.substeps == 2 * DECIMATION


def test_a_frame_that_opens_always_closes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shutdown asked for from FRAME_BEGIN still closes that frame: the phases stay paired.

    Plugins pair per-frame work across FRAME_BEGIN and FRAME_END, so a frame the loop opened must
    reach FRAME_END whatever a hook asks for meanwhile. A plugin that does not want its last frame
    egressed asks from FRAME_END instead, which is how MotionPlaybackPlugin ends a finite run.
    """
    sim = _FakeSim(frames=99)

    def ask_on_frame_begin() -> None:
        if sim.control_steps == 3:  # the third frame asks to stop while being opened
            sim.request_shutdown("test")

    sim.hooks.add(Phase.FRAME_BEGIN, ask_on_frame_begin, name="test.ask")

    _run(sim, monkeypatch)

    assert sim.captured == [0, 1, 2]
    assert sim.phases == _expected_phases(groups=3)
    assert sim.substeps == 3 * DECIMATION


def test_viewer_renders_on_its_own_schedule_when_not_headless(monkeypatch: pytest.MonkeyPatch) -> None:
    """A windowed run draws the viewport every ``viewer_dt``, independent of the control rate.

    ``viewer_dt=0.02`` against a 0.005 s physics step is every 4th step; ``render`` records the
    substep counter as it stands after that step, so 1/5/9 are the run's 1st, 5th and 9th substeps.
    The viewport is therefore drawn on the FIRST substep of each control step, unlike the camera
    egress on FRAME_END — the two schedules are deliberately independent.
    """
    sim = _FakeSim(frames=3)

    _run(sim, monkeypatch, headless=False, viewer_dt=DECIMATION / FPS)

    assert sim.viewer_renders == [1, 5, 9]


def test_headless_never_renders_the_viewport(monkeypatch: pytest.MonkeyPatch) -> None:
    """Headless leaves the viewport alone even at a viewer_dt that would draw every step."""
    sim = _FakeSim(frames=3)

    _run(sim, monkeypatch, headless=True, viewer_dt=1.0 / FPS)

    assert sim.viewer_renders == []


@pytest.mark.parametrize(
    ("viewer_dt", "expected"),
    [(1.0 / FPS, 1), (DECIMATION / FPS, DECIMATION), (0.1, 20), (0.0, 1), (1e-9, 1)],
)
def test_calculate_viewer_steps_floors_at_one_physics_step(viewer_dt: float, expected: int) -> None:
    """viewer_dt resolves to whole physics steps, and a sub-step dt still draws once per step."""
    config = _run_sim_config(decimation=DECIMATION, viewer_dt=viewer_dt)
    direct = DirectSimulation(config, SimpleNamespace(sim=None), device="cpu", simulation_app=None)

    assert direct._calculate_viewer_steps() == expected


# ----- integration: the real playback plugin driven by the real loop -----


CLIP_FRAMES = 5
CLIP_JOINT0 = [10.0 + i for i in range(CLIP_FRAMES)]
"""``joint0`` per clip frame — the value this section follows from the clip through to egress."""


@pytest.fixture
def local_path_loaders(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register the production local path loaders directly instead of through entry points.

    Path resolution is not what this section constrains, and entry-point discovery depends on install
    metadata rather than on the source under test, so a checkout whose metadata predates the loader
    entry points would fail here for a reason unrelated to the loop. ``LocalPathLoader`` itself still
    does the resolving.
    """
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
    """``_FakeSim`` plus the surface MotionPlaybackPlugin writes through.

    Frame identity is the clip's own ``joint0``: the plugin writes it into ``dof_pos`` on PRE_STEP,
    the render gate latches whatever is there, and FRAME_END egresses that latch — so ``captured`` is
    directly comparable to :data:`CLIP_JOINT0` with nothing of the fake's own invention in between.
    Shutdown comes only from the plugin, which is what makes an extra terminal frame observable.
    """

    def __init__(self, *, decimation: int, budget_frames: int) -> None:
        super().__init__(frames=budget_frames, render_interval=decimation, decimation=decimation)
        self.num_envs = 1
        self.num_dof = 2
        self.sim_device = "cpu"
        self.dof_pos = torch.zeros(1, self.num_dof)
        self.dof_vel = torch.zeros(1, self.num_dof)
        self.robot_root_states = torch.zeros(1, 13)
        # Derived from the base fake's own rates, so the plugin's view of the sim and the driving
        # loop's (built by _run_sim_config from the same fields) cannot drift apart.
        self.simulator_config = SimpleNamespace(
            sim=SimpleNamespace(
                fps=FPS, control_decimation_steps=self.decimation, render_interval_steps=self.render_interval
            )
        )

    @property
    def dof_state(self) -> torch.Tensor:
        return torch.cat([self.dof_pos[..., None], self.dof_vel[..., None]], dim=-1)

    @property
    def shutdown_requested(self) -> bool:
        return self._shutdown

    def request_shutdown(self, reason: str = "") -> None:
        self._shutdown = True

    def set_actor_root_state_tensor_robots(self, env_ids: torch.Tensor, root_states: torch.Tensor) -> None:
        pass

    def set_dof_state_tensor_robots(self, env_ids: torch.Tensor, dof_states: torch.Tensor) -> None:
        pass

    def set_actor_states(self, names: list[str], env_ids: torch.Tensor, states: torch.Tensor) -> None:
        pass

    def _scene_state(self) -> object:
        """A draw captures the joint the plugin pinned this substep."""
        return round(float(self.dof_pos[0, 0]), 4)


@pytest.mark.parametrize("clips", [1, 2])
@pytest.mark.parametrize("decimation", [1, 2, DECIMATION])
def test_playback_egresses_every_clip_frame_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, local_path_loaders: None, decimation: int, clips: int
) -> None:
    """A finite playlist egresses each clip frame once: no stale first frame, no doubled last one.

    The real :class:`MotionPlaybackPlugin` runs through the real loop at the clip's own rate, so one
    control step is one clip frame. The plugin ends the run from its final frame's FRAME_END, so that
    frame is the run's last and the loop opens no further one to repeat it — at any decimation.
    """
    sim = _PlaybackSim(decimation=decimation, budget_frames=(CLIP_FRAMES + 2) * clips)
    cfg = MotionPlaybackPluginConfig(
        motion_files=[_clip(tmp_path, f"clip{index}.npy") for index in range(clips)],
        fps=FPS / decimation,  # one clip frame per control step
        n_run=1,
    )
    cfg.get_cls()(cfg, sim)

    _run(sim, monkeypatch)

    assert sim.captured == CLIP_JOINT0 * clips
