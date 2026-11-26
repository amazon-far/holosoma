"""Contact publication, naming, read access, and reset behavior."""

from __future__ import annotations

import dataclasses

import pytest

from holosoma.config_types.full_sim import FullSimConfig
from holosoma.config_types.run_sim import RunSimConfig
from holosoma.config_types.sensor import (
    CameraSensorConfig,
    ContactSensorConfig,
    ContactTargetConfig,
    CustomLidarRayPatternConfig,
    LidarSensorConfig,
    SensorMountConfig,
)
from holosoma.hooks import HookRegistry, Phase
from holosoma.simulator.base_simulator.base_simulator import BaseSimulator
from holosoma.simulator.shared.sensor_manager import SensorManager
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_public_outputs_keep_grouped_float32_forces_and_owned_samples(dtype: torch.dtype) -> None:
    manager = SensorManager("cpu", 50.0, physics_hz=50.0, num_envs=2)
    config = ContactSensorConfig(
        counterparts=[ContactTargetConfig(target_kind="actor", target="a"), ContactTargetConfig(target_kind="terrain")]
    )
    frame = torch.tensor([[[[3.0, 0, 0], [-3.0, 0, 0]]], [[[0.5, 0, 0], [0, 0, 0]]]], dtype=dtype)
    record = manager.register_contact("grip", config, lambda: frame)
    sim = BaseSimulator.__new__(BaseSimulator)
    sim.sensor_manager = manager
    manager.contact_step_completed()

    assert set(record.buffers) == {"forces", "force_matrix"}
    assert sim.get_contact_data("grip").tolist() == [[[0.0, 0, 0]], [[0.5, 0, 0]]]
    torch.testing.assert_close(sim.get_contact_data("grip", "force_matrix"), frame.float())
    assert sim.get_contact_data("grip").dtype == torch.float32
    frame.zero_()
    assert sim.get_contact_data("grip", "force_matrix")[0, 0, 0, 0] == 3.0
    assert sim.get_contact_data("grip")[1, 0, 0] == 0.5


def test_contact_and_camera_names_share_one_namespace() -> None:
    manager = SensorManager("cpu", 50.0, physics_hz=50.0)
    manager.register_contact("same", ContactSensorConfig(), lambda: torch.zeros(1, 1, 1, 3))
    with pytest.raises(ValueError, match="already registered"):
        manager.register_camera("same", CameraSensorConfig(mount=SensorMountConfig(target_kind="world")))
    with pytest.raises(RuntimeError, match="no runtime mount state"):
        manager.get_mount_pose("same")


def test_simulator_accessor_selects_environments_and_rejects_bad_names() -> None:
    manager = SensorManager("cpu", 50.0, physics_hz=50.0, num_envs=3)
    manager.register_contact(
        "contacts", ContactSensorConfig(), lambda: torch.arange(9, dtype=torch.float32).reshape(3, 1, 1, 3)
    )
    manager.contact_step_completed()
    sim = BaseSimulator.__new__(BaseSimulator)
    sim.sensor_manager = manager
    assert sim.get_contact_data("contacts", env_ids=[2, 0]).tolist() == [[[6, 7, 8]], [[0, 1, 2]]]
    assert sim.get_contact_data("contacts", device="cpu") is manager.get_contact("contacts").buffers["forces"]
    with pytest.raises(NotImplementedError, match="no contact sensor"):
        sim.get_contact_data("absent")
    with pytest.raises(ValueError, match="no 'torques' output"):
        sim.get_contact_data("contacts", "torques")


def test_simulator_accepts_explicit_contact_sensor_name_without_creating_implicit_records() -> None:
    defaults = RunSimConfig()
    config = FullSimConfig(
        simulator=defaults.simulator.config,
        robot=defaults.robot,
        training=defaults.training,
        logger=defaults.logger,
        sensors={"contact_sensor": ContactSensorConfig()},
    )
    sim = BaseSimulator(config, None, "cpu", HookRegistry())  # type: ignore[arg-type]
    assert "contact_sensor" in sim.sensor_config
    assert sim.sensor_manager.contacts == []
    sim.sensor_manager.register_contact("contact_sensor", ContactSensorConfig(), lambda: torch.zeros(1, 1, 1, 3))
    assert sim.get_sensor_names() == ["contact_sensor"]
    assert sim.get_contact_data("contact_sensor").count_nonzero() == 0
    with pytest.raises(TypeError, match="name"):
        sim.get_contact_data()  # type: ignore[call-arg]
    config = dataclasses.replace(config, sensors={"__holosoma_contact_custom": ContactSensorConfig()})
    sim = BaseSimulator(config, None, "cpu", HookRegistry())  # type: ignore[arg-type]
    assert "__holosoma_contact_custom" in sim.sensor_config


def test_bad_native_output_shape_fails_at_publication() -> None:
    manager = SensorManager("cpu", 50.0, physics_hz=50.0)
    manager.register_contact("bad", ContactSensorConfig(), lambda: torch.ones(1, 2, 3))
    with pytest.raises(RuntimeError, match="expected forces"):
        manager.contact_step_completed()


@pytest.mark.parametrize(
    "optical",
    [
        CameraSensorConfig(mount=SensorMountConfig(target_kind="world")),
        LidarSensorConfig(
            mount=SensorMountConfig(target_kind="world"),
            pattern=CustomLidarRayPatternConfig(ray_directions=[[0, 0, -1]]),
        ),
    ],
)
def test_optical_sensor_name_cannot_replace_the_legacy_native_contact_view(
    optical: CameraSensorConfig | LidarSensorConfig,
) -> None:
    defaults = RunSimConfig()
    config = FullSimConfig(
        simulator=defaults.simulator.config,
        robot=defaults.robot,
        training=defaults.training,
        logger=defaults.logger,
        sensors={"contact_sensor": optical},
    )
    with pytest.raises(ValueError, match="native robot contact view"):
        BaseSimulator(config, None, "cpu", HookRegistry())  # type: ignore[arg-type]


def test_contacts_publish_at_independent_completed_physics_intervals() -> None:
    manager = SensorManager("cpu", 50.0, physics_hz=200.0)
    sample = torch.zeros(1, 1, 1, 3)
    fast = manager.register_contact("fast", ContactSensorConfig(), lambda: sample)
    slow = manager.register_contact("slow", ContactSensorConfig(update_decimation="100Hz"), lambda: sample)
    control = manager.register_contact("control", ContactSensorConfig(update_decimation="50Hz"), lambda: sample)
    quarter = manager.register_contact("quarter", ContactSensorConfig(update_decimation="25Hz"), lambda: sample)
    for value in range(1, 9):
        held = slow.buffers["forces"]
        sample.fill_(value)
        manager.contact_step_completed()
        assert fast.buffers["forces"][0, 0, 0] == value
        assert slow.buffers["forces"][0, 0, 0] == value // 2 * 2
        assert control.buffers["forces"][0, 0, 0] == value // 4 * 4
        assert quarter.buffers["forces"][0, 0, 0] == value // 8 * 8
        if value % 2:
            assert slow.buffers["forces"] is held
    assert manager.frames_produced("fast") == 8
    assert manager.frames_produced("slow") == 4
    assert manager.frames_produced("control") == 2
    assert manager.frames_produced("quarter") == 1


def test_zero_rerender_episode_reset_clears_contacts_without_restarting_physics_intervals() -> None:
    defaults = RunSimConfig()
    simulator = dataclasses.replace(
        defaults.simulator.config,
        sim=dataclasses.replace(defaults.simulator.config.sim, num_rerenders_on_reset=0),
    )
    config = FullSimConfig(
        simulator=simulator,
        robot=defaults.robot,
        training=dataclasses.replace(defaults.training, num_envs=2),
        logger=defaults.logger,
    )
    rendered: list[tuple[torch.Tensor | None, int | None, torch.Tensor]] = []

    class ResetSimulator(BaseSimulator):
        def render_sensors(self, env_ids: torch.Tensor | None = None, num_renders: int | None = None) -> None:
            rendered.append((env_ids, num_renders, self.get_contact_data("slow").clone()))

    sim = ResetSimulator(config, None, "cpu", HookRegistry())  # type: ignore[arg-type]
    sample = torch.ones(2, 1, 1, 3)
    captures: list[int] = []

    def capture() -> torch.Tensor:
        captures.append(len(captures))
        return sample

    record = sim.sensor_manager.register_contact("slow", ContactSensorConfig(update_decimation=4), capture)
    for _ in range(5):
        sim.hooks.emit(Phase.POST_STEP)
    old_frame = sim.get_contact_data("slow")
    old_copy = record.buffer_on("forces", "meta")
    ids = torch.tensor([0])

    sim.hooks.emit(Phase.EPISODE_START, ids)

    assert len(rendered) == 1
    assert torch.equal(rendered[0][0], ids)
    assert rendered[0][1] == 0
    assert rendered[0][2][:, 0, 0].tolist() == [0, 1]
    assert old_frame[:, 0, 0].tolist() == [1, 1]
    assert record.buffer_on("forces", "meta") is not old_copy
    assert record.buffers["force_matrix"][:, 0, 0, 0].tolist() == [0, 1]
    assert sim.sensor_manager.frames_produced("slow") == 1
    assert len(captures) == 1

    sample.fill_(2)
    for _ in range(2):
        sim.render_sensors(ids, 3)
        sim.hooks.emit(Phase.FRAME_END)
        sim.hooks.emit(Phase.POST_STEP)
        assert sim.get_contact_data("slow")[:, 0, 0].tolist() == [0, 1]
        assert sim.sensor_manager.frames_produced("slow") == 1
        assert len(captures) == 1

    sim.hooks.emit(Phase.POST_STEP)
    assert sim.get_contact_data("slow")[:, 0, 0].tolist() == [2, 2]
    assert sim.sensor_manager.frames_produced("slow") == 2
    assert len(captures) == 2


@pytest.mark.parametrize("start_in_playback", [False, True])
def test_post_step_publishes_contacts_before_consumers_and_skips_playback(start_in_playback: bool) -> None:
    from holosoma.simulator.shared.contact_substep import ContactSubstepRecorder

    defaults = RunSimConfig()
    config = FullSimConfig(
        simulator=dataclasses.replace(
            defaults.simulator.config,
            sim=dataclasses.replace(defaults.simulator.config.sim, kinematic_playback=start_in_playback),
        ),
        robot=defaults.robot,
        training=dataclasses.replace(defaults.training, num_envs=1),
        logger=defaults.logger,
    )
    native = torch.zeros(1, 1, 1, 3)

    class StepSimulator(BaseSimulator):
        def _advance_substep(self) -> None:
            if not self.kinematic_playback:
                native.add_(1)

        def render_sensors(self, env_ids: torch.Tensor | None = None, num_renders: int | None = None) -> None:
            pass

    sim = StepSimulator(config, None, "cpu", HookRegistry())  # type: ignore[arg-type]
    captures: list[str] = []

    def capture(name: str) -> torch.Tensor:
        captures.append(name)
        return native

    fast = sim.sensor_manager.register_contact("fast", ContactSensorConfig(), lambda: capture("fast"))
    slow = sim.sensor_manager.register_contact(
        "slow", ContactSensorConfig(update_decimation=2), lambda: capture("slow")
    )
    observed: list[float] = []
    sim.hooks.add(
        Phase.POST_STEP,
        lambda: observed.append(float(sim.get_contact_data("fast")[0, 0, 0])),
        name="contacts.consumer",
        after=sim.sensor_contact_hook,
    )
    recorder = ContactSubstepRecorder(sim.hooks, lambda: sim.get_contact_data("fast"), 1, 4, 1, "cpu")
    sim.hooks.emit(Phase.FRAME_BEGIN)
    if start_in_playback:
        sim.simulate_at_each_physics_step()
        assert fast.step_counter == slow.step_counter == 0
        assert captures == []
        assert observed == [0]
        observed.clear()
        sim.kinematic_playback = False
        sim.hooks.emit(Phase.FRAME_BEGIN)

    for value in (1, 2):
        old_frame = sim.get_contact_data("fast")
        sim._advance_substep()
        assert sim.get_contact_data("fast") is old_frame
        assert fast.step_counter == value - 1
        sim.hooks.emit(Phase.POST_STEP)
        assert sim.get_contact_data("fast")[0, 0, 0] == value
        assert fast.step_counter == value
    assert observed == [1, 2]
    assert captures == ["fast", "fast", "slow"]
    assert recorder.recorded_forces[0, :, 0, 0].tolist() == [1, 2]
    assert sim.get_contact_data("slow")[0, 0, 0] == 2

    held = sim.get_contact_data("fast")
    sim.kinematic_playback = True
    sim.simulate_at_each_physics_step()
    sim.hooks.emit(Phase.FRAME_END)
    assert sim.get_contact_data("fast") is held
    assert fast.step_counter == slow.step_counter == 2
    assert captures == ["fast", "fast", "slow"]

    sim.kinematic_playback = False
    for value in (3, 4):
        sim.simulate_at_each_physics_step()
        assert sim.get_contact_data("slow")[0, 0, 0] == (2 if value == 3 else 4)
    assert observed == [1, 2, 2, 3, 4]
