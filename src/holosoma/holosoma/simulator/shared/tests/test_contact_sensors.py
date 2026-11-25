"""Contact publication, naming, read access, and reset behavior."""

from __future__ import annotations

import dataclasses

import pytest

from holosoma.config_types.full_sim import FullSimConfig
from holosoma.config_types.run_sim import RunSimConfig
from holosoma.config_types.sensor import CameraSensorConfig, ContactSensorConfig, ContactTargetConfig, SensorMountConfig
from holosoma.simulator.base_simulator.base_simulator import BaseSimulator
from holosoma.simulator.base_simulator.hooks import HookRegistry, Phase
from holosoma.simulator.shared.sensor_manager import SensorManager
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim


def test_contact_outputs_keep_grouped_opposing_forces() -> None:
    manager = SensorManager("cpu", 50.0, num_envs=2)
    config = ContactSensorConfig(
        counterparts=[ContactTargetConfig(target_kind="actor", target="a"), ContactTargetConfig(target_kind="terrain")]
    )
    frame = torch.tensor([[[[3.0, 0, 0], [-3.0, 0, 0]]], [[[0.5, 0, 0], [0, 0, 0]]]])
    record = manager.register_contact("grip", config, lambda: frame)
    manager.contact_step_completed()

    assert manager.names == ["grip"]
    assert manager.get_sensor("grip") is record
    assert record.source_names == ["robot"]
    assert record.counterpart_names == ["actor:a", "terrain"]
    assert set(record.buffers) == {"forces", "force_matrix"}
    assert record.buffers["forces"].tolist() == [[[0.0, 0, 0]], [[0.5, 0, 0]]]
    assert torch.equal(record.buffers["force_matrix"], frame)
    frame.zero_()
    assert record.buffers["force_matrix"][0, 0, 0, 0] == 3.0
    assert manager.last_contact_due == {"grip"}


def test_public_outputs_convert_backend_forces_to_float32() -> None:
    manager = SensorManager("cpu", 50.0, num_envs=3)
    config = ContactSensorConfig(
        counterparts=[ContactTargetConfig(target_kind="actor", target="a"), ContactTargetConfig(target_kind="terrain")]
    )
    frame = torch.tensor(
        [
            [[[3, 4, 0], [0, 0, 0]]],
            [[[4, 4, 0], [0, 0, 0]]],
            [[[4, 0, 0], [4, 0, 0]]],
        ],
        dtype=torch.float64,
    )
    manager.register_contact("converted", config, lambda: frame)
    sim = BaseSimulator.__new__(BaseSimulator)
    sim.sensor_manager = manager
    manager.contact_step_completed()

    torch.testing.assert_close(sim.get_contact_data("converted", "force_matrix"), frame.float())
    assert sim.get_contact_data("converted").tolist() == [[[3, 4, 0]], [[4, 4, 0]], [[8, 0, 0]]]
    assert sim.get_contact_data("converted").dtype == torch.float32
    frame.zero_()
    assert sim.get_contact_data("converted")[0, 0].tolist() == [3, 4, 0]


def test_multiple_contact_sensors_publish_at_independent_rates() -> None:
    manager = SensorManager("cpu", 50.0)
    sample = torch.ones(1, 1, 1, 3)
    fast = manager.register_contact("fast", ContactSensorConfig(), lambda: sample)
    slow = manager.register_contact("slow", ContactSensorConfig(update_decimation="25Hz"), lambda: sample)
    manager.contact_step_completed()
    assert manager.last_contact_due == {"fast"}
    assert slow.buffers["forces"].count_nonzero() == 0
    assert manager.frames_produced("slow") == 0
    sample.fill_(2)
    manager.contact_step_completed()
    assert manager.last_contact_due == {"fast", "slow"}
    old_slow = slow.buffers["forces"]
    sample.fill_(3)
    manager.contact_step_completed()
    assert manager.last_contact_due == {"fast"}
    assert fast.buffers["forces"][0, 0].tolist() == [3.0] * 3
    assert slow.buffers["forces"] is old_slow
    sample.fill_(4)
    manager.contact_step_completed()
    assert slow.buffers["forces"][0, 0].tolist() == [4.0] * 3
    assert manager.frames_produced("fast") == 4
    assert manager.frames_produced("slow") == 2


def test_reset_drops_stale_rows_and_device_copies_until_a_new_physics_sample() -> None:
    manager = SensorManager("cpu", 50.0, num_envs=3)
    record = manager.register_contact("contacts", ContactSensorConfig(), lambda: torch.ones(3, 1, 1, 3))
    manager.contact_step_completed()
    old_frame = record.buffers["forces"]
    old_copy = record.buffer_on("forces", "meta")
    manager.reset_contacts(torch.tensor([1]))
    assert record.buffers["forces"][:, 0, 0].tolist() == [1, 0, 1]
    assert record.buffers["force_matrix"][:, 0, 0, 0].tolist() == [1, 0, 1]
    assert old_frame[:, 0, 0].tolist() == [1, 1, 1]
    assert record.buffer_on("forces", "meta") is not old_copy
    sim = BaseSimulator.__new__(BaseSimulator)
    sim.sensor_manager = manager
    assert sim.get_contact_data("contacts")[1].count_nonzero() == 0
    assert record.buffers["forces"][:, 0, 0].tolist() == [1, 0, 1]
    manager.contact_step_completed()
    assert record.buffers["forces"][:, 0, 0].tolist() == [1, 1, 1]


def test_contact_and_camera_names_share_one_namespace() -> None:
    manager = SensorManager("cpu", 50.0)
    manager.register_contact("same", ContactSensorConfig(), lambda: torch.zeros(1, 1, 1, 3))
    with pytest.raises(ValueError, match="already registered"):
        manager.register_camera("same", CameraSensorConfig(mount=SensorMountConfig(target_kind="world")))
    with pytest.raises(RuntimeError, match="no runtime mount state"):
        manager.get_mount_pose("same")


def test_simulator_accessor_selects_environments_and_rejects_bad_names() -> None:
    manager = SensorManager("cpu", 50.0, num_envs=3)
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
    with pytest.raises(ValueError, match="no 'substep_forces' output"):
        sim.get_contact_data("contacts", "substep_forces")


@pytest.mark.parametrize("name", ["contact_sensor", "__holosoma_contact_custom"])
def test_simulator_reserves_only_the_existing_robot_contact_name(name: str) -> None:
    defaults = RunSimConfig()
    config = FullSimConfig(
        simulator=defaults.simulator.config,
        robot=defaults.robot,
        training=defaults.training,
        logger=defaults.logger,
        sensors={name: ContactSensorConfig()},
    )
    if name == "contact_sensor":
        with pytest.raises(ValueError, match=f"Sensor name '{name}' is reserved"):
            BaseSimulator(config, None, "cpu")  # type: ignore[arg-type]
    else:
        sim = BaseSimulator(config, None, "cpu")  # type: ignore[arg-type]
        assert name in sim.sensor_config


def test_bad_native_output_shape_fails_at_publication() -> None:
    manager = SensorManager("cpu", 50.0)
    manager.register_contact("bad", ContactSensorConfig(), lambda: torch.ones(1, 2, 3))
    with pytest.raises(RuntimeError, match="expected forces"):
        manager.contact_step_completed()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA device")
def test_publication_moves_cpu_backend_results_to_the_simulator_device() -> None:
    manager = SensorManager("cuda:0", 50.0)
    record = manager.register_contact("cpu_result", ContactSensorConfig(), lambda: torch.ones(1, 1, 1, 3))
    manager.contact_step_completed()
    for output in record.buffers.values():
        assert output.device == torch.device("cuda:0")
    assert record.buffers["forces"].tolist() == [[[1, 1, 1]]]


def test_completed_physics_intervals_align_with_control_frames() -> None:
    hooks = HookRegistry()
    manager = SensorManager("cpu", 50.0, physics_hz=200.0)
    sample = torch.zeros(1, 1, 1, 3)
    slow = manager.register_contact("slow", ContactSensorConfig(update_decimation="100Hz"), lambda: sample)
    control = manager.register_contact("control", ContactSensorConfig(update_decimation="50Hz"), lambda: sample)
    quarter = manager.register_contact("quarter", ContactSensorConfig(update_decimation="25Hz"), lambda: sample)
    hooks.emit(Phase.FRAME_BEGIN)
    for value in (1, 2, 3, 4):
        sample.fill_(value)
        manager.contact_step_completed()
        hooks.emit(Phase.POST_STEP)
        assert slow.buffers["forces"][0, 0, 0] == value // 2 * 2
        assert control.buffers["forces"][0, 0, 0] == (value if value == 4 else 0)
        assert quarter.buffers["forces"].count_nonzero() == 0
    assert control.buffers["forces"][0, 0, 0] == 4
    assert manager.frames_produced("slow") == 2
    assert manager.frames_produced("control") == 1
    assert manager.frames_produced("quarter") == 0

    # A new frame does not restart the physics clock.
    hooks.emit(Phase.FRAME_BEGIN)
    for value in (5, 6, 7, 8):
        sample.fill_(value)
        manager.contact_step_completed()
        assert control.buffers["forces"][0, 0, 0] == (value if value == 8 else 4)
        assert quarter.buffers["forces"][0, 0, 0] == (value if value == 8 else 0)
    assert slow.buffers["forces"][0, 0, 0] == 8
    assert manager.frames_produced("control") == 2
    assert manager.frames_produced("quarter") == 1


def test_post_step_consumer_can_keep_a_brief_contact_and_start_a_new_history() -> None:
    hooks = HookRegistry()
    manager = SensorManager("cpu", 50.0, physics_hz=200.0, num_envs=2)
    sample = torch.zeros(2, 1, 1, 3)
    manager.register_contact("impact", ContactSensorConfig(), lambda: sample)
    sim = BaseSimulator.__new__(BaseSimulator)
    sim.sensor_manager = manager
    history: list[torch.Tensor] = []
    hooks.add(Phase.FRAME_BEGIN, history.clear, name="impact.begin")
    hooks.add(Phase.POST_STEP, lambda: history.append(sim.get_contact_data("impact").clone()), name="impact.sample")
    hooks.emit(Phase.FRAME_BEGIN)
    for value in (0, 10, 0, 0):
        sample.fill_(value)
        manager.contact_step_completed()
        hooks.emit(Phase.POST_STEP)
    assert sim.get_contact_data("impact").count_nonzero() == 0
    assert torch.stack(history, dim=1)[:, :, 0, 0].tolist() == [[0, 10, 0, 0], [0, 10, 0, 0]]
    hooks.emit(Phase.FRAME_BEGIN)
    assert history == []


def test_reset_does_not_restart_the_shared_physics_cadence() -> None:
    manager = SensorManager("cpu", 50.0, physics_hz=200.0, num_envs=2)
    sample = torch.ones(2, 1, 1, 3)
    record = manager.register_contact("slow", ContactSensorConfig(update_decimation=4), lambda: sample)
    for _ in range(4):
        manager.contact_step_completed()
    manager.contact_step_completed()
    manager.reset_contacts(torch.tensor([0]))
    sample.fill_(2)
    for _ in range(2):
        manager.contact_step_completed()
        assert record.buffers["forces"][:, 0, 0].tolist() == [0, 1]
    manager.contact_step_completed()
    assert record.buffers["forces"][:, 0, 0].tolist() == [2, 2]
    assert manager.frames_produced("slow") == 2


@pytest.mark.parametrize("num_renders", [0, 1, 3])
def test_episode_reset_clears_contacts_before_render_without_advancing_physics(num_renders: int) -> None:
    defaults = RunSimConfig()
    simulator = dataclasses.replace(
        defaults.simulator.config,
        sim=dataclasses.replace(defaults.simulator.config.sim, num_rerenders_on_reset=num_renders),
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

    sim = ResetSimulator(config, None, "cpu")  # type: ignore[arg-type]
    sample = torch.ones(2, 1, 1, 3)
    captures: list[int] = []

    def capture() -> torch.Tensor:
        captures.append(len(captures))
        return sample

    record = sim.sensor_manager.register_contact("slow", ContactSensorConfig(update_decimation=4), capture)
    for _ in range(5):
        sim.sensor_manager.contact_step_completed()
    old_frame = sim.get_contact_data("slow")
    old_copy = record.buffer_on("forces", "meta")
    ids = torch.tensor([0])

    sim.hooks.emit(Phase.EPISODE_START, ids)

    assert len(rendered) == 1
    assert torch.equal(rendered[0][0], ids)
    assert rendered[0][1] == num_renders
    assert rendered[0][2][:, 0, 0].tolist() == [0, 1]
    assert old_frame[:, 0, 0].tolist() == [1, 1]
    assert record.buffer_on("forces", "meta") is not old_copy
    assert record.buffers["force_matrix"][:, 0, 0, 0].tolist() == [0, 1]
    assert sim.sensor_manager.last_contact_due == set()
    assert sim.sensor_manager.frames_produced("slow") == 1
    assert len(captures) == 1

    sample.fill_(2)
    for _ in range(2):
        sim.render_sensors(ids, 3)
        sim.hooks.emit(Phase.FRAME_END)
        sim.sensor_manager.contact_step_completed()
        assert sim.get_contact_data("slow")[:, 0, 0].tolist() == [0, 1]
        assert sim.sensor_manager.frames_produced("slow") == 1
        assert len(captures) == 1

    sim.sensor_manager.contact_step_completed()
    assert sim.get_contact_data("slow")[:, 0, 0].tolist() == [2, 2]
    assert sim.sensor_manager.frames_produced("slow") == 2
    assert len(captures) == 2
