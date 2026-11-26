"""Native CUDA graph and stream regressions for solved contact measurements."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast
from weakref import ref

import numpy as np
import pytest

torch = pytest.importorskip("torch")
mujoco = pytest.importorskip("mujoco")
wp = pytest.importorskip("warp")
pytest.importorskip("mujoco_warp")
pytestmark = pytest.mark.mujoco_warp

if not torch.cuda.is_available():
    pytest.skip("Native Warp contact stream tests require CUDA", allow_module_level=True)

from holosoma.config_types.full_sim import FullSimConfig  # noqa: E402
from holosoma.config_types.sensor import ContactSensorConfig  # noqa: E402
from holosoma.config_values.simulator import mujoco as mujoco_config  # noqa: E402
from holosoma.hooks import Phase  # noqa: E402
from holosoma.simulator.mujoco.backends.warp_backend import WarpBackend  # noqa: E402
from holosoma.simulator.mujoco.contact_warp import WarpContactReader  # noqa: E402
from holosoma.simulator.shared.sensor_manager import SensorManager  # noqa: E402


def test_publication_moves_cpu_backend_results_to_the_simulator_device() -> None:
    manager = SensorManager("cuda:0", 50.0, physics_hz=50.0)
    record = manager.register_contact("cpu_result", ContactSensorConfig(), lambda: torch.ones(1, 1, 1, 3))
    manager.contact_step_completed()
    for output in record.buffers.values():
        assert output.device == torch.device("cuda:0")
    assert record.buffers["forces"].tolist() == [[[1, 1, 1]]]


def _backend(xml: str) -> WarpBackend:
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    config = cast(
        "FullSimConfig",
        SimpleNamespace(training=SimpleNamespace(num_envs=2), simulator=mujoco_config.config),
    )
    backend = WarpBackend(model, data, config, "cuda:0")
    backend.initialize_state(model, data)
    return backend


def test_solve_and_fk_wait_for_delayed_nondefault_torch_inputs() -> None:
    backend = _backend(
        """
        <mujoco>
          <option gravity="0 0 0" timestep=".001" integrator="Euler"/>
          <worldbody>
            <body name="cart">
              <joint name="slide" type="slide" axis="1 0 0"/>
              <geom type="sphere" size=".1" mass="2"/>
            </body>
          </worldbody>
          <actuator><motor joint="slide" gear="1"/></actuator>
        </mujoco>
        """
    )
    consumer = torch.cuda.Stream(device=backend.device)
    other_warp_stream = wp.Stream(backend.mjw_device)
    try:
        # Instantiate the physics graph before the delayed-write regression interval.
        backend.step()
        torch.cuda.synchronize(backend.device)
        with torch.cuda.stream(consumer), wp.ScopedStream(other_warp_stream):
            # The old solve launched before this delayed control write on an independent stream.
            torch.cuda._sleep(20_000_000)
            backend.ctrl_t.fill_(4)
            backend.step()
            velocity = backend.qvel_t.clone()
            torch.cuda._sleep(20_000_000)
            backend.qpos_t[:, 0] = torch.tensor([0.4, -0.3], device=backend.device)
            backend.forward_kinematics()
            position = backend.xpos_t[:, backend.model.body("cart").id].clone()
        consumer.synchronize()
        # The native CPU solve gives an independent oracle for the one-step actuator response.
        data = mujoco.MjData(backend.model)
        data.ctrl[:] = 4
        mujoco.mj_step(backend.model, data)
        expected = torch.as_tensor(data.qvel, device=backend.device, dtype=velocity.dtype).expand_as(velocity)
        torch.testing.assert_close(velocity, expected, rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(
            position,
            torch.tensor([[0.4, 0, 0], [-0.3, 0, 0]], device=backend.device),
            rtol=0,
            atol=1e-7,
        )
    finally:
        backend.close()


def test_simulator_close_releases_queries_with_readers_and_samples_retained() -> None:
    from tests.simulators.contact_sensor_assert import _build

    sim = _build("mjwarp", 2)
    backend = sim.backend
    readers = [record.capture for record in sim.sensor_manager._contacts.values()]
    resources = [
        ref(getattr(reader, attribute))
        for reader in readers
        for attribute in ("graph", "sources", "counterparts", "output", "tensor")
    ]
    query_bytes = sum(
        getattr(reader, attribute).capacity for reader in readers for attribute in ("sources", "counterparts", "output")
    )
    consumer = torch.cuda.Stream(device=backend.device)
    completed = torch.cuda.Event()
    try:
        sim.simulate_at_each_physics_step()
        expected = sim.get_contact_data("robot_ground").clone()
        torch.cuda.synchronize(backend.device)
        live_before = wp.get_mempool_used_mem_current(backend.mjw_device)
        with torch.cuda.stream(consumer):
            # A queued clone still reads the reader's scratch buffer when close starts.
            torch.cuda._sleep(100_000_000)
            sample = sim.sensor_manager.get_sensor("robot_ground").capture().clone()
            completed.record()
        assert not completed.query()

        sim.hooks.emit(Phase.CLOSE)
        assert completed.query()
        assert all(reader.backend is None for reader in readers)
        assert all(resource() is None for resource in resources)
        live_after = wp.get_mempool_used_mem_current(backend.mjw_device)
        assert live_before - live_after >= query_bytes
        sim.hooks.emit(Phase.CLOSE)
        for reader in readers:
            reader.close()
        assert wp.get_mempool_used_mem_current(backend.mjw_device) == live_after
        torch.testing.assert_close(sample[:, :, 0], expected)
        torch.testing.assert_close(sim.get_contact_data("robot_ground"), expected)
        print(f"Contact query bytes: {query_bytes}; Warp live pool bytes: {live_before} -> {live_after}")
        # CUDA allocators can retain free blocks. Reserved memory is not a count of live allocations.
        print(f"Torch allocated/reserved bytes: {torch.cuda.memory_allocated()} / {torch.cuda.memory_reserved()}")
    finally:
        sim.hooks.emit(Phase.CLOSE)


def test_reader_close_releases_selection_but_preserves_an_external_torch_alias() -> None:
    backend = _backend(
        '<mujoco><worldbody><body><freejoint/><geom type="sphere" size=".1"/></body></worldbody></mujoco>'
    )
    mask = np.ones((1, backend.model.ngeom), dtype=np.int32)
    reader = WarpContactReader(backend, mask, mask, "normal")
    graph, sources, counterparts = ref(reader.graph), ref(reader.sources), ref(reader.counterparts)
    output, tensor = ref(reader.output), ref(reader.tensor)
    torch.cuda.synchronize(backend.device)
    live_before = wp.get_mempool_used_mem_current(backend.mjw_device)
    selection_bytes = reader.sources.capacity + reader.counterparts.capacity
    output_bytes = reader.output.capacity
    consumer = torch.cuda.Stream(device=backend.device)
    completed = torch.cuda.Event()
    try:
        with torch.cuda.stream(consumer):
            torch.cuda._sleep(100_000_000)
            alias = reader()
            sample = alias.clone()
            completed.record()
        assert not completed.query()
        reader.close()
        assert completed.query()
        assert graph() is None and sources() is None and counterparts() is None
        assert reader.backend is None
        assert output() is not None and tensor() is alias
        assert alias.count_nonzero() == 0
        assert sample.count_nonzero() == 0
        live_with_alias = wp.get_mempool_used_mem_current(backend.mjw_device)
        assert live_before - live_with_alias >= selection_bytes
        reader.close()
        assert wp.get_mempool_used_mem_current(backend.mjw_device) == live_with_alias
        del alias
        wp.synchronize_device(backend.mjw_device)
        assert output() is None and tensor() is None
        assert live_with_alias - wp.get_mempool_used_mem_current(backend.mjw_device) >= output_bytes
    finally:
        reader.close()
        backend.close()


def test_warp_without_configured_contacts_keeps_legacy_history_without_readers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.simulators.contact_sensor_assert import _assert_legacy_only, _build

    def unexpected_reader(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Unconfigured contacts must not allocate a Warp query graph or its arrays.")

    monkeypatch.setattr(WarpContactReader, "__init__", unexpected_reader)
    sim = _build("mjwarp", 2, configured_contacts=False)
    try:
        _assert_legacy_only(sim)
    finally:
        sim.hooks.emit(Phase.CLOSE)


def test_captured_queries_keep_empty_opposite_and_delayed_previous_force_samples() -> None:
    backend = _backend(
        """
        <mujoco>
          <option gravity="0 0 0" timestep=".001" integrator="Euler"/>
          <worldbody>
            <body name="source">
              <freejoint/><geom name="source" type="sphere" size=".1" mass="1"/>
            </body>
            <body name="counterpart" pos="1 0 0">
              <freejoint/><geom name="counterpart" type="sphere" size=".1" mass="1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    masks = np.eye(2, dtype=np.int32)
    reader = WarpContactReader(backend, masks[:1], masks[1:], "normal")
    reverse = WarpContactReader(backend, masks[1:], masks[:1], "total")
    graph = reader.graph
    output_ptr = reader.output.ptr
    source_body = backend.model.body("source").id
    joint = backend.model.body_jntadr[backend.model.body("counterpart").id]
    counterpart_qpos = backend.model.jnt_qposadr[joint]
    with wp.ScopedStream(backend.wp_stream):
        poses = []
        for positions in ((1.0, 1.0), (0.16, -0.16), (-0.16, 0.16), (1.0, 1.0)):
            qpos = np.tile(backend.model.qpos0, (backend.num_envs, 1)).astype(np.float32)
            qpos[:, counterpart_qpos] = positions
            poses.append(wp.array(qpos, dtype=wp.float32))
    consumers = [torch.cuda.Stream(device=backend.device), torch.cuda.Stream(device=backend.device)]
    other_warp_stream = wp.Stream(backend.mjw_device)
    samples = []
    reactions = []
    body_samples = []
    try:
        with wp.ScopedStream(other_warp_stream):
            for index, pose in enumerate(poses):
                consumer = consumers[index % len(consumers)]
                with torch.cuda.stream(consumer):
                    # Keep input writes on the solver stream to isolate the prior-force read hazard.
                    with wp.ScopedStream(backend.wp_stream):
                        wp.copy(backend.mjw_data.qpos, pose)
                        backend.mjw_data.qvel.zero_()
                    backend.step()
                    matrix = reader()
                    reaction = reverse()
                    body_forces = backend.compute_contact_forces()
                    # Switch callers after a delayed clone. Only the tracked prior consumer protects it.
                    torch.cuda._sleep(20_000_000)
                    samples.append(matrix.clone())
                    reactions.append(reaction.clone())
                    body_samples.append(body_forces.clone())
        for consumer in consumers:
            consumer.synchronize()
        assert reader.graph is graph
        assert reader.output.ptr == output_ptr
        assert samples[0].count_nonzero() == 0
        assert samples[3].count_nonzero() == 0
        assert samples[1][0, 0, 0, 0] < -1
        assert samples[1][1, 0, 0, 0] > 1
        assert samples[2][0, 0, 0, 0] > 1
        assert samples[2][1, 0, 0, 0] < -1
        for sample, reaction, body in zip(samples, reactions, body_samples, strict=True):
            torch.testing.assert_close(reaction, -sample, rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(body[:, source_body], sample[:, 0, 0], rtol=1e-5, atol=1e-5)
    finally:
        reader.close()
        reverse.close()
        backend.close()
