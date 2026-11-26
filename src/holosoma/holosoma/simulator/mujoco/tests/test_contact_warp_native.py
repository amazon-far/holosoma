"""Native CUDA graph and stream regressions for solved contact measurements."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

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
        backend.close()
