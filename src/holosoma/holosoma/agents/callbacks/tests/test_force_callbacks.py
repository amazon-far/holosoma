"""Push and payload callbacks compose through the real shared force lifecycle."""

from types import SimpleNamespace

import numpy as np
import pytest

from holosoma.agents.callbacks.payload import EvalPayloadCallback
from holosoma.agents.callbacks.push import EvalPushCallback
from holosoma.agents.callbacks.recording import EvalRecordingCallback
from holosoma.config_types.eval_callback import PayloadConfig, PushConfig, RecordingConfig
from holosoma.config_values import simulator as simulator_values
from holosoma.simulator.base_simulator.base_simulator import BaseSimulator
from holosoma.simulator.base_simulator.hooks import HookRegistry, Phase
from holosoma.simulator.base_simulator.state import SimState
from holosoma.simulator.shared.external_wrench import ExternalWrenchAccumulator, WrenchTarget
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim


class _ForceSim(BaseSimulator):
    """Record native flushes while keeping the public hooks and accumulator."""

    def __init__(self) -> None:
        self.num_envs = 3
        self.sim_device = "cpu"
        self.simulator_config = simulator_values.isaacsim.config
        self.body_names = ["pelvis", "torso_link", "hand"]
        self.state = SimState("cpu")
        self._rigid_body_pos = torch.arange(27, dtype=torch.float32).reshape(3, 3, 3)
        self.hooks = HookRegistry()
        self._external_wrench = ExternalWrenchAccumulator(self)
        self.wrench = torch.zeros(3, 3, 6)

    def find_rigid_body_indice(self, body_name: str) -> int:
        return self.body_names.index(body_name) if body_name in self.body_names else -1

    def _write_external_wrench_native(self, targets: list[WrenchTarget]) -> None:
        for target in targets:
            assert target.actor_name == "robot"
            self.wrench[:, target.write_cols] = target.wrench[:, target.write_cols]

    def substep(self) -> torch.Tensor:
        self.hooks.emit(Phase.PRE_STEP)
        self.flush_external_wrench()
        return self.wrench.clone()


@pytest.mark.parametrize("teardown_order", [(0, 1), (1, 0)])
def test_push_and_payload_compose_and_remove_independently(
    monkeypatch: pytest.MonkeyPatch, teardown_order: tuple[int, int]
) -> None:
    """Both callbacks target env 1; neither can overwrite another callback or hook."""
    monkeypatch.setattr("holosoma.agents.callbacks.push.random.uniform", lambda low, _high: low)
    sim = _ForceSim()
    env = SimpleNamespace(simulator=sim, dt=0.1)
    training = SimpleNamespace(device="cpu", eval_callbacks=[], _unwrap_env=lambda: env)
    recording = EvalRecordingCallback(RecordingConfig(), training)
    push = EvalPushCallback(
        PushConfig(
            body_names="torso_link",
            env_id=1,
            force_range=(10.0, 10.0),
            duration_s=(0.2, 0.2),
            interval_s=(0.1, 0.1),
        ),
        training,
    )
    payload = EvalPayloadCallback(PayloadConfig(body_names="torso_link,hand", env_id=1, mass_kg=2.0), training)
    training.eval_callbacks = [recording, push, payload]
    sim.hooks.add(
        Phase.PRE_STEP,
        lambda: sim.apply_external_force(
            "robot",
            forces=torch.tensor([1.0, 2.0, 3.0]),
            body_names=["pelvis"],
            env_ids=[2],
        ),
    )
    push.on_pre_evaluate_policy()
    payload.on_pre_evaluate_policy()

    # One idle environment step starts a two-step pulse. Physics substeps must
    # repeat its force without accumulating it across substeps.
    push.on_post_eval_env_step({})
    payload.on_post_eval_env_step({})
    expected = torch.zeros(3, 3, 6)
    expected[1, 1, :3] = torch.tensor([10.0, 0.0, -9.81])
    expected[1, 2, :3] = torch.tensor([0.0, 0.0, -9.81])
    expected[2, 0, :3] = torch.tensor([1.0, 2.0, 3.0])
    for _ in range(3):
        torch.testing.assert_close(sim.substep(), expected)

    # Recording indexes the simulator's public body order without native IDs.
    np.testing.assert_array_equal(recording._buffers["push_body_pos_w"][-1], [12.0, 13.0, 14.0])
    np.testing.assert_array_equal(
        recording._buffers["payload_body_pos_w"][-1],
        [[12.0, 13.0, 14.0], [15.0, 16.0, 17.0]],
    )
    push.on_post_eval_env_step({})
    torch.testing.assert_close(sim.substep(), expected)
    push.on_post_eval_env_step({})
    expected[1, 1, 0] = 0.0
    torch.testing.assert_close(sim.substep(), expected)
    assert recording._buffers["push_active"][-1] == 0.0

    callbacks = [push, payload]
    for index in teardown_order:
        callbacks[index].on_post_evaluate_policy()
    expected[1] = 0.0
    torch.testing.assert_close(sim.substep(), expected)
    torch.testing.assert_close(sim.substep(), expected)

    # A second evaluation must register exactly one hook per callback.
    push.on_pre_evaluate_policy()
    payload.on_pre_evaluate_policy()
    push.on_post_eval_env_step({})
    expected[1, 1, :3] = torch.tensor([10.0, 0.0, -9.81])
    expected[1, 2, :3] = torch.tensor([0.0, 0.0, -9.81])
    torch.testing.assert_close(sim.substep(), expected)
    push.on_post_evaluate_policy()
    payload.on_post_evaluate_policy()
