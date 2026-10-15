"""The transport-free facade: get_config / get_default_policy / reset / step.

Driven through a stub ONNX of the configured width, so no checkpoint and no robot are needed.
The test that matters is the last one: the published snippet, run verbatim.
"""

from __future__ import annotations

import dataclasses as dc
import json
import sys
import types

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

# netifaces is imported at module scope by base.py for the booster DDS path and is not
# installable on every dev platform; the facade never reaches it.
sys.modules.setdefault("netifaces", types.SimpleNamespace(interfaces=list, ifaddresses=lambda _: {}, AF_INET=2))

import holosoma_inference as inference  # noqa: E402
from holosoma_inference.command import JointCommand  # noqa: E402
from holosoma_inference.facade import NullInterface  # noqa: E402

PRESET = "g1-29dof-loco"
NUM_DOFS = 29


def actor_obs_width(config) -> int:
    obs = config.observation
    history = obs.history_length_dict.get("actor_obs", 1)
    return sum(obs.obs_dims[term] for term in obs.obs_dict["actor_obs"]) * history


def write_stub(path, obs_dim, action_dim=NUM_DOFS, metadata=None):
    """A linear actor of the configured width, with the metadata a real export carries."""
    weight = numpy_helper.from_array(
        ((np.arange(obs_dim * action_dim, dtype=np.float32) % 5 - 2) * 1e-3).reshape(obs_dim, action_dim), "W"
    )
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["actor_obs", "W"], ["action"])],
        "actor",
        [helper.make_tensor_value_info("actor_obs", TensorProto.FLOAT, [1, obs_dim])],
        [helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, action_dim])],
        [weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10  # newer onnx writers emit an IR the installed runtime may reject
    defaults = {
        "kp": [100.0] * NUM_DOFS,
        "kd": [2.0] * NUM_DOFS,
        "action_scale": 0.25,
        "dof_names": [f"j{i}" for i in range(NUM_DOFS)],
    }
    for key, value in (defaults if metadata is None else metadata).items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, json.dumps(value)
    onnx.save(model, path)
    return str(path)


@pytest.fixture
def config(tmp_path):
    base = inference.get_config(PRESET)
    model = write_stub(tmp_path / "actor.onnx", actor_obs_width(base))
    return dc.replace(base, task=dc.replace(base.task, model_path=model))


@pytest.fixture
def policy(config):
    return inference.get_default_policy(config)


# ------------------------------------------------------------------------ get_config


def test_get_config_resolves_a_core_preset():
    assert actor_obs_width(inference.get_config(PRESET)) == 100


def test_get_config_lists_what_it_knows_when_the_name_is_wrong():
    with pytest.raises(KeyError, match="Known presets:.*g1-29dof-loco"):
        inference.get_config("not-a-preset")


def test_get_config_mentions_extensions_for_an_unknown_name():
    with pytest.raises(KeyError, match="needs that package installed"):
        inference.get_config("alpha-29dof-nonexistent")


# ---------------------------------------------------------------- get_default_policy


def test_builds_without_a_transport(policy):
    assert isinstance(policy.interface, NullInterface)
    assert policy.num_dofs == NUM_DOFS


def test_input_channels_are_neutralised(policy):
    assert policy.config.task.velocity_input == "injected"
    assert policy.config.task.state_input == "injected"
    assert policy._velocity_input.poll_velocity() is None
    assert policy._command_provider.poll_commands() == []


def test_the_null_transport_refuses_rather_than_inventing_a_pose(policy):
    with pytest.raises(RuntimeError, match="without a transport"):
        policy.interface.get_low_state()
    with pytest.raises(RuntimeError, match="without a transport"):
        policy.interface.send_low_command(np.zeros(NUM_DOFS))


def test_secondary_is_dropped(config):
    paired = dc.replace(config, secondary=config)
    assert inference.get_default_policy(paired).config.secondary is None


def test_a_preset_without_a_checkpoint_says_so(config):
    headless = dc.replace(config, task=dc.replace(config.task, model_path=""))
    with pytest.raises(ValueError, match="ships no model_path"):
        inference.get_default_policy(headless)


def test_model_path_override(config, tmp_path):
    other = write_stub(tmp_path / "other.onnx", actor_obs_width(config))
    assert inference.get_default_policy(config, model_path=other).config.task.model_path == other


# --------------------------------------------------------------------------- step


def test_step_returns_a_joint_command(policy):
    cmd = policy.step(np.zeros((1, 100)))
    assert isinstance(cmd, JointCommand)
    assert cmd.q.shape == (NUM_DOFS,)
    assert cmd.kp.shape == (NUM_DOFS,) and cmd.kd.shape == (NUM_DOFS,)


def test_a_zero_observation_commands_the_default_pose(policy):
    # The stub is linear, so zero observation -> zero action -> default pose + offsets.
    expected = np.asarray(policy.default_dof_angles).reshape(-1) + np.asarray(policy.joint_offsets).reshape(-1)
    np.testing.assert_allclose(policy.step(np.zeros((1, 100))).q, expected, atol=1e-12)


def test_step_applies_default_pose_and_action_scale(policy):
    obs = np.full((1, 100), 0.01, dtype=np.float32)
    action = policy.policy({"actor_obs": obs})
    expected = action[0] * policy.policy_action_scale + np.asarray(policy.default_dof_angles).reshape(-1)
    np.testing.assert_allclose(policy.step(obs).q - policy.joint_offsets, expected, rtol=1e-6)


def test_step_accepts_an_unbatched_observation(policy):
    np.testing.assert_allclose(policy.step(np.zeros(100)).q, policy.step(np.zeros((1, 100))).q)


@pytest.mark.parametrize("shape", [(1, 99), (1, 101), (2, 100), (1, 100, 1)])
def test_step_rejects_the_wrong_width(policy, shape):
    with pytest.raises(ValueError, match=r"shape \(1, 100\)"):
        policy.step(np.zeros(shape))


def test_step_rejects_a_non_finite_observation(policy):
    obs = np.zeros((1, 100))
    obs[0, 7] = np.nan
    with pytest.raises(ValueError, match="not finite"):
        policy.step(obs)


def test_step_is_pure_for_an_obs_in_policy(policy):
    """The caller owns the history here, so repeated steps must not drift."""
    obs = np.full((1, 100), 0.02, dtype=np.float32)
    first = policy.step(obs).q.copy()
    for _ in range(5):
        np.testing.assert_allclose(policy.step(obs).q, first)


def test_reset_clears_the_carried_state(policy):
    policy.step(np.full((1, 100), 0.02, dtype=np.float32))
    assert np.any(policy.last_policy_action != 0)
    policy.reset()
    np.testing.assert_allclose(policy.last_policy_action, 0)


def test_reset_does_not_change_an_obs_in_step(policy):
    obs = np.full((1, 100), 0.02, dtype=np.float32)
    before = policy.step(obs).q.copy()
    policy.reset()
    np.testing.assert_allclose(policy.step(obs).q, before)


def test_step_rejects_a_model_whose_action_width_is_wrong(config, tmp_path):
    narrow = write_stub(tmp_path / "narrow.onnx", actor_obs_width(config), action_dim=7)
    policy = inference.get_default_policy(config, model_path=narrow)
    with pytest.raises(ValueError, match="7 actions for 29 joints"):
        policy.step(np.zeros((1, 100)))


# ------------------------------------------------------------------- the snippet


def test_the_published_snippet(tmp_path):
    """The target API, verbatim apart from the preset name and the stub checkpoint.

    `alpha-29dof-loco` resolves identically once holosoma-ext-alpha-inference is installed:
    the extension aliases `loco_alpha_29dof = loco_g1_29dof`, so the width is the same 100.
    """
    config = inference.get_config("g1-29dof-loco")
    config = dc.replace(config, task=dc.replace(config.task, model_path=write_stub(tmp_path / "m.onnx", 100)))

    policy = inference.get_default_policy(config)
    policy.reset()  # no-op for loco
    obs = np.zeros((1, 100))
    cmd = policy.step(obs)

    assert cmd.q.shape == (29,)
    assert np.isfinite(cmd.q).all()


def test_the_alpha_preset_resolves_when_the_extension_is_installed():
    try:
        config = inference.get_config("alpha-29dof-loco")
    except KeyError:
        pytest.skip("holosoma-ext-alpha-inference is not installed")
    assert actor_obs_width(config) == 100
