"""Minimal unit test for ExecuTorch export functionality."""

import json
import tempfile
from pathlib import Path

import pytest
import torch
from torch import nn

from holosoma.agents.modules.module_utils import setup_ppo_actor_module
from holosoma.config_types.algo import LayerConfig, ModuleConfig
from holosoma.utils.inference_helpers import export_policy_as_pte

runtime = pytest.importorskip("executorch.runtime")


class ActorWrapper(nn.Module):
    """Wrapper matching PPO's actor_onnx_wrapper pattern."""

    def __init__(self, actor: nn.Module):
        super().__init__()
        self.actor = actor

    def forward(self, actor_obs: torch.Tensor) -> torch.Tensor:
        return self.actor.act_inference({"actor_obs": actor_obs})


class TupleActor(nn.Module):
    """Returns (action, extra) like a raw FastSAC actor, which the exporter must refuse."""

    def __init__(self, wrapper: nn.Module):
        super().__init__()
        self.wrapper = wrapper

    def forward(self, actor_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        action = self.wrapper(actor_obs)
        return action, action * 2


def test_export_policy_as_pte():
    """Test ExecuTorch export, load, signature and output against eager PyTorch."""
    OBS_DIM, ACT_DIM = 10, 5

    module_config = ModuleConfig(
        type="MLP",
        input_dim=["actor_obs"],
        output_dim=[ACT_DIM],
        layer_config=LayerConfig(
            hidden_dims=[64],
            activation="ReLU",
            dropout_prob=0.0,
        ),
        min_noise_std=None,
        min_mean_noise_std=None,
    )
    actor = setup_ppo_actor_module(
        obs_dim_dict={"actor_obs": OBS_DIM},
        module_config=module_config,
        num_actions=ACT_DIM,
        init_noise_std=0.1,
        device="cpu",
        history_length={"actor_obs": 1},
    )
    wrapper = ActorWrapper(actor)
    wrapper.eval()
    metadata = {"dof_names": [f"joint_{i}" for i in range(ACT_DIM)], "kp": [20.0] * ACT_DIM}

    with tempfile.TemporaryDirectory() as tmpdir:
        pte_path = str(Path(tmpdir) / "test_policy.pte")
        export_policy_as_pte(
            wrapper=wrapper,
            pte_file_path=pte_path,
            example_obs_dict={"actor_obs": torch.zeros(1, OBS_DIM)},
            metadata=metadata,
        )

        program = runtime.Runtime.get().load_program(pte_path)
        signature = json.loads(program.load_method("signature").execute([])[0])
        assert signature["inputs"] == ["actor_obs"]
        assert signature["outputs"] == ["action"]
        assert {key: json.loads(value) for key, value in signature["metadata"].items()} == metadata

        obs = torch.randn(1, OBS_DIM)
        (action,) = program.load_method("forward").execute([obs])
        with torch.no_grad():
            expected = wrapper(obs)
        assert action.shape == (1, ACT_DIM)
        torch.testing.assert_close(action, expected, rtol=1e-5, atol=1e-5)

        with pytest.raises(ValueError, match="only the action tensor"):
            export_policy_as_pte(TupleActor(wrapper), pte_path, {"actor_obs": torch.zeros(1, OBS_DIM)})
        with pytest.raises(ValueError, match="Unknown ExecuTorch backend"):
            export_policy_as_pte(wrapper, pte_path, {"actor_obs": torch.zeros(1, OBS_DIM)}, backend="metal")
