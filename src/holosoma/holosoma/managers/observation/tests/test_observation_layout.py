"""``ObservationManager.get_observation_layout`` describes the exact flat policy input (no sim).

Pins:
- terms appear in concatenation order (groups as given, terms sorted within a group) with
  per-frame ``dim``, ``history_length``, ``scale``, ``clip`` and ``params``;
- slicing the concatenated observation with the layout recovers each term, oldest frame first;
- the layout survives the ONNX metadata round trip and matches the exported input width.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import onnx
import pytest

from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg
from holosoma.managers.observation.manager import ObservationManager
from holosoma.utils.inference_helpers import attach_onnx_metadata, export_policy_as_onnx
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim

_TERMS = "holosoma.managers.observation.terms.locomotion"
_NUM_ENVS = 2


def _make_manager() -> tuple[ObservationManager, SimpleNamespace]:
    env = SimpleNamespace(
        num_envs=_NUM_ENVS,
        device="cpu",
        simulator=SimpleNamespace(dof_pos=torch.zeros(_NUM_ENVS, 3), dof_vel=torch.zeros(_NUM_ENVS, 3)),
        default_dof_pos=torch.zeros(1, 3),
        action_manager=SimpleNamespace(action=torch.zeros(_NUM_ENVS, 2)),
    )
    cfg = ObservationManagerCfg(
        groups={
            "actor_obs": ObsGroupCfg(
                history_length=2,
                terms={
                    "dof_vel": ObsTermCfg(func=f"{_TERMS}:dof_vel", scale=0.5),
                    "dof_pos": ObsTermCfg(func=f"{_TERMS}:dof_pos"),
                    "actions": ObsTermCfg(func=f"{_TERMS}:actions", clip=(-1.0, 1.0)),
                },
            ),
            "gravity": ObsGroupCfg(
                terms={"gravity": ObsTermCfg(func=f"{_TERMS}:gravity_vector", params={"up_axis_idx": 2})},
            ),
        }
    )
    return ObservationManager(cfg, env, device="cpu"), env


def test_layout_lists_terms_in_concatenation_order() -> None:
    mgr, _ = _make_manager()
    layout = mgr.get_observation_layout(["actor_obs", "gravity"])

    assert [(t["group"], t["name"], t["dim"], t["history_length"]) for t in layout] == [
        ("actor_obs", "actions", 2, 2),
        ("actor_obs", "dof_pos", 3, 2),
        ("actor_obs", "dof_vel", 3, 2),
        ("gravity", "gravity", 3, 1),
    ]
    assert layout[0]["clip"] == (-1.0, 1.0)
    assert layout[2]["scale"] == 0.5
    assert layout[3]["params"] == {"up_axis_idx": 2}
    assert layout[1]["func"] == f"{_TERMS}:dof_pos"


def test_layout_slices_the_concatenated_observation() -> None:
    mgr, env = _make_manager()
    groups = ["actor_obs", "gravity"]
    first, second = torch.full((_NUM_ENVS, 3), 1.0), torch.full((_NUM_ENVS, 3), 2.0)
    env.simulator.dof_pos = first
    mgr.compute()
    env.simulator.dof_pos = second
    flat = torch.cat([mgr.compute()[g] for g in groups], dim=-1)

    layout = mgr.get_observation_layout(groups)
    assert flat.shape[1] == sum(t["dim"] * t["history_length"] for t in layout)

    start = 0
    for term in layout:
        width = term["dim"] * term["history_length"]
        if term["name"] == "dof_pos":
            assert torch.equal(flat[:, start : start + width], torch.cat([first, second], dim=-1))
        start += width


def test_layout_round_trips_through_onnx_metadata() -> None:
    mgr, _ = _make_manager()
    layout = mgr.get_observation_layout(["actor_obs", "gravity"])
    obs_dim = sum(t["dim"] * t["history_length"] for t in layout)

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = str(Path(tmpdir) / "policy.onnx")
        export_policy_as_onnx(torch.nn.Linear(obs_dim, 4), onnx_path, {"actor_obs": torch.zeros(1, obs_dim)})
        attach_onnx_metadata(onnx_path, {"observation_layout": layout, "observation_clip": mgr.cfg.clip_observations})
        model = onnx.load(onnx_path)

    metadata = {entry.key: json.loads(entry.value) for entry in model.metadata_props}
    assert metadata["observation_layout"] == json.loads(json.dumps(layout))
    assert metadata["observation_clip"] == 100.0
    assert model.graph.input[0].type.tensor_type.shape.dim[1].dim_value == obs_dim
