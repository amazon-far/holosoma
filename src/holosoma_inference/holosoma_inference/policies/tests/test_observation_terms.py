"""Unit tests for BasePolicy.get_observation_terms schema generation.

The schema must mirror the per-group concat order in ``_update_obs_history``
(terms in ``obs_terms_sorted`` order, each spanning ``dim * history_len``), so a
consumer can slice ``/holosoma/observation`` by name. Exercised on a bare
instance — no ONNX/rclpy — since the method only reads the obs metadata dicts.
"""

from holosoma_inference.policies.base import BasePolicy


def _make_policy(obs_terms_sorted, obs_dims, history_length_dict, wandb_id=None):
    p = object.__new__(BasePolicy)
    p.obs_terms_sorted = obs_terms_sorted
    p.obs_dims = obs_dims
    p.history_length_dict = history_length_dict
    p.wandb_id = wandb_id
    return p


def test_slices_are_contiguous_and_match_concat_order():
    p = _make_policy(
        obs_terms_sorted={"actor_obs": ["base_ang_vel", "dof_pos", "dof_vel"]},
        obs_dims={"base_ang_vel": 3, "dof_pos": 29, "dof_vel": 29},
        history_length_dict={"actor_obs": 1},
        wandb_id="abc123",
    )
    schema = p.get_observation_terms()

    assert schema["schema_version"] == 1
    assert schema["dtype"] == "float32"
    assert schema["wandb_id"] == "abc123"

    terms = schema["groups"]["actor_obs"]["terms"]
    assert [t["name"] for t in terms] == ["base_ang_vel", "dof_pos", "dof_vel"]
    # start of each term == cumulative sum of prior total_dims
    assert [t["start"] for t in terms] == [0, 3, 32]
    assert [t["total_dim"] for t in terms] == [3, 29, 29]
    assert schema["groups"]["actor_obs"]["total_dim"] == 61


def test_history_multiplies_term_span():
    p = _make_policy(
        obs_terms_sorted={"actor_obs": ["dof_pos", "dof_vel"]},
        obs_dims={"dof_pos": 29, "dof_vel": 29},
        history_length_dict={"actor_obs": 4},
    )
    terms = p.get_observation_terms()["groups"]["actor_obs"]["terms"]
    assert [t["dim"] for t in terms] == [29, 29]
    assert [t["history_len"] for t in terms] == [4, 4]
    assert [t["total_dim"] for t in terms] == [116, 116]
    assert [t["start"] for t in terms] == [0, 116]


def test_wandb_id_defaults_to_none():
    p = _make_policy(
        obs_terms_sorted={"actor_obs": ["dof_pos"]},
        obs_dims={"dof_pos": 29},
        history_length_dict={},
    )
    schema = p.get_observation_terms()
    assert schema["wandb_id"] is None
    # missing group -> history_len defaults to 1
    assert schema["groups"]["actor_obs"]["total_dim"] == 29


def _wandb_policy(source_paths, onnx_wandb_run_path=None):
    p = object.__new__(BasePolicy)
    p._source_paths = source_paths
    p.onnx_wandb_run_path = onnx_wandb_run_path
    return p


def test_resolve_wandb_id_prefers_cli_wandb_path():
    # Explicit wandb:// launch wins even when the ONNX carries a run path.
    p = _wandb_policy(["wandb://entity/project/run_cli/model.onnx"], "entity/project/run_onnx")
    assert p._resolve_wandb_id(0) == "run_cli"


def test_resolve_wandb_id_falls_back_to_onnx_metadata():
    # Local .onnx path -> run id recovered from the export-stamped run path.
    p = _wandb_policy(["/models/policy.onnx"], "entity/project/run_onnx")
    assert p._resolve_wandb_id(0) == "run_onnx"


def test_resolve_wandb_id_none_when_no_provenance():
    p = _wandb_policy(["/models/local.onnx"], None)
    assert p._resolve_wandb_id(0) is None
