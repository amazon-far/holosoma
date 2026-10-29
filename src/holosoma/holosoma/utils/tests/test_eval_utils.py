from __future__ import annotations

import copy
import io
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
import torch
import yaml
from omegaconf import OmegaConf

from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_types.mujoco import MujocoConfig, MujocoWarpConfig
from holosoma.config_values.simulator import SIMULATOR_REGISTRY
from holosoma.utils.config_registry import parse_config
from holosoma.utils.config_utils import CONFIG_NAME
from holosoma.utils.eval_utils import (
    CheckpointConfig,
    _drop_retired_isaacgym_fields,
    get_all_checkpoint_metadata,
    load_checkpoint,
    load_saved_experiment_config,
    replace_retired_simulator,
)
from holosoma.utils.simulator_config import set_simulator_type


@pytest.fixture
def mock_wandb_run() -> mock.MagicMock:
    """Create a mock wandb run object."""
    return mock.MagicMock()


@pytest.fixture
def mock_wandb_api(mock_wandb_run: mock.MagicMock) -> mock.MagicMock:
    """Create a mock wandb API object."""
    mock_api = mock.MagicMock()
    mock_api.run.return_value = mock_wandb_run
    return mock_api


def test_get_all_checkpoint_metadata_from_wandb(mock_wandb_api: mock.MagicMock) -> None:
    """Test getting checkpoint metadata from wandb run."""
    # Create mock files with proper name attribute
    mock_files = []
    for name in ["model_100.pt", "model_2.pt", "config.yaml", "model_10.pt", "invalid.pt"]:
        mock_file = mock.MagicMock()
        mock_file.name = name
        mock_files.append(mock_file)

    # Set up the mock to return the files
    mock_wandb_api.run.return_value.files.return_value = mock_files

    # Mock the scan_history to return runtime data
    mock_history = [
        {"global_step": 2, "_runtime": 100.0, "Train/num_samples": 1000},
        {"global_step": 10, "_runtime": 200.0, "Train/num_samples": 5000},
        {"global_step": 100, "_runtime": 300.0, "Train/num_samples": 50000},
    ]
    mock_wandb_api.run.return_value.scan_history.return_value = mock_history

    # Create override config
    override_config = OmegaConf.create(
        {
            "wandb_run_path": "test_user/test_project/test_run",
            "checkpoint_dir": None,
        }
    )

    with mock.patch("wandb.Api", return_value=mock_wandb_api):
        checkpoint_metadata = get_all_checkpoint_metadata(override_config)

    # Verify the checkpoint metadata is in order
    expected_metadata = [
        {"file_name": "model_2.pt", "global_step": 2, "train_runtime": 100.0, "num_samples": 1000},
        {"file_name": "model_10.pt", "global_step": 10, "train_runtime": 200.0, "num_samples": 5000},
        {"file_name": "model_100.pt", "global_step": 100, "train_runtime": 300.0, "num_samples": 50000},
    ]
    assert checkpoint_metadata == expected_metadata
    mock_wandb_api.run.assert_called_once_with("test_user/test_project/test_run")


def test_get_all_checkpoint_metadata_from_local(tmp_path: Path) -> None:
    """Test getting checkpoint metadata from local directory."""
    # Create checkpoint files
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model_100.pt").touch()
    (checkpoint_dir / "model_2.pt").touch()
    (checkpoint_dir / "config.yaml").touch()
    (checkpoint_dir / "model_10.pt").touch()
    (checkpoint_dir / "invalid.pt").touch()

    # Create override config
    override_config = OmegaConf.create(
        {
            "wandb_run_path": None,
            "checkpoint_dir": str(checkpoint_dir),
        }
    )

    checkpoint_metadata = get_all_checkpoint_metadata(override_config)

    # Verify the checkpoint metadata is in order
    expected_metadata = [
        {"file_name": "model_2.pt", "global_step": 2, "train_runtime": None, "num_samples": None},
        {"file_name": "model_10.pt", "global_step": 10, "train_runtime": None, "num_samples": None},
        {"file_name": "model_100.pt", "global_step": 100, "train_runtime": None, "num_samples": None},
    ]
    assert checkpoint_metadata == expected_metadata


def test_get_all_checkpoint_metadata_resolves_user_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model_7.pt").touch()
    monkeypatch.setenv("HOME", str(tmp_path))

    metadata = get_all_checkpoint_metadata(
        OmegaConf.create({"wandb_run_path": None, "checkpoint_dir": "~/checkpoints"})
    )

    assert metadata == [
        {
            "file_name": "model_7.pt",
            "global_step": 7,
            "train_runtime": None,
            "num_samples": None,
        }
    ]


def test_get_all_checkpoint_metadata_no_inputs() -> None:
    """Test that get_all_checkpoint_metadata raises ValueError when no inputs are provided."""
    override_config = OmegaConf.create(
        {
            "wandb_run_path": None,
            "checkpoint_dir": None,
        }
    )

    with pytest.raises(ValueError, match="No checkpoint directory or wandb run path provided"):
        get_all_checkpoint_metadata(override_config)


def _create_yaml_config(tmp_path: Path, content: str | None = None) -> Path:
    config_path = tmp_path / CONFIG_NAME
    config_content = (
        content
        or """
    base_field: original_value
    nested:
        field1: original_nested_value
        field2: unchanged_value
    override_field:
        base_field: override_value
        nested:
            field1: overridden_nested_value
    """
    )
    with open(config_path, "w") as f:
        f.write(config_content)
    return config_path


def _mock_wandb_config_download(mock_wandb_api: mock.MagicMock, config_path: Path) -> mock.MagicMock:
    mock_file: mock.MagicMock = mock_wandb_api.run.return_value.file.return_value
    mock_download = mock_file.download.return_value

    # Read the actual YAML content from the file
    with open(config_path) as f:
        file_content = f.read()

    # Create a file-like object with the content
    file_like_obj = io.StringIO(file_content)
    mock_download.__enter__.return_value = file_like_obj
    mock_download.__exit__.return_value = None
    return mock_file


def test_load_saved_experiment_config_from_wandb(tmp_path: Path) -> None:
    """Test loading config from W&B run."""
    config_path = _create_yaml_config(tmp_path)
    checkpoint_cfg = CheckpointConfig(
        checkpoint="wandb://test_user/test_project/test_run",
    )
    # Mock get_cached_file_path to return our config
    with mock.patch("holosoma.utils.eval_utils.get_cached_file_path", return_value=str(config_path)):
        loaded_cfg, run_path = load_saved_experiment_config(checkpoint_cfg)
    assert loaded_cfg is not None
    assert run_path == "test_user/test_project/test_run"


def test_load_saved_experiment_config_with_wandb_prefix(tmp_path: Path) -> None:
    """Test loading config with wandb:// prefix and checkpoint name."""
    config_path = _create_yaml_config(tmp_path)
    checkpoint_cfg = CheckpointConfig(
        checkpoint="wandb://test_entity/test_project/test_run_id/model_100.pt",
    )
    with mock.patch("holosoma.utils.eval_utils.get_cached_file_path", return_value=str(config_path)):
        loaded_cfg, run_path = load_saved_experiment_config(checkpoint_cfg)
    assert loaded_cfg is not None
    assert run_path == "test_entity/test_project/test_run_id"


def test_load_saved_experiment_config_with_wandb_runs_segment(tmp_path: Path) -> None:
    """Test loading config with 'runs' segment in path."""
    config_path = _create_yaml_config(tmp_path)
    checkpoint_cfg = CheckpointConfig(
        checkpoint="wandb://test_entity/test_project/runs/test_run_id/model_100.pt",
    )
    with mock.patch("holosoma.utils.eval_utils.get_cached_file_path", return_value=str(config_path)):
        loaded_cfg, run_path = load_saved_experiment_config(checkpoint_cfg)
    assert loaded_cfg is not None
    assert run_path == "test_entity/test_project/test_run_id"


def test_load_saved_experiment_config_with_wandb_run_only(tmp_path: Path) -> None:
    """Ensure wandb:// URIs without explicit checkpoint names can load configs."""
    config_path = _create_yaml_config(tmp_path)
    checkpoint_cfg = CheckpointConfig(
        checkpoint="wandb://test_entity/test_project/test_run_id",
    )
    with mock.patch("holosoma.utils.eval_utils.get_cached_file_path", return_value=str(config_path)):
        loaded_cfg, run_path = load_saved_experiment_config(checkpoint_cfg)
    assert loaded_cfg is not None
    assert run_path == "test_entity/test_project/test_run_id"


def test_load_saved_experiment_config_from_checkpoint(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "model.pt"
    cfg = ExperimentConfig()
    torch.save(
        {
            "actor_model_state_dict": {},
            "experiment_config": cfg.to_serializable_dict(),
            "wandb_run_path": "entity/project/run",
        },
        checkpoint_path,
    )
    checkpoint_cfg = CheckpointConfig(
        checkpoint=str(checkpoint_path),
    )
    loaded_cfg, run_path = load_saved_experiment_config(checkpoint_cfg)
    assert loaded_cfg == cfg
    assert run_path == "entity/project/run"


def _load_saved_config(
    data: dict[str, Any], source: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ExperimentConfig, str | None]:
    if source == "checkpoint":
        path = tmp_path / "model.pt"
        torch.save({"experiment_config": data, "wandb_run_path": "entity/project/run"}, path)
        reference = str(path)
    else:
        path = tmp_path / CONFIG_NAME
        path.write_text(yaml.safe_dump(data))
        monkeypatch.setattr("holosoma.utils.eval_utils.get_cached_file_path", lambda _: str(path))
        reference = "wandb://entity/project/run"
    return load_saved_experiment_config(CheckpointConfig(checkpoint=reference))


@pytest.mark.parametrize("source", ["checkpoint", "wandb"])
@pytest.mark.parametrize("backend", ["isaacsim", "mujoco", "mjwarp"])
def test_historical_saved_configs_transition_warp_capacities(
    source: str, backend: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = replace(ExperimentConfig(), simulator=SIMULATOR_REGISTRY[backend], plugin={})
    saved = cfg.to_serializable_dict()
    saved["simulator"]["config"]["sim"].pop("mujoco")
    saved["simulator"]["config"]["mujoco_warp"] = {"nconmax_per_env": 128, "njmax_per_env": 900}
    loaded, run_path = _load_saved_config(saved, source, tmp_path, monkeypatch)
    assert run_path == "entity/project/run"
    assert loaded.simulator.config.name == cfg.simulator.config.name
    assert loaded.simulator.config.mujoco_backend == cfg.simulator.config.mujoco_backend
    assert loaded.simulator.config.sim.mujoco.warp == MujocoWarpConfig(nconmax_per_env=128, njmax_per_env=900)
    assert loaded.robot == cfg.robot
    assert loaded.training == cfg.training
    assert loaded.plugin == {}
    assert "mujoco_warp" not in loaded.to_serializable_dict()["simulator"]["config"]


@pytest.mark.parametrize("source", ["checkpoint", "wandb"])
@pytest.mark.parametrize("gravity", [None, [0, 0, 0], [0, 0, -2]])
def test_saved_native_gravity_transitions_to_extra(
    source: str, gravity: list[float] | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved = ExperimentConfig().to_serializable_dict()
    saved["simulator"]["config"]["sim"]["mujoco"]["gravity"] = gravity
    loaded, _ = _load_saved_config(saved, source, tmp_path, monkeypatch)
    assert loaded.simulator.config.sim.mujoco.extra == ({} if gravity is None else {"gravity": gravity})


@pytest.mark.parametrize("source", ["checkpoint", "wandb"])
def test_saved_native_physics_preserves_zero_false_and_extras(
    source: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulator = SIMULATOR_REGISTRY["mjwarp"]
    native = MujocoConfig(
        integrator="rk4",
        cone="elliptic",
        impratio=10,
        iterations=0,
        extra={"gravity": [0, 0, -2]},
        warp=MujocoWarpConfig(nconmax_per_env=128, naconmax=300, graph_conditional=False),
    )
    simulator = replace(simulator, config=replace(simulator.config, sim=replace(simulator.config.sim, mujoco=native)))
    saved = replace(ExperimentConfig(), simulator=simulator, plugin={}).to_serializable_dict()
    loaded, _ = _load_saved_config(saved, source, tmp_path, monkeypatch)
    assert loaded.simulator.config.sim.mujoco == native
    assert loaded.plugin == {}
    assert loaded.to_serializable_dict() == saved


@pytest.mark.parametrize("source", ["checkpoint", "wandb"])
def test_saved_warp_transition_rejects_conflicting_values(
    source: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved = ExperimentConfig().to_serializable_dict()
    saved["simulator"]["config"]["mujoco_warp"] = {"nconmax_per_env": 128}
    saved["simulator"]["config"]["sim"]["mujoco"]["warp"]["nconmax_per_env"] = 64
    with pytest.raises(ValueError, match=r"conflicting mujoco_warp\.nconmax_per_env.*sim\.mujoco\.warp"):
        _load_saved_config(saved, source, tmp_path, monkeypatch)


def test_load_saved_experiment_config_uses_generic_remote_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_path = tmp_path / "model.pt"
    cfg = ExperimentConfig()
    torch.save(
        {
            "experiment_config": cfg.to_serializable_dict(),
            "wandb_run_path": None,
        },
        checkpoint_path,
    )
    uri = "s3://bucket/model.pt"
    monkeypatch.setattr(
        "holosoma.utils.file_cache.get_cached_file_path",
        lambda value: str(checkpoint_path) if value == uri else None,
    )

    loaded_cfg, run_path = load_saved_experiment_config(CheckpointConfig(checkpoint=uri))

    assert loaded_cfg == cfg
    assert run_path is None


def test_load_saved_experiment_config_no_inputs() -> None:
    """Test that load_saved_experiment_config raises ValueError when no inputs are provided."""
    checkpoint_cfg = CheckpointConfig(
        checkpoint=None,
    )

    with pytest.raises(ValueError, match="No checkpoint provided"):
        load_saved_experiment_config(checkpoint_cfg)


def test_load_checkpoint(tmp_path: Path) -> None:
    """Test downloading checkpoints from W&B and using local checkpoints.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory for test files
    """
    # Create a fake cached checkpoint
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cached_file = cache_dir / "model_100.pt"
    cached_file.write_bytes(b"fake checkpoint")

    # Test W&B download (mock get_cached_file_path)
    log_dir = tmp_path / "logs"
    with mock.patch("holosoma.utils.eval_utils.get_cached_file_path", return_value=str(cached_file)):
        checkpoint_path = load_checkpoint(
            checkpoint="wandb://test_user/test_project/test_run/model_100.pt",
            log_dir=str(log_dir),
        )

    # Should be copied to log_dir (not cache) with the ORIGINAL filename, under
    # the run-id prefix that keeps multi-teacher checkpoints from colliding.
    assert checkpoint_path == log_dir / "test_run_model_100.pt"
    assert Path(checkpoint_path).exists()
    assert Path(checkpoint_path).read_bytes() == b"fake checkpoint"
    # Verify original filename is used, not hash
    assert checkpoint_path.name == "test_run_model_100.pt"

    # Test local checkpoint
    local_checkpoint = tmp_path / "local_model.pt"
    local_checkpoint.write_bytes(b"local checkpoint")
    checkpoint_path = load_checkpoint(
        checkpoint=str(local_checkpoint),
        log_dir=str(log_dir),
    )
    assert str(checkpoint_path) == str(local_checkpoint)


def test_load_checkpoint_uses_generic_remote_loader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    local_checkpoint = tmp_path / "cached.pt"
    local_checkpoint.write_bytes(b"checkpoint")
    uri = "s3://bucket/model.pt"
    monkeypatch.setattr(
        "holosoma.utils.file_cache.get_cached_file_path",
        lambda value: str(local_checkpoint) if value == uri else None,
    )

    assert load_checkpoint(uri, str(tmp_path / "logs")) == local_checkpoint


def test_load_checkpoint_with_wandb_prefix(tmp_path: Path) -> None:
    """Test loading checkpoint with wandb:// prefix.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory for test files
    """
    # Create fake cached checkpoint (with hash name in cache)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cached_file = cache_dir / "abc123hash.pt"
    cached_file.write_bytes(b"fake checkpoint")

    log_dir = tmp_path / "logs"
    with mock.patch("holosoma.utils.eval_utils.get_cached_file_path", return_value=str(cached_file)):
        checkpoint_path = load_checkpoint(
            checkpoint="wandb://test_entity/test_project/test_run_id/model_100.pt",
            log_dir=str(log_dir),
        )

    # Should be copied to log_dir with ORIGINAL filename, not cache hash
    assert checkpoint_path == log_dir / "test_run_id_model_100.pt"
    assert Path(checkpoint_path).exists()
    assert checkpoint_path.name == "test_run_id_model_100.pt"  # Original name preserved


def test_load_checkpoint_with_wandb_runs_segment(tmp_path: Path) -> None:
    """Test loading checkpoint with 'runs' segment in path."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cached_file = cache_dir / "def456hash.pt"
    cached_file.write_bytes(b"fake checkpoint")

    log_dir = tmp_path / "logs"
    with mock.patch("holosoma.utils.eval_utils.get_cached_file_path", return_value=str(cached_file)):
        checkpoint_path = load_checkpoint(
            checkpoint="wandb://test_entity/test_project/runs/test_run_id/model_100.pt",
            log_dir=str(log_dir),
        )

    # Should be copied to log_dir with ORIGINAL filename, not cache hash. The
    # run-id prefix drops the "runs" segment along with the entity/project.
    assert checkpoint_path == log_dir / "test_run_id_model_100.pt"
    assert Path(checkpoint_path).exists()
    assert checkpoint_path.name == "test_run_id_model_100.pt"  # Original name preserved


def test_load_checkpoint_log_dir_copy_behavior(tmp_path: Path) -> None:
    """Test that checkpoints are copied from cache to log_dir."""
    import time

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cached_file = cache_dir / "model_5000.pt"
    cached_file.write_bytes(b"fake checkpoint content")

    log_dir = tmp_path / "logs"

    # Mock get_cached_file_path to return our fake cache
    with mock.patch("holosoma.utils.eval_utils.get_cached_file_path", return_value=str(cached_file)):
        # First call - creates copy
        result1 = load_checkpoint("wandb://entity/project/run_id/model_5000.pt", str(log_dir))
        result1_path = Path(result1)
        mtime1 = result1_path.stat().st_mtime

        # Verify copy exists in log_dir
        assert log_dir in result1_path.parents
        assert result1_path.exists()
        assert result1_path.read_bytes() == b"fake checkpoint content"

        # Small delay
        time.sleep(0.01)

        # Second call - should reuse existing copy
        result2 = load_checkpoint("wandb://entity/project/run_id/model_5000.pt", str(log_dir))
        result2_path = Path(result2)
        mtime2 = result2_path.stat().st_mtime

        # Should return same path and not re-copy (mtime unchanged)
        assert result1 == result2
        assert mtime1 == mtime2


def test_load_checkpoint_updates_stale_copy(tmp_path: Path) -> None:
    """Test that log_dir copy is refreshed when cache is newer."""
    import os
    import time

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cached_file = cache_dir / "model_5000.pt"
    cached_file.write_bytes(b"new content")

    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    # Create old file in log_dir, under the run-id-prefixed name load_checkpoint
    # actually writes -- seeding a bare "model_5000.pt" would leave the refresh
    # path untested, since the copy would simply be missing.
    old_log_file = log_dir / "run_id_model_5000.pt"
    old_log_file.write_bytes(b"old content")
    old_mtime = time.time() - 3600  # 1 hour ago
    os.utime(old_log_file, (old_mtime, old_mtime))

    # Mock get_cached_file_path to return newer cache
    with mock.patch("holosoma.utils.eval_utils.get_cached_file_path", return_value=str(cached_file)):
        result = load_checkpoint("wandb://entity/project/run_id/model_5000.pt", str(log_dir))
        result_path = Path(result)

        # Should have updated content from cache
        assert result_path.read_bytes() == b"new content"
        assert result_path.stat().st_mtime > old_mtime


# Configs saved by `to_serializable_dict()` at 8b83d915, before the Isaac Gym retirement (see holosoma-v0.0.3).
# They carry the `isaacgym` blocks every backend serialized before the retirement.
_PREVIOUS_SCHEMA_DIR = Path(__file__).parent / "data" / "pre_isaacgym_retirement"
_SURVIVING_BACKEND_FIXTURES = {
    "g1_29dof_mujoco": "mujoco",
    "t1_29dof_isaacsim": "isaacsim",
    "g1_29dof_wbt_w_object_mjwarp_g1_head": "mjwarp",
}
_MATERIAL_TERM = "randomize_object_rigid_body_material_startup"


def _previous_schema(name: str) -> dict[str, Any]:
    with open(_PREVIOUS_SCHEMA_DIR / f"{name}.yaml") as f:
        data: dict[str, Any] = yaml.safe_load(f)
    return data


def _without(block: dict[str, Any], key: str = "isaacgym") -> dict[str, Any]:
    return {k: v for k, v in block.items() if k != key}


def _load_through(boundary: str, data: dict[str, Any], tmp_path: Path) -> ExperimentConfig:
    if boundary == "checkpoint":
        checkpoint_path = tmp_path / "model.pt"
        torch.save({"experiment_config": data, "wandb_run_path": None}, checkpoint_path)
        loaded_cfg, _ = load_saved_experiment_config(CheckpointConfig(checkpoint=str(checkpoint_path)))
        return loaded_cfg
    config_path = tmp_path / CONFIG_NAME
    config_path.write_text(yaml.safe_dump(data))
    with mock.patch("holosoma.utils.eval_utils.get_cached_file_path", return_value=str(config_path)):
        loaded_cfg, _ = load_saved_experiment_config(CheckpointConfig(checkpoint="wandb://entity/project/run"))
    return loaded_cfg


@pytest.mark.parametrize("boundary", ["checkpoint", "wandb"])
@pytest.mark.parametrize("fixture_name", sorted(_SURVIVING_BACKEND_FIXTURES))
def test_previous_schema_configs_load_with_backend_settings_intact(
    fixture_name: str, boundary: str, tmp_path: Path
) -> None:
    saved = _previous_schema(fixture_name)

    loaded_cfg = _load_through(boundary, saved, tmp_path)
    loaded = loaded_cfg.to_serializable_dict()

    preset = SIMULATOR_REGISTRY[_SURVIVING_BACKEND_FIXTURES[fixture_name]]
    assert (loaded_cfg.simulator._target_, loaded_cfg.simulator.config.name) == (preset._target_, preset.config.name)
    saved_simulator = copy.deepcopy(saved["simulator"])
    saved_simulator["config"].pop("contact_sensor_history_length")
    saved_warp = saved_simulator["config"].pop("mujoco_warp")
    loaded_simulator = copy.deepcopy(loaded["simulator"])
    loaded_simulator["config"]["sim"].pop("mujoco")
    assert loaded_simulator == saved_simulator
    assert loaded_cfg.simulator.config.sim.mujoco.warp.nconmax_per_env == saved_warp["nconmax_per_env"]
    assert loaded_cfg.simulator.config.sim.mujoco.warp.njmax_per_env == saved_warp["njmax_per_env"]
    assert loaded["robot"]["asset"]["link_physics"] == _without(saved["robot"]["asset"]["link_physics"])
    assert "apply_dof_armature_in_isaacgym" not in loaded["robot"]
    for name, sensor in saved["sensor"].items():
        loaded_sensor = copy.deepcopy(loaded["sensor"][name])
        loaded_isaacsim = loaded_sensor["isaacsim"]
        assert (loaded_isaacsim.pop("renderer"), loaded_isaacsim.pop("raycast_link_meshes")) == ("rtx", {})
        assert loaded_sensor == _without(sensor)
    saved_material_term = saved["randomization"]["setup_terms"].get(_MATERIAL_TERM)
    if saved_material_term is not None:
        loaded_material = loaded["randomization"]["setup_terms"][_MATERIAL_TERM]["params"]["material"]
        assert loaded_material == _without(saved_material_term["params"]["material"])


def test_scene_object_physics_drops_only_the_isaacgym_block() -> None:
    mujoco_physics = {"friction": [0.7, 0.005, 0.0001]}
    physics = {"isaacgym": {"friction": 0.5}, "mujoco": mujoco_physics}
    saved: dict[str, Any] = {
        "scene": {
            "rigid_objects": {"box": {"physics": copy.deepcopy(physics)}},
            "scene_files": {"room": {"object_configs": {"*chair*": {"physics": copy.deepcopy(physics)}}}},
        }
    }

    migrated = _drop_retired_isaacgym_fields(saved)

    assert migrated["scene"]["rigid_objects"]["box"]["physics"] == {"mujoco": mujoco_physics}
    assert migrated["scene"]["scene_files"]["room"]["object_configs"]["*chair*"]["physics"] == {
        "mujoco": mujoco_physics
    }
    assert saved["scene"]["rigid_objects"]["box"]["physics"] == physics  # the caller's dict is not mutated


def test_saved_isaacgym_simulator_reports_unsupported_without_override(tmp_path: Path) -> None:
    saved_cfg = _load_through("checkpoint", _previous_schema("g1_29dof_isaacgym"), tmp_path)

    eval_cfg = replace_retired_simulator(saved_cfg.get_eval_config(), [])

    with pytest.raises(ValueError, match="Isaac Gym simulator is no longer supported"):
        set_simulator_type(eval_cfg.simulator)


@pytest.mark.parametrize("backend", ["isaacsim", "mujoco", "mjwarp"])
def test_saved_isaacgym_simulator_can_be_overridden_on_the_cli(backend: str, tmp_path: Path) -> None:
    # Without replace_retired_simulator, tyro attaches the unmatched Isaac Gym default to the
    # isaacsim slot, so `simulator:isaacsim` would come back as Isaac Gym.
    saved_cfg = _load_through("checkpoint", _previous_schema("g1_29dof_isaacgym"), tmp_path)
    args = [f"simulator:{backend}"]

    eval_cfg = parse_config(
        ExperimentConfig, default=replace_retired_simulator(saved_cfg.get_eval_config(), args), args=args
    )

    assert eval_cfg.simulator == SIMULATOR_REGISTRY[backend]
