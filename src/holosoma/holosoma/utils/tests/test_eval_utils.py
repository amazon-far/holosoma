from __future__ import annotations

import io
from pathlib import Path
from unittest import mock

import pytest
import torch
from omegaconf import OmegaConf

from holosoma.config_types.experiment import ExperimentConfig
from holosoma.utils.config_utils import CONFIG_NAME
from holosoma.utils.eval_utils import (
    CheckpointConfig,
    get_all_checkpoint_metadata,
    load_checkpoint,
    load_saved_experiment_config,
)


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

    # Should be copied to log_dir (not cache) with ORIGINAL filename preserved
    assert checkpoint_path == log_dir / "model_100.pt"
    assert Path(checkpoint_path).exists()
    assert Path(checkpoint_path).read_bytes() == b"fake checkpoint"
    # Verify original filename is used, not hash
    assert checkpoint_path.name == "model_100.pt"

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
    assert checkpoint_path == log_dir / "model_100.pt"
    assert Path(checkpoint_path).exists()
    assert checkpoint_path.name == "model_100.pt"  # Original name preserved


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

    # Should be copied to log_dir with ORIGINAL filename, not cache hash
    assert checkpoint_path == log_dir / "model_100.pt"
    assert Path(checkpoint_path).exists()
    assert checkpoint_path.name == "model_100.pt"  # Original name preserved


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

    # Create old file in log_dir
    old_log_file = log_dir / "model_5000.pt"
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
