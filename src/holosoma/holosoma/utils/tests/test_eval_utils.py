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


def _create_yaml_config(tmp_path, content=None):
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


def _mock_wandb_config_download(mock_wandb_api, config_path):
    mock_file = mock_wandb_api.run.return_value.file.return_value
    mock_download = mock_file.download.return_value

    # Read the actual YAML content from the file
    with open(config_path) as f:
        file_content = f.read()

    # Create a file-like object with the content
    file_like_obj = io.StringIO(file_content)
    mock_download.__enter__.return_value = file_like_obj
    mock_download.__exit__.return_value = None
    return mock_file


def test_load_saved_experiment_config_from_wandb(mock_wandb_api: mock.MagicMock, tmp_path: Path) -> None:
    config_path = _create_yaml_config(tmp_path)
    _mock_wandb_config_download(mock_wandb_api, config_path)
    checkpoint_cfg = CheckpointConfig(
        checkpoint="wandb://test_user/test_project/test_run",
    )
    with mock.patch("wandb.Api", return_value=mock_wandb_api):
        loaded_cfg, run_path = load_saved_experiment_config(checkpoint_cfg)
    assert loaded_cfg is not None
    assert run_path == "test_user/test_project/test_run"
    mock_wandb_api.run.assert_called_once_with("test_user/test_project/test_run")
    mock_wandb_api.run.return_value.file.assert_called_once_with(CONFIG_NAME)


def test_load_saved_experiment_config_with_wandb_prefix(mock_wandb_api: mock.MagicMock, tmp_path: Path) -> None:
    config_path = _create_yaml_config(tmp_path)
    _mock_wandb_config_download(mock_wandb_api, config_path)
    checkpoint_cfg = CheckpointConfig(
        checkpoint="wandb://test_entity/test_project/test_run_id/model_100.pt",
    )
    with mock.patch("wandb.Api", return_value=mock_wandb_api):
        loaded_cfg, run_path = load_saved_experiment_config(checkpoint_cfg)
    assert loaded_cfg is not None
    assert run_path == "test_entity/test_project/test_run_id"
    mock_wandb_api.run.assert_called_once_with("test_entity/test_project/test_run_id")
    mock_wandb_api.run.return_value.file.assert_called_once_with(CONFIG_NAME)


def test_load_saved_experiment_config_with_wandb_runs_segment(mock_wandb_api: mock.MagicMock, tmp_path: Path) -> None:
    config_path = _create_yaml_config(tmp_path)
    _mock_wandb_config_download(mock_wandb_api, config_path)
    checkpoint_cfg = CheckpointConfig(
        checkpoint="wandb://test_entity/test_project/runs/test_run_id/model_100.pt",
    )
    with mock.patch("wandb.Api", return_value=mock_wandb_api):
        loaded_cfg, run_path = load_saved_experiment_config(checkpoint_cfg)
    assert loaded_cfg is not None
    assert run_path == "test_entity/test_project/test_run_id"
    mock_wandb_api.run.assert_called_once_with("test_entity/test_project/test_run_id")
    mock_wandb_api.run.return_value.file.assert_called_once_with(CONFIG_NAME)


def test_load_saved_experiment_config_with_wandb_run_only(mock_wandb_api: mock.MagicMock, tmp_path: Path) -> None:
    """Ensure wandb:// URIs without explicit checkpoint names can load configs."""
    config_path = _create_yaml_config(tmp_path)
    _mock_wandb_config_download(mock_wandb_api, config_path)
    checkpoint_cfg = CheckpointConfig(
        checkpoint="wandb://test_entity/test_project/test_run_id",
    )
    with mock.patch("wandb.Api", return_value=mock_wandb_api):
        loaded_cfg, run_path = load_saved_experiment_config(checkpoint_cfg)
    assert loaded_cfg is not None
    assert run_path == "test_entity/test_project/test_run_id"
    mock_wandb_api.run.assert_called_once_with("test_entity/test_project/test_run_id")
    mock_wandb_api.run.return_value.file.assert_called_once_with(CONFIG_NAME)


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


def test_load_saved_experiment_config_no_inputs() -> None:
    """Test that load_saved_experiment_config raises ValueError when no inputs are provided."""
    checkpoint_cfg = CheckpointConfig(
        checkpoint=None,
    )

    with pytest.raises(ValueError, match="No checkpoint provided"):
        load_saved_experiment_config(checkpoint_cfg)


def test_load_checkpoint(mock_wandb_api: mock.MagicMock, tmp_path: Path) -> None:
    """Test downloading checkpoints from W&B and using local checkpoints.

    Parameters
    ----------
    mock_wandb_api : mock.MagicMock
        Mock wandb API object
    tmp_path : Path
        Temporary directory for test files
    """
    # Test W&B download
    mock_run = mock.MagicMock()
    mock_file = mock.MagicMock()
    mock_run.file.return_value = mock_file
    mock_wandb_api.run.return_value = mock_run

    with mock.patch("wandb.Api", return_value=mock_wandb_api):
        checkpoint_path = load_checkpoint(
            checkpoint="wandb://test_user/test_project/test_run/model_100.pt",
            log_dir=str(tmp_path),
        )

    mock_wandb_api.run.assert_called_once_with("test_user/test_project/test_run")
    mock_run.file.assert_called_once_with("model_100.pt")
    mock_file.download.assert_called_once_with(root=str(tmp_path))
    assert checkpoint_path == tmp_path / "model_100.pt"

    # Test local checkpoint
    local_checkpoint = tmp_path / "local_model.pt"
    local_checkpoint.touch()  # Create empty file
    checkpoint_path = load_checkpoint(
        checkpoint=str(local_checkpoint),
        log_dir=str(tmp_path),
    )
    assert checkpoint_path == local_checkpoint


def test_load_checkpoint_with_wandb_prefix(mock_wandb_api: mock.MagicMock, tmp_path: Path) -> None:
    """Test loading checkpoint with wandb:// prefix.

    Parameters
    ----------
    mock_wandb_api : mock.MagicMock
        Mock wandb API object
    tmp_path : Path
        Temporary directory for test files
    """
    # Test wandb:// prefix parsing
    mock_run = mock.MagicMock()
    mock_file = mock.MagicMock()
    mock_run.file.return_value = mock_file
    mock_wandb_api.run.return_value = mock_run

    with mock.patch("wandb.Api", return_value=mock_wandb_api):
        checkpoint_path = load_checkpoint(
            checkpoint="wandb://test_entity/test_project/test_run_id/model_100.pt",
            log_dir=str(tmp_path),
        )

    # Verify the wandb path was correctly parsed and used
    mock_wandb_api.run.assert_called_once_with("test_entity/test_project/test_run_id")
    mock_run.file.assert_called_once_with("model_100.pt")
    mock_file.download.assert_called_once_with(root=str(tmp_path))
    assert checkpoint_path == tmp_path / "model_100.pt"


def test_load_checkpoint_with_wandb_runs_segment(mock_wandb_api: mock.MagicMock, tmp_path: Path) -> None:
    mock_run = mock.MagicMock()
    mock_file = mock.MagicMock()
    mock_run.file.return_value = mock_file
    mock_wandb_api.run.return_value = mock_run

    with mock.patch("wandb.Api", return_value=mock_wandb_api):
        checkpoint_path = load_checkpoint(
            checkpoint="wandb://test_entity/test_project/runs/test_run_id/model_100.pt",
            log_dir=str(tmp_path),
        )

    mock_wandb_api.run.assert_called_once_with("test_entity/test_project/test_run_id")
    mock_run.file.assert_called_once_with("model_100.pt")
    mock_file.download.assert_called_once_with(root=str(tmp_path))
    assert checkpoint_path == tmp_path / "model_100.pt"


def test_load_checkpoint_with_wandb_prefix_missing_checkpoint_name(tmp_path: Path) -> None:
    """Ensure wandb:// URIs for checkpoints include the artifact name."""
    with pytest.raises(
        ValueError,
        match="Expected format: wandb://<entity>/<project>/<run_id>/<checkpoint_name>",
    ):
        load_checkpoint(
            checkpoint="wandb://test_entity/test_project/test_run_id",
            log_dir=str(tmp_path),
        )
