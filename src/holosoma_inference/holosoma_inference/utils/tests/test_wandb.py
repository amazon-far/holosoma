"""Unit tests for holosoma_inference.utils.wandb module."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from holosoma_inference.utils import wandb as wandb_utils
from holosoma_inference.utils.wandb import _resolve_wandb_source, load_checkpoint


@pytest.mark.parametrize(
    ("run_path", "ckpt", "expected"),
    [
        (None, "wandb://ent/proj/rid/model.onnx", ("ent/proj/rid", "rid", "model.onnx")),
        (None, "https://wandb.ai/ent/proj/runs/rid/files/model.onnx", ("ent/proj/rid", "rid", "model.onnx")),
        ("ent/proj/rid", "model.onnx", ("ent/proj/rid", "rid", "model.onnx")),
        (None, "/local/model.onnx", (None, None, "/local/model.onnx")),
    ],
)
def test_resolve_wandb_source(run_path, ckpt, expected):
    assert _resolve_wandb_source(run_path, ckpt) == expected


def test_resolve_wandb_source_malformed_uri_raises():
    with pytest.raises(ValueError, match="Invalid wandb checkpoint path"):
        _resolve_wandb_source(None, "wandb://only/two")


def test_load_checkpoint_local_passthrough():
    assert load_checkpoint(None, "/local/model.onnx") == Path("/local/model.onnx")


def test_load_checkpoint_cache_hit_skips_download(tmp_path, monkeypatch):
    monkeypatch.setattr(wandb_utils, "_CACHE_DIR", tmp_path)
    (tmp_path / "rid").mkdir()
    (tmp_path / "rid" / "model.onnx").write_bytes(b"cached")

    with mock.patch.object(wandb_utils, "wandb") as mock_wandb:
        result = load_checkpoint("ent/proj/rid", "model.onnx")

    assert result == tmp_path / "rid" / "model.onnx"
    mock_wandb.Api.assert_not_called()


def test_load_checkpoint_atomic_download(tmp_path, monkeypatch):
    """Staged-then-renamed: final path absent mid-download, present after."""
    monkeypatch.setattr(wandb_utils, "_CACHE_DIR", tmp_path)
    final_path = tmp_path / "rid" / "model.onnx"

    def fake_download(root, replace):
        assert Path(root) != final_path.parent, "download must stage, not write to final path"
        Path(root, "model.onnx").write_bytes(b"weights")

    mock_file = mock.Mock()
    mock_file.download.side_effect = fake_download
    with mock.patch.object(wandb_utils, "wandb") as mock_wandb:
        mock_wandb.Api.return_value.run.return_value.file.return_value = mock_file
        result = load_checkpoint("ent/proj/rid", "model.onnx")

    assert result == final_path
    assert final_path.read_bytes() == b"weights"


def test_load_checkpoint_interrupted_download_leaves_no_cache_entry(tmp_path, monkeypatch):
    """Regression for JC's concern #1: crash mid-download must not populate the cache key."""
    monkeypatch.setattr(wandb_utils, "_CACHE_DIR", tmp_path)
    final_path = tmp_path / "rid" / "model.onnx"

    mock_file = mock.Mock()
    mock_file.download.side_effect = RuntimeError("network died")
    with mock.patch.object(wandb_utils, "wandb") as mock_wandb:
        mock_wandb.Api.return_value.run.return_value.file.return_value = mock_file
        with pytest.raises(RuntimeError, match="network died"):
            load_checkpoint("ent/proj/rid", "model.onnx")

    assert not final_path.exists()
    assert not any(tmp_path.rglob(".tmp.*"))  # staging dir cleaned up too
