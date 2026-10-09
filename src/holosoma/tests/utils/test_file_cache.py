"""Unit tests for file caching utilities.

This test suite focuses on internal protocol-agnostic functions to minimize
test count while maintaining comprehensive coverage. Protocol-specific logic
is tested minimally, as most bugs are in the core caching logic.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from holosoma.utils.file_cache import (
    _get_cache_path,
    _get_metadata_path,
    _get_protocol,
    _is_cache_valid,
    _is_remote_uri,
    _load_metadata,
    _save_metadata,
    _uri_to_hash,
    cached_open,
    clear_cache,
    get_cache_stats,
    get_cached_file_path,
)
from holosoma.utils.wandb import parse_wandb_uri


@pytest.fixture
def temp_cache_dir(tmp_path, monkeypatch):
    """Use temporary directory for cache during tests."""
    cache_dir = tmp_path / "test_cache"
    cache_dir.mkdir()
    monkeypatch.setenv("HOLOSOMA_CACHE_DIR", str(cache_dir))
    return cache_dir


@pytest.fixture
def mock_smart_open(monkeypatch):
    """Mock smart_open for S3/HTTP downloads."""

    class MockFileReader:
        def __init__(self, content: bytes):
            self.content = content

        def read(self) -> bytes:
            return self.content

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def mock_open(uri: str, mode: str):
        # Return fake content based on URI
        return MockFileReader(f"fake content for {uri}".encode())

    # Mock the _get_smart_open function to return our mock
    mock_module = MagicMock()
    mock_module.open = mock_open

    def mock_get_smart_open():
        return mock_module

    monkeypatch.setattr("holosoma.utils.file_cache._get_smart_open", mock_get_smart_open)
    return mock_open


@pytest.fixture
def mock_wandb(monkeypatch):
    """Mock wandb API for checkpoint downloads."""

    class MockFile:
        def __init__(self, name: str):
            self.name = name

        def download(self, root: str, replace: bool = False):
            # Create fake file
            path = Path(root) / self.name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fake wandb checkpoint content")

    class MockRun:
        def file(self, name: str):
            return MockFile(name)

    class MockApi:
        def run(self, path: str):
            return MockRun()

    # Mock the _get_wandb function
    mock_wandb_module = MagicMock()
    mock_wandb_module.Api = MockApi

    def mock_get_wandb():
        return mock_wandb_module

    monkeypatch.setattr("holosoma.utils.wandb.get_wandb", mock_get_wandb)
    return mock_wandb_module


class TestCacheInternals:
    """Test core caching logic - protocol-agnostic."""

    def test_uri_to_hash_deterministic(self):
        """Hash same URI consistently."""
        uri = "fake://example.com/file.txt"
        hash1 = _uri_to_hash(uri)
        hash2 = _uri_to_hash(uri)
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 produces 64 hex chars

    def test_uri_to_hash_collision_resistant(self):
        """Different URIs get different hashes."""
        hash1 = _uri_to_hash("fake://a")
        hash2 = _uri_to_hash("fake://b")
        hash3 = _uri_to_hash("fake://a/different")
        assert hash1 != hash2
        assert hash1 != hash3
        assert hash2 != hash3

    def test_cache_path_structure(self, temp_cache_dir):
        """Cache path follows expected structure."""
        # Use s3:// since fake:// protocol isn't recognized
        path = _get_cache_path("s3://bucket/file.npz")

        # Should be under cache dir
        assert temp_cache_dir in path.parents

        # Should have protocol subdirectory
        assert path.parent.name == "s3"

        # Should preserve extension
        assert path.suffix == ".npz"

    def test_cache_path_extension_preservation(self, temp_cache_dir):
        """Cache path preserves file extensions."""
        extensions = [".pt", ".npz", ".yaml", ".onnx", ".txt", ""]
        for ext in extensions:
            path = _get_cache_path(f"fake://file{ext}")
            assert path.suffix == ext

    def test_is_cache_valid_nonexistent(self):
        """Non-existent file is invalid."""
        fake_path = Path("/nonexistent/directory/file.txt")
        assert not _is_cache_valid(fake_path)

    def test_is_cache_valid_empty_file(self, tmp_path):
        """Empty file is invalid."""
        empty_file = tmp_path / "empty.txt"
        empty_file.touch()  # Creates empty file
        assert not _is_cache_valid(empty_file)

    def test_is_cache_valid_with_content(self, tmp_path):
        """File with content is valid."""
        valid_file = tmp_path / "valid.txt"
        valid_file.write_bytes(b"some content")
        assert _is_cache_valid(valid_file)

    def test_ttl_expiration(self, tmp_path, monkeypatch):
        """Cache expires after TTL."""
        cache_file = tmp_path / "file.txt"
        cache_file.write_bytes(b"content")

        # Save metadata with old timestamp (2 hours ago)
        old_time = time.time() - 7200  # 2 hours = 120 minutes ago
        _save_metadata(cache_file, "s3://test/uri")

        # Manually update the timestamp to be old
        metadata_path = _get_metadata_path(cache_file)
        with open(metadata_path) as f:
            metadata = json.load(f)
        metadata["cached_at"] = old_time
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        # Set TTL to 60 minutes (default)
        monkeypatch.setenv("HOLOSOMA_CACHE_TTL_MINUTES", "60")

        # Should be invalid (expired)
        assert not _is_cache_valid(cache_file)

    def test_ttl_zero_never_expires(self, tmp_path, monkeypatch):
        """TTL of 0 means cache never expires."""
        cache_file = tmp_path / "file.txt"
        cache_file.write_bytes(b"content")

        # Save metadata with very old timestamp (10 years ago)
        old_time = time.time() - (10 * 365 * 24 * 3600)
        _save_metadata(cache_file, "s3://test/uri")

        # Manually update the timestamp to be very old
        metadata_path = _get_metadata_path(cache_file)
        with open(metadata_path) as f:
            metadata = json.load(f)
        metadata["cached_at"] = old_time
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        # Set TTL to 0 (never expires)
        monkeypatch.setenv("HOLOSOMA_CACHE_TTL_MINUTES", "0")

        # Should still be valid
        assert _is_cache_valid(cache_file)

    def test_ttl_not_yet_expired(self, tmp_path, monkeypatch):
        """Cache not expired if within TTL."""
        cache_file = tmp_path / "file.txt"
        cache_file.write_bytes(b"content")

        # Save metadata with recent timestamp (30 minutes ago)
        recent_time = time.time() - 1800  # 30 minutes ago
        _save_metadata(cache_file, "s3://test/uri")

        # Manually update the timestamp
        metadata_path = _get_metadata_path(cache_file)
        with open(metadata_path) as f:
            metadata = json.load(f)
        metadata["cached_at"] = recent_time
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        # Set TTL to 60 minutes
        monkeypatch.setenv("HOLOSOMA_CACHE_TTL_MINUTES", "60")

        # Should still be valid (within TTL)
        assert _is_cache_valid(cache_file)

    def test_metadata_save_and_load(self, tmp_path):
        """Metadata round-trips correctly."""
        cache_file = tmp_path / "file.txt"
        cache_file.write_bytes(b"test content")

        # Save metadata (use s3:// since fake:// isn't recognized)
        _save_metadata(cache_file, "s3://test/uri", {"extra": "info"})

        # Load and verify
        metadata = _load_metadata(cache_file)
        assert metadata is not None
        assert metadata["uri"] == "s3://test/uri"
        assert metadata["extra"] == "info"
        assert "cached_at" in metadata
        assert "cached_at_iso" in metadata
        assert "size_bytes" in metadata
        assert metadata["size_bytes"] == 12  # len(b"test content")
        assert metadata["protocol"] == "s3"

    def test_metadata_path_generation(self, tmp_path):
        """Metadata path is correctly derived from cache path."""
        cache_path = tmp_path / "abc123.npz"
        meta_path = _get_metadata_path(cache_path)

        assert meta_path.parent == cache_path.parent
        # The implementation uses {stem}.meta.json format
        assert meta_path.name == "abc123.meta.json"
        assert meta_path.suffix == ".json"

    def test_metadata_load_missing_file(self, tmp_path):
        """Loading metadata from non-existent file returns None."""
        cache_path = tmp_path / "nonexistent.txt"
        metadata = _load_metadata(cache_path)
        assert metadata is None

    def test_metadata_load_corrupted_file(self, tmp_path):
        """Loading corrupted metadata returns None and logs warning."""
        cache_path = tmp_path / "file.txt"
        meta_path = _get_metadata_path(cache_path)
        meta_path.write_text("not valid json{")

        metadata = _load_metadata(cache_path)
        assert metadata is None


class TestProtocolDetection:
    """Test protocol detection and classification."""

    def test_is_remote_uri_detection(self):
        """Correctly identify remote URIs."""
        assert _is_remote_uri("s3://bucket/file")
        assert _is_remote_uri("wandb://entity/project/run/file")
        assert _is_remote_uri("http://example.com/file")
        assert _is_remote_uri("https://example.com/file")

        assert not _is_remote_uri("/local/path")
        assert not _is_remote_uri("./relative/path")
        assert not _is_remote_uri("file.txt")

    def test_get_protocol_classification(self):
        """Correctly classify protocols."""
        assert _get_protocol("s3://bucket/file") == "s3"
        assert _get_protocol("wandb://e/p/r/f") == "wandb"
        assert _get_protocol("http://example.com") == "http"
        assert _get_protocol("https://example.com") == "https"
        assert _get_protocol("/local/path") == "local"
        assert _get_protocol("./relative") == "local"


class TestWandbUriParsing:
    """Test W&B-specific URI parsing."""

    def test_basic_wandb_uri_parsing(self):
        """Parse basic W&B URI."""
        uri = "wandb://entity/project/run_id/file.pt"
        run_path, file_name = parse_wandb_uri(uri)

        assert run_path == "entity/project/run_id"
        assert file_name == "file.pt"

    def test_wandb_uri_with_runs_keyword(self):
        """Parse W&B URI with 'runs' keyword."""
        uri = "wandb://entity/project/runs/run_id/file.pt"
        run_path, file_name = parse_wandb_uri(uri)

        assert run_path == "entity/project/run_id"
        assert file_name == "file.pt"

    def test_wandb_uri_with_nested_path(self):
        """Parse W&B URI with nested file path."""
        uri = "wandb://entity/project/run_id/checkpoints/best/model.pt"
        run_path, file_name = parse_wandb_uri(uri)

        assert run_path == "entity/project/run_id"
        assert file_name == "checkpoints/best/model.pt"

    def test_wandb_uri_invalid_format(self):
        """Reject invalid W&B URIs."""
        with pytest.raises(ValueError, match="Invalid wandb URI"):
            parse_wandb_uri("wandb://invalid")

        with pytest.raises(ValueError, match="Invalid wandb URI"):
            parse_wandb_uri("wandb://entity/project")

    def test_wandb_uri_wrong_protocol(self):
        """Reject non-wandb URIs."""
        with pytest.raises(ValueError, match="Not a wandb URI"):
            parse_wandb_uri("s3://bucket/file")


class TestCacheManagement:
    """Test cache management operations."""

    def test_cache_stats_empty(self, temp_cache_dir):
        """Empty cache returns zero stats."""
        stats = get_cache_stats()

        assert stats["total_files"] == 0
        assert stats["total_size_bytes"] == 0
        assert stats["protocols"] == {}

    def test_cache_stats_with_files(self, temp_cache_dir):
        """Cache stats count files correctly."""
        # Create fake cached files
        s3_dir = temp_cache_dir / "s3"
        s3_dir.mkdir()
        (s3_dir / "file1.txt").write_bytes(b"abc")
        (s3_dir / "file2.txt").write_bytes(b"12345")

        wandb_dir = temp_cache_dir / "wandb"
        wandb_dir.mkdir()
        (wandb_dir / "model.pt").write_bytes(b"1234567890")

        stats = get_cache_stats()

        assert stats["total_files"] == 3
        assert stats["total_size_bytes"] == 18  # 3 + 5 + 10

        assert "s3" in stats["protocols"]
        assert stats["protocols"]["s3"]["files"] == 2
        assert stats["protocols"]["s3"]["size_bytes"] == 8

        assert "wandb" in stats["protocols"]
        assert stats["protocols"]["wandb"]["files"] == 1
        assert stats["protocols"]["wandb"]["size_bytes"] == 10

    def test_cache_stats_ignores_metadata(self, temp_cache_dir):
        """Cache stats ignore .json metadata files."""
        protocol_dir = temp_cache_dir / "s3"
        protocol_dir.mkdir()

        # Create cache file and metadata
        (protocol_dir / "file.txt").write_bytes(b"content")
        (protocol_dir / "file.meta.json").write_text('{"uri": "test"}')

        stats = get_cache_stats()

        # Should only count the cache file, not the metadata
        assert stats["total_files"] == 1

    def test_clear_cache_specific_uri(self, temp_cache_dir, mock_smart_open):
        """Clear cache for specific URI."""
        # Cache a file
        uri = "s3://bucket/file.txt"
        path = get_cached_file_path(uri)
        assert Path(path).exists()

        # Clear it
        clear_cache(uri=uri)

        # Should be gone
        assert not Path(path).exists()

    def test_clear_cache_protocol(self, temp_cache_dir):
        """Clear all files for a protocol."""
        # Create files in multiple protocols
        s3_dir = temp_cache_dir / "s3"
        s3_dir.mkdir()
        (s3_dir / "file1.txt").write_bytes(b"test")
        (s3_dir / "file2.txt").write_bytes(b"test")

        http_dir = temp_cache_dir / "http"
        http_dir.mkdir()
        (http_dir / "file3.txt").write_bytes(b"test")

        # Clear only S3
        clear_cache(protocol="s3")

        # S3 files should be gone
        assert not (s3_dir / "file1.txt").exists()
        assert not (s3_dir / "file2.txt").exists()

        # HTTP files should remain
        assert (http_dir / "file3.txt").exists()

    def test_clear_entire_cache(self, temp_cache_dir):
        """Clear entire cache."""
        # Create files in multiple protocols
        s3_dir = temp_cache_dir / "s3"
        s3_dir.mkdir()
        (s3_dir / "file1.txt").write_bytes(b"test")

        http_dir = temp_cache_dir / "http"
        http_dir.mkdir()
        (http_dir / "file2.txt").write_bytes(b"test")

        # Clear everything
        clear_cache()

        # All should be gone
        assert not (s3_dir / "file1.txt").exists()
        assert not (http_dir / "file2.txt").exists()


class TestLocalFilePaths:
    """Test handling of local file paths."""

    def test_local_absolute_path_unchanged(self):
        """Absolute local paths are returned unchanged."""
        path = "/absolute/path/to/file.txt"
        result = get_cached_file_path(path)
        assert result == path

    def test_local_relative_path_resolved(self, tmp_path, monkeypatch):
        """Relative paths are resolved to absolute."""
        # Change to temp directory
        monkeypatch.chdir(tmp_path)

        # Create a file
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"content")

        # Get cached path with relative reference
        result = get_cached_file_path("test.txt")

        # Should be resolved to absolute
        assert Path(result).is_absolute()
        assert Path(result).exists()

    def test_package_prefixed_path_resolved(self):
        """Registered package prefixes use the shared path resolver."""
        result = get_cached_file_path("@holosoma/data/scene_objects/boxes/small_box.urdf")
        assert Path(result).is_absolute()
        assert Path(result).is_file()

    def test_local_files_not_cached(self, temp_cache_dir):
        """Local files don't get cached."""
        local_path = "/local/file.txt"
        result = get_cached_file_path(local_path)

        # Should return original path
        assert result == local_path

        # Cache dir should be empty
        stats = get_cache_stats()
        assert stats["total_files"] == 0


class TestIntegrationWithMocks:
    """Integration tests with mocked downloaders."""

    def test_s3_download_and_cache(self, mock_smart_open, temp_cache_dir):
        """S3 file is downloaded and cached."""
        uri = "s3://bucket/data.npz"

        # First call - should download
        path1 = get_cached_file_path(uri)
        assert Path(path1).exists()
        assert temp_cache_dir in Path(path1).parents

        # Second call - should use cache
        path2 = get_cached_file_path(uri)
        assert path1 == path2

    def test_wandb_download_and_cache(self, mock_wandb, temp_cache_dir):
        """W&B file is downloaded and cached."""
        uri = "wandb://entity/project/run_id/model.pt"

        # First call - should download
        path1 = get_cached_file_path(uri)
        assert Path(path1).exists()
        assert temp_cache_dir in Path(path1).parents

        # Second call - should use cache
        path2 = get_cached_file_path(uri)
        assert path1 == path2

    def test_http_download_and_cache(self, mock_smart_open, temp_cache_dir):
        """HTTP file is downloaded and cached."""
        uri = "http://example.com/file.txt"

        path1 = get_cached_file_path(uri)
        assert Path(path1).exists()

        path2 = get_cached_file_path(uri)
        assert path1 == path2

    def test_cached_open_context_manager(self, mock_smart_open, temp_cache_dir):
        """cached_open context manager works correctly."""
        uri = "s3://bucket/file.txt"

        with cached_open(uri, "rb") as f:
            content = f.read()

        assert content is not None
        assert len(content) > 0

    def test_cache_disabled_via_parameter(self, mock_smart_open, temp_cache_dir):
        """use_cache=False bypasses cache."""
        uri = "s3://bucket/file.txt"

        # Download with cache disabled
        path = get_cached_file_path(uri, use_cache=False)

        # Should be in temp location, not cache
        assert temp_cache_dir not in Path(path).parents

    def test_cache_disabled_via_env_var(self, mock_smart_open, temp_cache_dir, monkeypatch):
        """HOLOSOMA_CACHE_ENABLED=false disables caching."""
        monkeypatch.setenv("HOLOSOMA_CACHE_ENABLED", "false")

        uri = "s3://bucket/file.txt"
        path = get_cached_file_path(uri)

        # Should be in temp location, not cache
        assert temp_cache_dir not in Path(path).parents


class TestRaceConditionHandling:
    """Test race condition protection mechanisms."""

    def test_process_specific_temp_files(self, mock_smart_open, temp_cache_dir):
        """Temp files include process ID to avoid conflicts."""
        import os

        uri = "s3://bucket/file.txt"

        # Download (will use temp file internally)
        path = get_cached_file_path(uri)

        # Verify PID would be used in temp file name
        # (We can't directly test this without mocking deeper, but we verify
        # the final result exists and is correct)
        assert Path(path).exists()
        assert str(os.getpid()) not in str(path)  # Final path shouldn't have PID

    def test_double_check_pattern(self, mock_smart_open, temp_cache_dir):
        """Cache is checked again before download."""
        uri = "s3://bucket/file.txt"

        # First call caches the file
        path1 = get_cached_file_path(uri)

        # Second call should find it immediately without download attempt
        path2 = get_cached_file_path(uri)

        assert path1 == path2
        assert Path(path1).exists()

    def test_file_exists_error_handling(self, mock_smart_open, temp_cache_dir):
        """FileExistsError is handled gracefully."""
        uri = "s3://bucket/file.txt"

        # Pre-populate cache to simulate race condition
        cache_path = _get_cache_path(uri)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(b"pre-existing content")

        # Should use existing cache, not fail
        result = get_cached_file_path(uri)
        assert result == str(cache_path)
        assert Path(result).exists()


if __name__ == "__main__":
    # Allow running test file directly for debugging
    pytest.main([__file__, "-v"])
