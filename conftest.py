"""Pytest configuration to ensure proper import order for isaacgym compatibility."""

# Import torch safely before any isaacgym imports during test collection
from holosoma.utils.safe_torch_import import torch  # noqa: F401


def mark_str(marker, msg):
    return f"{marker}: marks tests as requiring {msg} (deselect with '-m \"not {marker}\"')"


def pytest_configure(config):
    """Register custom markers for pytest."""
    config.addinivalue_line(
        "markers",
        mark_str("isaacsim", "Isaac Sim"),
    )
    config.addinivalue_line(
        "markers",
        mark_str("multi_gpu", "multiple GPUs"),
    )
    config.addinivalue_line(
        "markers",
        mark_str("requires_inference", "inference environment"),
    )
