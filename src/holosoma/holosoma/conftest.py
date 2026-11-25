"""Shared fixtures for Holosoma tests."""

from collections.abc import Iterator

import pytest
from loguru import logger


@pytest.fixture
def warnings() -> Iterator[list[str]]:
    """Capture loguru warnings, which do not reach pytest's caplog."""
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(message.record["message"]), level="WARNING")
    try:
        yield messages
    finally:
        logger.remove(sink)
