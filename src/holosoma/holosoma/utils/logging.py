from __future__ import annotations

import logging
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from types import FrameType

from loguru import logger


class LoguruLoggingBridge(logging.Handler):
    """Bridge Python's standard logging to loguru.

    This handler redirects all standard logging calls to loguru,
    providing unified logging output.
    """

    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding loguru level
        level: str | int
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where the logged message originated
        frame: FrameType | None = logging.currentframe()
        depth = 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


class LoguruStream:
    def write(self, message: str) -> None:
        if message.strip():  # Only log non-empty messages
            logger.info(message.strip())  # Changed to debug level

    def flush(self) -> None:
        pass


@contextmanager
def capture_stdout_to_loguru() -> Iterator[None]:
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    loguru_stream = LoguruStream()
    old_stdout = sys.stdout
    sys.stdout = loguru_stream
    try:
        yield
    finally:
        sys.stdout = old_stdout
        logger.remove()
        console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
        logger.add(sys.stdout, level=console_log_level, colorize=True)
