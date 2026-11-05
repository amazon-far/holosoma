"""Boot every Isaac Sim harness subprocess with Kit's headless experience (Isaac Lab's ``HEADLESS``)."""

import pytest


@pytest.fixture(autouse=True)
def _headless_kit(monkeypatch):
    monkeypatch.setenv("HEADLESS", "1")
