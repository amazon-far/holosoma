"""MotionCommand's fixed-base guard at setup (no sim).

The term writes the clip's root pose into the robot root on every reset, which a welded base cannot follow, so
``setup`` raises before touching the simulator on a fixed base and goes on to the simulator on a floating one.
The unbound method runs against a stub term/env.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from holosoma.managers.command.terms.wbt import MotionCommand

pytestmark = pytest.mark.no_sim


class _PastTheGuardError(Exception):
    """Raised by the stub simulator on the first attribute ``setup`` reads after the guard."""


class _StubSimulator:
    @property
    def body_names(self) -> list[str]:
        raise _PastTheGuardError


def _term(*, floating_base: bool) -> Any:
    env = SimpleNamespace(
        robot_config=SimpleNamespace(asset=SimpleNamespace(fix_base_link=not floating_base)),
        num_envs=1,
        device="cpu",
        simulator=_StubSimulator(),
    )
    return SimpleNamespace(_env=env)


def test_setup_requires_floating_base() -> None:
    term = _term(floating_base=False)
    with pytest.raises(ValueError, match="requires a floating base"):
        MotionCommand.setup(term)
    assert not hasattr(term, "num_envs")  # nothing past the guard ran


def test_setup_proceeds_past_the_guard_for_a_floating_base() -> None:
    term = _term(floating_base=True)
    with pytest.raises(_PastTheGuardError):
        MotionCommand.setup(term)
    assert (term.num_envs, term.device) == (1, "cpu")  # the guard passed and setup reached the simulator
