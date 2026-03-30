"""Shared fixtures for input-provider tests."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest


def _make_policy(**overrides):
    """Build a minimal mock policy with all attributes providers touch."""
    p = MagicMock()
    p.lin_vel_command = np.array([[0.0, 0.0]])
    p.ang_vel_command = np.array([[0.0]])
    p.stand_command = np.array([[0]])
    p.base_height_command = np.array([[0.5]])
    p.desired_base_height = 0.5
    p.active_policy_index = 0
    p.model_paths = ["a.onnx", "b.onnx"]
    p.config = SimpleNamespace(
        task=SimpleNamespace(
            ros_cmd_vel_topic="cmd_vel",
            ros_state_input_topic="holosoma/state_input",
            velocity_input="keyboard",
            state_input="keyboard",
        )
    )
    for k, v in overrides.items():
        setattr(p, k, v)
    return p


@pytest.fixture
def policy():
    return _make_policy()


def _try_import_policies():
    """Try to import policy modules; skip tests if heavy deps are missing."""
    try:
        from holosoma_inference.policies.base import BasePolicy  # noqa: F401
        from holosoma_inference.policies.locomotion import LocomotionPolicy  # noqa: F401
        from holosoma_inference.policies.wbt import WholeBodyTrackingPolicy  # noqa: F401

        return True
    except (ImportError, ModuleNotFoundError):
        return False


_has_policies = _try_import_policies()
skip_policies = pytest.mark.skipif(not _has_policies, reason="Policy deps not installed")


def _try_import_dual_mode():
    try:
        from holosoma_inference.policies.dual_mode import DualModePolicy  # noqa: F401

        return True
    except (ImportError, ModuleNotFoundError):
        return False


_has_dual_mode = _try_import_dual_mode()
skip_dual_mode = pytest.mark.skipif(not _has_dual_mode, reason="DualMode deps not installed")


def _make_dual():
    """Build a DualModePolicy with mock policies, skipping __init__."""
    from holosoma_inference.policies.dual_mode import DualModePolicy

    dual = object.__new__(DualModePolicy)
    dual.primary = _make_policy()
    dual.secondary = _make_policy()
    dual.active = dual.primary
    dual.active_label = "primary"

    # Set up _command_provider mocks with a _mapping dict (new API)
    dual.primary._velocity_input = MagicMock()
    dual.secondary._velocity_input = MagicMock()
    dual.primary._command_provider = MagicMock()
    dual.primary._command_provider._mapping = {}
    dual.secondary._command_provider = MagicMock()
    dual.secondary._command_provider._mapping = {}

    dual._setup_command_intercept()
    return dual
