"""Unit tests for ``dual_mode._select_policy_class``.

Covers the resolution order added for the unified service node:

1. ``config.task.policy_type`` → ``holosoma.policies.by_type`` entry-point group
   (the mechanism extension WBT presets use, so the service node resolves them
   identically to the standalone ``run_policy`` path).
2. The ``robot_type``-keyed groups + ``motion_command`` observation heuristic
   (unchanged core behaviour: WBT vs locomotion).

All lookups are by string against entry-point groups, so these tests stub the
``entry_points`` factory rather than installing any package.
"""

from types import SimpleNamespace
from unittest import mock

from holosoma_inference.policies import dual_mode
from holosoma_inference.policies.locomotion import LocomotionPolicy
from holosoma_inference.policies.wbt import WholeBodyTrackingPolicy


def _config(*, robot_type="g1_29dof", actor_obs=None, policy_type=None):
    """Build a duck-typed stand-in for InferenceConfig (no pydantic construction).

    ``policy_type`` is omitted from ``task`` when None so we also exercise the
    ``getattr(config.task, "policy_type", None)`` path for core TaskConfigs that
    do not define the field.
    """
    task = SimpleNamespace()
    if policy_type is not None:
        task.policy_type = policy_type
    obs = SimpleNamespace(obs_dict={"actor_obs": list(actor_obs or [])})
    robot = SimpleNamespace(robot_type=robot_type)
    return SimpleNamespace(task=task, robot=robot, observation=obs)


class _FakeEP:
    def __init__(self, name, cls):
        self.name = name
        self._cls = cls

    def load(self):
        return self._cls


class _SentinelPolicy:
    """Stand-in for an extension policy class registered under by_type."""


def _patch_entry_points(groups):
    """Patch the entry_points factory to return the given {group: [EP, ...]}.

    ``_select_policy_class`` does ``from holosoma_inference.compat import entry_points``
    *inside* the function, so the patch must target the source module, not a name bound
    on ``dual_mode``.
    """

    def _fake(group):
        return groups.get(group, [])

    return mock.patch("holosoma_inference.compat.entry_points", _fake)


def test_policy_type_resolves_via_by_type():
    """An explicit policy_type resolves through the by_type entry-point group."""
    groups = {"holosoma.policies.by_type": [_FakeEP("ext_custom_type", _SentinelPolicy)]}
    cfg = _config(policy_type="ext_custom_type", actor_obs=["motion_command"])
    with _patch_entry_points(groups):
        assert dual_mode._select_policy_class(cfg) is _SentinelPolicy


def test_unregistered_policy_type_falls_through_to_heuristic():
    """policy_type set but not registered → fall through (not a hard failure)."""
    cfg = _config(policy_type="not_registered", actor_obs=["motion_command"])
    with _patch_entry_points({"holosoma.policies.by_type": []}):
        # motion_command + no robot_type WBT registration → core WBT.
        assert dual_mode._select_policy_class(cfg) is WholeBodyTrackingPolicy


def test_no_policy_type_motion_command_is_core_wbt():
    """No policy_type field, motion_command obs → core WholeBodyTrackingPolicy."""
    cfg = _config(actor_obs=["motion_command"])  # task has no policy_type attr
    with _patch_entry_points({}):
        assert dual_mode._select_policy_class(cfg) is WholeBodyTrackingPolicy


def test_no_policy_type_no_motion_command_is_locomotion():
    """No policy_type, no motion_command obs → core LocomotionPolicy."""
    cfg = _config(actor_obs=["base_lin_vel"])
    with _patch_entry_points({}):
        assert dual_mode._select_policy_class(cfg) is LocomotionPolicy


def test_robot_type_wbt_group_takes_precedence_over_core_default():
    """A robot_type-keyed wbt registration is used when policy_type is absent."""
    groups = {"holosoma.policies.wbt": [_FakeEP("g1_29dof", _SentinelPolicy)]}
    cfg = _config(robot_type="g1_29dof", actor_obs=["motion_command"])
    with _patch_entry_points(groups):
        assert dual_mode._select_policy_class(cfg) is _SentinelPolicy


def test_policy_type_precedence_over_robot_type_groups():
    """policy_type (group by_type) wins over the robot_type-keyed wbt group."""
    groups = {
        "holosoma.policies.by_type": [_FakeEP("ext_custom_type", _SentinelPolicy)],
        "holosoma.policies.wbt": [_FakeEP("g1_29dof", WholeBodyTrackingPolicy)],
    }
    cfg = _config(robot_type="g1_29dof", actor_obs=["motion_command"], policy_type="ext_custom_type")
    with _patch_entry_points(groups):
        assert dual_mode._select_policy_class(cfg) is _SentinelPolicy
