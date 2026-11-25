"""BaseTask creates name resolvers before managers and rejects a DOF count mismatch."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from holosoma.envs.base_task import base_task as base_task_module
from holosoma.envs.base_task.base_task import BaseTask

pytestmark = pytest.mark.no_sim

BODY_NAMES = ["pelvis", "left_hip_link", "right_hip_link", "left_foot", "right_foot", "torso"]
DOF_NAMES = ["left_hip", "left_knee", "right_hip", "right_knee", "waist"]
# The simulator's body order deliberately differs from body_names so the two cannot be confused.
SIM_BODY_ORDER = ["torso", "pelvis", "right_foot", "left_foot", "right_hip_link", "left_hip_link"]

BODY_TAGS = {
    "#feet": ["left_foot", "right_foot"],
    "#hips": ["left_hip_link", "right_hip_link"],
    "#legs": ["#hips", "#feet"],
}
JOINT_TAGS = {
    "#left_leg": ["left_hip", "left_knee"],
    "#right_leg": ["right_hip", "right_knee"],
    "#legs": ["#left_leg", "#right_leg"],
}


class _Simulator:
    def __init__(self) -> None:
        self.lookups: list[str] = []

    def find_rigid_body_indice(self, body_name: str) -> int:
        self.lookups.append(body_name)
        return BODY_NAMES.index(body_name)


# --------------------------------------------------------------------------- the real constructor


class _ConstructorSimulator(_Simulator):
    """The lookup stub plus the setup surface ``_initialize_base_task`` calls before the first manager."""

    def __init__(self, **_kwargs: object) -> None:
        super().__init__()
        self.sim_dt = 0.005
        self.hooks = SimpleNamespace()
        self.simulator_config = SimpleNamespace(reset_manager=object())
        self.num_dof, self.num_bodies = len(DOF_NAMES), len(BODY_NAMES)
        self.dof_names, self.body_names = list(DOF_NAMES), list(BODY_NAMES)

    def set_headless(self, _headless: bool) -> None:
        pass

    def setup(self) -> None:
        pass

    def setup_terrain(self) -> None:
        pass

    def load_assets(self) -> None:
        pass

    def get_dof_limits_properties(self) -> tuple[object, object, object]:
        return object(), object(), object()

    def prepare_sim(self) -> None:
        pass

    def render_sensors(self, env_ids: object = None, num_renders: object = None) -> None:
        pass

    def register_mount_poses(self) -> None:
        pass

    def install_plugins(self) -> None:
        pass

    def close(self) -> None:
        pass


def _constructor_config(robot_dof_names: list[str]) -> Any:
    return SimpleNamespace(
        device="cpu",
        observation=object(),
        simulator=SimpleNamespace(
            _target_="test.Simulator",
            config=SimpleNamespace(
                name="mujoco", sim=SimpleNamespace(fps=200, control_decimation_steps=4, max_episode_length_s=20.0)
            ),
        ),
        terrain=object(),
        robot=SimpleNamespace(
            dof_names=robot_dof_names,
            body_names=BODY_NAMES,
            body_tags=BODY_TAGS,
            joint_tags=JOINT_TAGS,
            init_state=SimpleNamespace(root_state_list=lambda: [0.0] * 13),
        ),
        action=object(),
        reward=object(),
        termination=object(),
        randomization=object(),
        command=object(),
        curriculum=object(),
        training=SimpleNamespace(num_envs=1, seed=0, headless=True, torch_deterministic=False),
        logger=object(),
        scene=object(),
        sensors={},
        plugin={},
    )


def _wire_constructor(monkeypatch: pytest.MonkeyPatch, first_manager: Any) -> None:
    """Run ``BaseTask.__init__`` against the stub simulator, the real ``_load_assets`` and ``first_manager`` as the
    observation manager; ``_create_envs`` (simulator actor creation) is a no-op."""
    monkeypatch.setattr(base_task_module, "get_class", lambda _target: _ConstructorSimulator)
    monkeypatch.setattr(base_task_module, "FullSimConfig", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(base_task_module, "TerrainManager", lambda *_args: SimpleNamespace())
    monkeypatch.setattr(base_task_module, "RandomizationManager", first_manager)
    monkeypatch.setattr(BaseTask, "_create_envs", lambda _task: None)
    monkeypatch.setattr("holosoma.envs.base_task.base_task.get_timestamp", lambda: "timestamp")
    monkeypatch.setattr("holosoma.envs.base_task.base_task.get_experiment_dir", lambda *_args, **_kwargs: ".")


def test_resolvers_are_usable_by_the_first_manager_through_the_real_constructor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_initialize_base_task`` must create the resolve caches before any manager is built.

    This test uses the real constructor with a stub simulator and resolves from the first manager.
    """

    class _StopAfterResolveError(Exception):
        pass

    resolved: dict[str, Any] = {}

    def first_manager(_cfg: object, task: BaseTask, _device: str) -> None:
        resolved["bodies"] = task.robot_names.resolve_body_indices(["pelvis", "#feet"]).tolist()
        resolved["joints"] = task.robot_names.resolve_joint_indices(["#legs"]).tolist()
        resolved["dim_actions"] = task.dim_actions
        raise _StopAfterResolveError

    _wire_constructor(monkeypatch, first_manager)

    with pytest.raises(_StopAfterResolveError):
        BaseTask(_constructor_config(list(DOF_NAMES)))

    assert resolved == {
        "bodies": [BODY_NAMES.index(n) for n in ("pelvis", "left_foot", "right_foot")],
        "joints": [0, 1, 2, 3],
        "dim_actions": len(DOF_NAMES),  # one action per simulator DOF
    }


def test_simulator_dof_count_must_equal_robot_dof_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """``dim_actions`` comes from the simulator DOF count; a config listing a different number of DOFs fails
    loudly in ``_load_assets``, before any manager sizes the policy."""

    def no_manager(*_args: object) -> None:
        pytest.fail("no manager may be built after the DOF-count mismatch")

    _wire_constructor(monkeypatch, no_manager)

    with pytest.raises(AssertionError, match=r"Simulator reports 5 DOFs but robot_config.dof_names has 6"):
        BaseTask(_constructor_config([*DOF_NAMES, "tail"]))
