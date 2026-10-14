"""BaseTask.resolve_* : tag-aware body/joint name and index resolution with per-spec caches."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from holosoma.envs.base_task import base_task as base_task_module
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.utils.robot_name_resolver import RobotNameResolver
from holosoma.utils.safe_torch_import import torch

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


def _make_task(
    body_tags: dict[str, list[str]] | None = None, joint_tags: dict[str, list[str]] | None = None
) -> tuple[BaseTask, _Simulator]:
    # Built without __init__ (no simulator, no managers): the resolvers must not depend on setup state.
    task = BaseTask.__new__(BaseTask)
    simulator = _Simulator()
    task.robot_config = SimpleNamespace(  # type: ignore[assignment]
        body_names=BODY_NAMES,
        dof_names=DOF_NAMES,
        body_tags=BODY_TAGS if body_tags is None else body_tags,
        joint_tags=JOINT_TAGS if joint_tags is None else joint_tags,
    )
    task.body_names = list(BODY_NAMES)
    task.dof_names = list(DOF_NAMES)
    task.device = "cpu"
    task.simulator = simulator  # type: ignore[assignment]
    task.robot_names = RobotNameResolver(task.robot_config, task.device)
    return task, simulator


def test_resolve_body_indices_use_canonical_observation_order() -> None:
    task, simulator = _make_task()

    idx = task.robot_names.resolve_body_indices(["pelvis", "#feet"])

    assert idx.dtype == torch.long
    assert idx.device.type == "cpu"
    assert idx.tolist() == [BODY_NAMES.index(n) for n in ("pelvis", "left_foot", "right_foot")]
    assert simulator.lookups == []


def test_resolve_body_indices_cache_hit_returns_same_tensor() -> None:
    task, simulator = _make_task()

    first = task.robot_names.resolve_body_indices(["#feet"])
    second = task.robot_names.resolve_body_indices(("#feet",))

    assert second is first
    assert simulator.lookups == []


def test_resolve_body_names_composes_tags_and_dedupes() -> None:
    task, _ = _make_task()

    names = task.robot_names.resolve_body_names(["#legs", "left_foot", "pelvis"])

    assert names == ["left_hip_link", "right_hip_link", "left_foot", "right_foot", "pelvis"]


def test_resolve_body_names_returns_a_copy() -> None:
    task, _ = _make_task()

    task.robot_names.resolve_body_names(["#feet"]).append("torso")

    assert task.robot_names.resolve_body_names(["#feet"]) == ["left_foot", "right_foot"]


def test_unknown_body_tag_lists_defined_tags() -> None:
    task, _ = _make_task()

    with pytest.raises(
        ValueError, match=r"Unknown body tag '#wings'.*defined body tags: \['#feet', '#hips', '#legs'\]"
    ):
        task.robot_names.resolve_body_indices(["#wings"])


def test_unknown_body_name_lists_known_bodies() -> None:
    task, _ = _make_task()

    with pytest.raises(ValueError, match=r"Unknown body name 'tail'.*known bodies: \['pelvis'"):
        task.robot_names.resolve_body_names(["tail"])


def test_resolve_joint_indices_are_positions_in_dof_names() -> None:
    task, simulator = _make_task()

    idx = task.robot_names.resolve_joint_indices(["#legs", "waist"])

    assert idx.dtype == torch.long
    assert idx.tolist() == [0, 1, 2, 3, 4]
    assert task.robot_names.resolve_joint_indices(["#legs", "waist"]) is idx
    assert simulator.lookups == []


def test_unknown_joint_tag_lists_defined_tags() -> None:
    task, _ = _make_task()

    with pytest.raises(
        ValueError, match=r"Unknown joint tag '#arms'.*defined joint tags: \['#left_leg', '#legs', '#right_leg'\]"
    ):
        task.robot_names.resolve_joint_names(["#arms"])


def test_body_and_joint_caches_do_not_collide() -> None:
    task, _ = _make_task(body_tags={"#left": ["left_foot"]}, joint_tags={"#left": ["left_hip", "left_knee"]})

    assert task.robot_names.resolve_body_indices(["#left"]).tolist() == [BODY_NAMES.index("left_foot")]
    assert task.robot_names.resolve_joint_indices(["#left"]).tolist() == [0, 1]


def test_empty_spec_gives_empty_long_tensor() -> None:
    task, _ = _make_task()

    assert task.robot_names.resolve_body_indices([]).shape == (0,)
    assert task.robot_names.resolve_body_indices([]).dtype == torch.long
    assert task.robot_names.resolve_joint_indices([]).shape == (0,)


@pytest.mark.parametrize("resolver", ["resolve_body_names", "resolve_body_indices"])
def test_bare_string_body_spec_raises_type_error(resolver: str) -> None:
    """A term param ``bodies="pelvis"`` fails at the resolver instead of being iterated character by character."""
    task, simulator = _make_task()

    with pytest.raises(
        TypeError, match=r"body spec must be a list of names, got the string 'pelvis'; wrap it in a list: \['pelvis'\]"
    ):
        getattr(task.robot_names, resolver)("pelvis")

    assert simulator.lookups == []
    assert task.robot_names._cache == {}  # nothing is cached under the character-tuple key


@pytest.mark.parametrize("resolver", ["resolve_joint_names", "resolve_joint_indices"])
def test_bare_string_joint_spec_raises_type_error(resolver: str) -> None:
    task, _ = _make_task()

    with pytest.raises(
        TypeError, match=r"joint spec must be a list of names, got the string 'waist'; wrap it in a list"
    ):
        getattr(task.robot_names, resolver)("waist")

    assert task.robot_names._cache == {}


# --------------------------------------------------------------------------- the real constructor


class _ConstructorSimulator(_Simulator):
    """The lookup stub plus the setup surface ``_initialize_base_task`` calls before the first manager."""

    def __init__(self, **_kwargs: object) -> None:
        super().__init__()
        self.sim_dt = 0.005
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

    def install_plugins(self) -> None:
        pass

    def close(self) -> None:
        pass


def _constructor_config(robot_dof_names: list[str]) -> Any:
    return SimpleNamespace(
        observation=object(),
        simulator=SimpleNamespace(
            _target_="test.Simulator",
            config=SimpleNamespace(sim=SimpleNamespace(control_decimation_steps=4, max_episode_length_s=20.0)),
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
        training=SimpleNamespace(num_envs=1, seed=0, headless=True),
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
    monkeypatch.setattr(base_task_module, "ObservationManager", first_manager)
    monkeypatch.setattr(BaseTask, "_create_envs", lambda _task: None)
    monkeypatch.setattr("holosoma.utils.experiment_paths.get_timestamp", lambda: "timestamp")
    monkeypatch.setattr("holosoma.utils.experiment_paths.get_experiment_dir", lambda *_args, **_kwargs: ".")


def test_resolvers_are_usable_by_the_first_manager_through_the_real_constructor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_initialize_base_task`` must create the resolve caches before any manager is built.

    The fixtures above bypass ``__init__``; this drives the real constructor with a stub simulator and
    resolves from inside the first manager's constructor, then stops.
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
        BaseTask(_constructor_config(list(DOF_NAMES)), device="cpu")

    assert resolved == {
        "bodies": [BODY_NAMES.index(n) for n in ("pelvis", "left_foot", "right_foot")],
        "joints": [0, 1, 2, 3],
        "dim_actions": len(DOF_NAMES),  # one action per simulator DOF (replaced the actions_dim field)
    }


def test_simulator_dof_count_must_equal_robot_dof_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """``dim_actions`` comes from the simulator DOF count; a config listing a different number of DOFs fails
    loudly in ``_load_assets``, before any manager sizes the policy."""

    def no_manager(*_args: object) -> None:
        pytest.fail("no manager may be built after the DOF-count mismatch")

    _wire_constructor(monkeypatch, no_manager)

    with pytest.raises(AssertionError, match=r"Simulator reports 5 DOFs but robot_config.dof_names has 6"):
        BaseTask(_constructor_config([*DOF_NAMES, "tail"]), device="cpu")
