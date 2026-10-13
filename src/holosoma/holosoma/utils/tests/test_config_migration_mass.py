"""Legacy mass selections through checkpoint loading and the production mass consumer."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
import yaml
from loguru import logger

from holosoma.config_types.experiment import ExperimentConfig
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.managers.randomization.terms.locomotion import randomize_mass_startup
from holosoma.simulator.base_simulator.base_simulator import BaseSimulator
from holosoma.utils.config_migration import migrate_legacy_experiment_config
from holosoma.utils.eval_utils import CheckpointConfig, load_saved_experiment_config
from holosoma.utils.robot_name_resolver import RobotNameResolver
from holosoma.utils.robot_names import configure_robot_names
from holosoma.utils.sampler import STAGE_SETUP, TermSampler
from holosoma.utils.simulator_config import SimulatorType

pytest.importorskip("mujoco")

pytestmark = pytest.mark.no_sim


class _SimulatorTransport(BaseSimulator):
    body_ids: list[int]

    def __init__(self, config: ExperimentConfig) -> None:
        self.robot_config = config.robot
        # Resolve real canonical/native mappings; a different native order catches positional shortcuts.
        configure_robot_names(self, config.robot.dof_names, list(reversed(config.robot.body_names)))
        self.sim_device = "cpu"
        self.num_envs = 2
        count = len(config.robot.body_names)
        bridge = SimpleNamespace(
            body_mass=torch.arange(10, 10 + count, dtype=torch.float64).repeat(2, 1),
            body_inertia=torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64).repeat(2, count, 1),
        )
        self.backend = SimpleNamespace(
            warp_model_bridge=bridge,
            mjw_model=SimpleNamespace(_expanded_fields={"body_mass", "body_inertia"}),
        )

    @property
    def body_names(self) -> list[str]:
        return self.robot_config.body_names

    def get_simulator_type(self) -> SimulatorType:
        return SimulatorType.MUJOCO


def _mass_params(config: ExperimentConfig) -> dict[str, Any]:
    assert config.randomization is not None
    return config.randomization.setup_terms["mass_randomizer"].params


def _run_mass(config: ExperimentConfig) -> tuple[_SimulatorTransport, tuple[torch.Tensor, torch.Tensor]]:
    simulator = _SimulatorTransport(config)
    before = (
        simulator.backend.warp_model_bridge.body_mass.clone(),
        simulator.backend.warp_model_bridge.body_inertia.clone(),
    )
    env = BaseTask.__new__(BaseTask)
    env.robot_config = config.robot
    env.simulator = cast("BaseSimulator", simulator)
    env.body_names = list(config.robot.body_names)
    env.num_envs, env.device = 2, "cpu"
    env.robot_names = RobotNameResolver(config.robot, "cpu")
    randomize_mass_startup(
        env,
        torch.tensor([1]),
        sampler=TermSampler.bind(0, "migration_mass", STAGE_SETUP),
        **_mass_params(config),
    )
    return simulator, before


def _load(raw: dict[str, Any], tmp_path: Path) -> ExperimentConfig:
    path = tmp_path / "model.pt"
    torch.save({"experiment_config": raw}, path)
    config, _ = load_saved_experiment_config(CheckpointConfig(checkpoint=str(path)))
    return config


def _legacy(link_names: list[str]) -> dict[str, Any]:
    raw: dict[str, Any] = yaml.safe_load(
        (Path(__file__).parent / "data" / "legacy_g1_29dof_experiment_config.yaml").read_text()
    )
    raw["robot"]["randomize_link_body_names"] = link_names
    raw["randomization"]["setup_terms"]["mass_randomizer"]["params"] = {
        "enable_link_mass": True,
        "enable_base_mass": True,
        "link_mass_range": [2.0, 2.0],
        "added_mass_range": [2.0, 2.0],
    }
    return raw


@pytest.fixture
def migration_warnings() -> Iterator[list[str]]:
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(message.record["message"]), level="WARNING")
    yield messages
    logger.remove(sink)


def _assert_mass_changes(
    simulator: _SimulatorTransport,
    before: tuple[torch.Tensor, torch.Tensor],
    *,
    scale_body: str | None = None,
    add_base: bool = True,
) -> None:
    body_id_by_name = dict(zip(simulator.body_names, simulator.body_ids, strict=False))
    base_id = body_id_by_name["torso_link"]
    scale_id = body_id_by_name[scale_body] if scale_body else None
    bridge = simulator.backend.warp_model_bridge
    assert torch.equal(bridge.body_mass[0], before[0][0])
    assert torch.equal(bridge.body_inertia[0], before[1][0])
    for index, old in enumerate(before[0][1]):
        expected_mass = old * (2.0 if index == scale_id else 1.0)
        if add_base and index == base_id:
            expected_mass += 2.0
        assert bridge.body_mass[1, index] == pytest.approx(float(expected_mass))
        ratio = expected_mass / old
        assert torch.allclose(bridge.body_inertia[1, index], before[1][1, index] * ratio)


@pytest.mark.parametrize("link_names", [[], ["stale_link"]], ids=["empty", "all-stale"])
@pytest.mark.parametrize("explicit_enable", [False, True], ids=["default-enable", "explicit-enable"])
def test_empty_legacy_link_selection_keeps_base_mass_after_checkpoint_load(
    tmp_path: Path, migration_warnings: list[str], link_names: list[str], explicit_enable: bool
) -> None:
    raw = _legacy(link_names)
    if not explicit_enable:
        raw["randomization"]["setup_terms"]["mass_randomizer"]["params"].pop("enable_link_mass")
    config = _load(raw, tmp_path)
    params = _mass_params(config)

    simulator, before = _run_mass(config)

    assert params["link_bodies"] == [] and params["enable_link_mass"] is False
    assert params["enable_base_mass"] is True and params["added_mass_range"] == [2.0, 2.0]
    _assert_mass_changes(simulator, before)
    assert any("no bodies" in message and "enable_link_mass=False" in message for message in migration_warnings)
    dumped = config.to_serializable_dict()
    assert migrate_legacy_experiment_config(dumped) == dumped


@pytest.mark.parametrize("disabled", ["enabled", "enable_link_mass", "enable_base_mass"])
def test_empty_legacy_selection_preserves_disabled_operations(tmp_path: Path, disabled: str) -> None:
    raw = _legacy([])
    params = raw["randomization"]["setup_terms"]["mass_randomizer"]["params"]
    params[disabled] = False
    # Empty/disabled operations must not start validating or drawing their unused distributions.
    params["link_mass_range"] = [2.0, 1.0]
    if disabled in ("enabled", "enable_base_mass"):
        params["added_mass_range"] = [2.0, 1.0]
    config = _load(raw, tmp_path)

    simulator, before = _run_mass(config)

    assert _mass_params(config)[disabled] is False
    _assert_mass_changes(simulator, before, add_base=disabled == "enable_link_mass")


@pytest.mark.parametrize("link_bodies", [[], ["pelvis"]], ids=["explicit-empty", "explicit-nonempty"])
@pytest.mark.parametrize("enable_link_mass", [False, True], ids=["disabled", "enabled"])
def test_explicit_migrated_link_params_win_over_empty_legacy_selection(
    tmp_path: Path, migration_warnings: list[str], link_bodies: list[str], enable_link_mass: bool
) -> None:
    raw = _legacy([])
    params = raw["randomization"]["setup_terms"]["mass_randomizer"]["params"]
    params.update(link_bodies=link_bodies, enable_link_mass=enable_link_mass, base_body="torso_link")
    config = _load(raw, tmp_path)
    assert _mass_params(config) == params
    assert not any("no bodies" in message and "enable_link_mass=False" in message for message in migration_warnings)

    if enable_link_mass and not link_bodies:
        with pytest.raises(ValueError, match="enable_link_mass=True requires link_bodies"):
            _run_mass(config)
    else:
        simulator, before = _run_mass(config)
        _assert_mass_changes(simulator, before, scale_body="pelvis" if enable_link_mass else None)


@pytest.mark.parametrize("obsolete_object", [False, True], ids=["current", "unrelated-obsolete-field"])
@pytest.mark.parametrize("selection_present", [False, True], ids=["missing-selection", "empty-selection"])
def test_current_empty_link_selection_remains_strict(
    tmp_path: Path, obsolete_object: bool, selection_present: bool
) -> None:
    raw = ExperimentConfig(**migrate_legacy_experiment_config(_legacy(["pelvis"]))).to_serializable_dict()
    params = raw["randomization"]["setup_terms"]["mass_randomizer"]["params"]
    if selection_present:
        params["link_bodies"] = []
    else:
        params.pop("link_bodies")
    if obsolete_object:
        raw["robot"]["object"] = {}
    config = _load(raw, tmp_path)
    loaded_params = _mass_params(config)
    assert loaded_params["enable_link_mass"] is True
    assert loaded_params["enable_base_mass"] is True
    if selection_present:
        assert loaded_params == params
    with pytest.raises(ValueError, match="enable_link_mass=True requires link_bodies"):
        _run_mass(config)
