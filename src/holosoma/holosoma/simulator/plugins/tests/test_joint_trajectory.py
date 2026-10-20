"""Scripted control through real lifecycle hooks, with no SDK transport or physics engine."""

from __future__ import annotations

from dataclasses import asdict, replace
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from holosoma.config_types.action import JointControllerCfg
from holosoma.config_types.plugin import JointTrajectoryPluginConfig, JointWaypoint
from holosoma.config_types.robot import RobotBridgeConfig
from holosoma.config_types.simulator import BridgeConfig
from holosoma.simulator.base_simulator.hooks import HookRegistry, Phase
from holosoma.simulator.plugins.joint_trajectory import JointTrajectoryPlugin
from tests.simulators._robot_presets import test_arm_fixed

pytestmark = pytest.mark.no_sim

DOFS = ["shoulder_joint", "elbow_joint", "slide_joint"]
TRAJECTORY = JointTrajectoryPluginConfig(
    waypoints=[
        JointWaypoint(time=1.0, targets={"#hinges": 1.0}),
        JointWaypoint(time=2.0, targets={"slide_joint": 0.04}),
        JointWaypoint(time=2.5),
    ],
    controller=JointControllerCfg(
        stiffness={"shoulder_joint": 20.0, "elbow_joint": 20.0, "slide_joint": 200.0},
        damping={"shoulder_joint": 1.0, "elbow_joint": 1.0, "slide_joint": 5.0},
    ),
)


class _Sim:
    def __init__(self, *, robot: Any = None, now: float = 0.0, num_envs: int = 1) -> None:
        self.robot_config = robot or replace(
            test_arm_fixed,
            joint_tags={"#hinges": ["shoulder_joint", "elbow_joint"]},
            init_state=replace(test_arm_fixed.init_state, default_joint_positions={"#hinges": 0.2}),
        )
        self.simulator_config = SimpleNamespace(bridge=BridgeConfig(enabled=False))
        self.dof_names = self.robot_config.dof_names
        self.num_dof = len(self.dof_names)
        self.dof_pos = torch.zeros(num_envs, self.num_dof)
        self.dof_vel = torch.zeros_like(self.dof_pos)
        self.now = now
        self.hooks = HookRegistry()
        self.writes: list[tuple[torch.Tensor, list[int] | None]] = []

    def time(self) -> float:
        return self.now

    def get_dof_limits_properties(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.zeros(3, 2), torch.zeros(3), torch.tensor([20.0, 20.0, 50.0])

    def apply_torques_at_dof(self, torques: torch.Tensor, dof_indices: list[int] | None = None) -> None:
        self.writes.append((torques.clone(), dof_indices))


def _plugin(
    cfg: JointTrajectoryPluginConfig = TRAJECTORY, sim: _Sim | None = None
) -> tuple[JointTrajectoryPlugin, _Sim]:
    sim = sim or _Sim()
    return JointTrajectoryPlugin(cfg, cast("Any", sim)), sim


@pytest.mark.parametrize(
    ("time", "target"),
    [
        (-1.0, [0.2, 0.2, 0.0]),
        (0.5, [0.6, 0.6, 0.0]),
        (1.0, [1.0, 1.0, 0.0]),
        (1.5, [1.0, 1.0, 0.02]),
        (2.25, [1.0, 1.0, 0.04]),
        (10.0, [1.0, 1.0, 0.04]),
    ],
)
def test_interpolation_and_hold(time: float, target: list[float]) -> None:
    plugin, _ = _plugin()
    torch.testing.assert_close(plugin.target_at(time), torch.tensor(target))


def test_empty_trajectory_holds_defaults_and_loop_restarts() -> None:
    hold, _ = _plugin(replace(TRAJECTORY, waypoints=[]))
    torch.testing.assert_close(hold.target_at(100), torch.tensor([0.2, 0.2, 0.0]))
    loop, _ = _plugin(replace(TRAJECTORY, loop=True))
    torch.testing.assert_close(loop.target_at(3.0), loop.target_at(0.5))
    torch.testing.assert_close(loop.target_at(2.5), loop.target_at(0.0))


def test_hook_applies_clipped_pd_torques_to_every_environment() -> None:
    plugin, sim = _plugin(sim=_Sim(now=100.0, num_envs=2))
    sim.dof_pos[:] = torch.tensor([[0.5, 0.5, 0.0], [-5.0, 0.0, 0.0]])
    sim.dof_vel[0, 1] = 1.0
    sim.now = 100.5
    assert sim.writes == []
    sim.hooks.emit(Phase.PRE_STEP)
    torques, indices = sim.writes[-1]
    assert indices is None
    torch.testing.assert_close(torques, torch.tensor([[2.0, 1.0, 0.0], [20.0, 12.0, 0.0]]))
    assert plugin.target_at(0.5).shape == (3,)


def test_selection_expands_and_deduplicates_before_ordering_torques() -> None:
    sim = _Sim()
    sim.robot_config = replace(
        sim.robot_config,
        joint_tags={"#chosen": ["slide_joint", "shoulder_joint", "slide_joint"]},
        init_state=replace(sim.robot_config.init_state, default_joint_positions={}),
    )
    cfg = JointTrajectoryPluginConfig(
        joints=("#chosen", "shoulder_joint"),
        waypoints=[JointWaypoint(time=1.0, targets={"#chosen": 0.03})],
        controller=TRAJECTORY.controller,
    )
    plugin, sim = _plugin(cfg, sim)
    sim.now = 1.0
    sim.hooks.emit(Phase.PRE_STEP)
    assert plugin.dof_indices == [2, 0]
    assert sim.writes[-1][1] == [2, 0]
    torch.testing.assert_close(sim.writes[-1][0], torch.tensor([[6.0, 0.6]]))


@pytest.mark.parametrize(
    "targets",
    [
        {"#hinges": 1.0, "elbow_joint": 1.0},
        {"elbow_joint": 1.0, "#hinges": 1.0},
    ],
)
def test_later_ambiguous_waypoint_fails_before_any_hook_or_write(targets: dict[str, float]) -> None:
    sim = _Sim()
    cfg = JointTrajectoryPluginConfig(
        waypoints=[JointWaypoint(time=1.0), JointWaypoint(time=2.0, targets=targets)],
        controller=TRAJECTORY.controller,
    )
    with pytest.raises(ValueError, match=r"(?s)#hinges.*elbow_joint|elbow_joint.*#hinges") as exc:
        _plugin(cfg, sim)
    assert "DOF 'elbow_joint'" in str(exc.value)
    sim.hooks.emit(Phase.PRE_STEP)
    assert sim.writes == []


def test_revalidates_mutable_targets_and_rejects_uncontrolled_joints() -> None:
    cfg = JointTrajectoryPluginConfig(
        waypoints=[JointWaypoint(time=1.0, targets={"slide_joint": 0.01})],
        controller=TRAJECTORY.controller,
    )
    cfg.waypoints[0].targets["slide_joint"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        _plugin(cfg)
    with pytest.raises(ValueError, match="uncontrolled DOFs.*slide_joint"):
        _plugin(replace(TRAJECTORY, joints=("#hinges",)))
    with pytest.raises(ValueError, match="#missing"):
        _plugin(
            JointTrajectoryPluginConfig(
                waypoints=[JointWaypoint(time=1.0, targets={"#missing": 0.0})],
                controller=TRAJECTORY.controller,
            )
        )


@pytest.mark.parametrize("times", [[0.0], [1.0, 1.0], [2.0, 1.0], [float("nan")], [float("inf")]])
def test_invalid_waypoint_times(times: list[float]) -> None:
    with pytest.raises(ValueError, match="finite, > 0 and strictly increasing"):
        JointTrajectoryPluginConfig(
            waypoints=[JointWaypoint(time=time) for time in times],
            controller=TRAJECTORY.controller,
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_targets(value: float) -> None:
    with pytest.raises(ValueError, match="targets must be finite"):
        JointWaypoint(time=1.0, targets={"shoulder_joint": value})


def test_explicit_and_tagged_waypoints_produce_identical_control() -> None:
    explicit = replace(
        TRAJECTORY,
        waypoints=[
            JointWaypoint(time=1.0, targets={"shoulder_joint": 1.0, "elbow_joint": 1.0}),
            *TRAJECTORY.waypoints[1:],
        ],
    )
    tagged, tagged_sim = _plugin()
    exact, exact_sim = _plugin(explicit)
    authored = asdict(TRAJECTORY)
    for time in (0.0, 0.25, 1.25, 2.25, 3.0):
        tagged_sim.now = exact_sim.now = time
        tagged_sim.hooks.emit(Phase.PRE_STEP)
        exact_sim.hooks.emit(Phase.PRE_STEP)
        torch.testing.assert_close(tagged_sim.writes[-1][0], exact_sim.writes[-1][0])
        torch.testing.assert_close(tagged.target_at(time), exact.target_at(time))
    assert asdict(TRAJECTORY) == authored


def test_waypoint_precedence_does_not_depend_on_key_spelling() -> None:
    cfg = JointTrajectoryPluginConfig(
        waypoints=[
            JointWaypoint(time=1.0, targets={"#hinges": 1.0}),
            JointWaypoint(time=2.0, targets={"elbow_joint": 0.5}),
            JointWaypoint(time=3.0, targets={"#hinges": -1.0}),
        ],
        controller=TRAJECTORY.controller,
    )
    plugin, _ = _plugin(cfg)
    torch.testing.assert_close(plugin.target_at(2), torch.tensor([1.0, 0.5, 0.0]))
    torch.testing.assert_close(plugin.target_at(3), torch.tensor([-1.0, -1.0, 0.0]))


def test_sdk_and_plugin_controllers_must_have_disjoint_joints() -> None:
    sim = _Sim()
    sim.simulator_config.bridge = BridgeConfig(enabled=True)
    with pytest.raises(ValueError, match="both control joints"):
        _plugin(sim=sim)
    sim.robot_config = replace(sim.robot_config, bridge=RobotBridgeConfig(controlled_dof_names=("elbow_joint",)))
    _plugin(
        JointTrajectoryPluginConfig(
            joints=("shoulder_joint",),
            controller=JointControllerCfg(
                stiffness={"shoulder_joint": 20.0},
                damping={"shoulder_joint": 1.0},
            ),
        ),
        sim,
    )
    sim.hooks.emit(Phase.PRE_STEP)
    assert sim.writes[-1][1] == [0]


def test_plugin_registry_owns_the_demo() -> None:
    from holosoma.config_values.plugin import PLUGIN_REGISTRY

    cfg = PLUGIN_REGISTRY["yam-trajectory"]
    assert cfg.get_cls() is JointTrajectoryPlugin
    assert cfg.loop and cfg.waypoints[-1].time == 10.75


def test_documented_yam_demo_command_disables_sdk_bridge() -> None:
    from holosoma.config_types.run_sim import RunSimConfig
    from holosoma.utils.config_registry import parse_config

    config = parse_config(
        RunSimConfig,
        args=["robot:yam", "plugin.motion:yam-trajectory", "--simulator.config.bridge.enabled=False"],
    )
    assert config.robot.asset.robot_type == "yam"
    assert not config.simulator.config.bridge.enabled
    assert isinstance(config.plugin["motion"], JointTrajectoryPluginConfig)
