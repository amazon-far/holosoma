"""Apply a scripted joint-space PD controller through the simulator lifecycle hooks."""

from __future__ import annotations

from bisect import bisect_right
from typing import TYPE_CHECKING

from holosoma.config_types.plugin import validate_waypoint_times
from holosoma.simulator.base_simulator.hooks import Phase
from holosoma.utils.name_tags import expand_joint_value_map, expand_name_spec
from holosoma.utils.safe_torch_import import torch

if TYPE_CHECKING:
    from holosoma.config_types.plugin import JointTrajectoryPluginConfig
    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator


class JointTrajectoryPlugin:
    """Interpolate joint targets and apply PD torques before every physics step.

    This controller has no SDK transport. The complete trajectory is validated before any
    execution buffers or hooks are installed, and each waypoint overrides the preceding pose.
    """

    def __init__(self, cfg: JointTrajectoryPluginConfig, simulator: BaseSimulator) -> None:
        validate_waypoint_times(cfg.waypoints)
        self.cfg = cfg
        self.simulator = simulator
        robot = simulator.robot_config
        names = expand_name_spec(cfg.joints, robot.joint_tags, robot.dof_names, kind="joint")
        self.dof_indices = [robot.dof_names.index(name) for name in names]
        expanded = [
            expand_joint_value_map(
                waypoint.targets,
                robot.joint_tags,
                robot.dof_names,
                what=f"joint_trajectory waypoint at t={waypoint.time}",
                required=False,
            )
            for waypoint in cfg.waypoints
        ]
        for waypoint, targets in zip(cfg.waypoints, expanded):
            outside = [name for name in targets if name not in names]
            if outside:
                raise ValueError(
                    f"joint_trajectory waypoint at t={waypoint.time} targets uncontrolled DOFs {outside}; "
                    f"selected joints: {names}."
                )
        if simulator.simulator_config.bridge.enabled:
            bridge_names = robot.bridge.controlled_dof_names or tuple(
                name for name in robot.dof_names if name not in robot.bridge.excluded_dof_names
            )
            overlap = [name for name in names if name in bridge_names]
            if overlap:
                raise ValueError(f"JointTrajectoryPlugin and the SDK bridge both control joints {overlap}.")

        current = robot.default_dof_positions()
        self._times = [0.0]
        rows = [[current[name] for name in names]]
        for waypoint, targets in zip(cfg.waypoints, expanded):
            current.update(targets)
            self._times.append(waypoint.time)
            rows.append([current[name] for name in names])
        dof_pos = simulator.dof_pos[:, self.dof_indices]
        device = dof_pos.device
        self._targets = torch.tensor(rows, dtype=dof_pos.dtype, device=device)
        self._period = self._times[-1] if cfg.loop and cfg.waypoints else None
        self._t0 = simulator.time()
        kp, kd = cfg.controller.resolve_gains(robot, required_dof_names=names)
        self.kp = torch.tensor([kp[i] for i in self.dof_indices], device=device)
        self.kd = torch.tensor([kd[i] for i in self.dof_indices], device=device)
        self.torque_limits = simulator.get_dof_limits_properties()[2][self.dof_indices]
        self._apply_indices = None if self.dof_indices == list(range(simulator.num_dof)) else self.dof_indices
        simulator.hooks.add(Phase.PRE_STEP, self.apply, name="joint_trajectory.apply")

    def target_at(self, elapsed: float) -> torch.Tensor:
        """Joint targets in the plugin's selected order, at seconds from installation."""
        t = elapsed % self._period if self._period else elapsed
        right = bisect_right(self._times, t)
        if right == 0:
            return self._targets[0]
        if right == len(self._times):
            return self._targets[-1]
        alpha = (t - self._times[right - 1]) / (self._times[right] - self._times[right - 1])
        return torch.lerp(self._targets[right - 1], self._targets[right], alpha)

    def apply(self) -> None:
        """Write torques for all environments, affecting only this controller's selected joints."""
        sim = self.simulator
        targets = self.target_at(sim.time() - self._t0)
        torques = self.kp * (targets - sim.dof_pos[:, self.dof_indices]) - self.kd * sim.dof_vel[:, self.dof_indices]
        if self.cfg.controller.clip_torques:
            torques = torch.clamp(torques, -self.torque_limits, self.torque_limits)
        sim.apply_torques_at_dof(torques, self._apply_indices)
