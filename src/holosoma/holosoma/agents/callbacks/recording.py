"""Eval callback that records per-step trajectory data to an NPZ file.

Records joint positions, velocities, torques, body poses, and root state
for later visualization with viser_eval_viewer.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from holosoma.agents.callbacks.base_callback import RLEvalCallback
from holosoma.config_types.eval_callback import RecordingConfig
from holosoma.utils.safe_torch_import import torch


class EvalRecordingCallback(RLEvalCallback):
    """Records per-step data during evaluation and saves to .npz on completion."""

    def __init__(
        self,
        config: RecordingConfig,
        training_loop: Any = None,
    ):
        super().__init__(config, training_loop)
        self.env_id = config.env_id

        output_path = config.output_path
        if not output_path.endswith(".npz"):
            output_path += ".npz"
        if training_loop is not None and hasattr(training_loop, "log_dir"):
            output_path = str(Path(training_loop.log_dir) / output_path)
        self.output_path = output_path

        self._buffers: dict[str, list[np.ndarray]] = {}
        self._metadata: dict[str, Any] = {}
        self._step_count = 0

    def _get_env(self):
        """Get the unwrapped BaseTask environment."""
        return self.training_loop._unwrap_env()

    def _save(self) -> None:
        """Save recorded data to NPZ."""
        if self._step_count == 0:
            return

        arrays: dict[str, np.ndarray] = {}
        for name, values in self._buffers.items():
            if values:
                arrays[name] = np.stack(values, axis=0)

        arrays["_metadata_json"] = np.array(json.dumps(self._metadata))

        path = Path(self.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(path), **arrays)

        channel_summary = ", ".join(
            f"{name}{list(arr.shape)}" for name, arr in arrays.items() if name != "_metadata_json"
        )
        logger.info(f"EvalRecordingCallback: saved {self._step_count} steps to {path}\n  Channels: {channel_summary}")

    def on_pre_evaluate_policy(self) -> None:
        env = self._get_env()
        sim = env.simulator

        self._metadata["dt"] = float(env.dt)
        self._metadata["fps"] = round(1.0 / float(env.dt))
        self._metadata["sim_dt"] = float(env.sim_dt)
        self._metadata["sim_fps"] = round(1.0 / float(env.sim_dt))
        self._metadata["control_decimation"] = env.simulator.simulator_config.sim.control_decimation_steps
        self._metadata["env_id"] = self.env_id
        if hasattr(sim, "dof_names"):
            self._metadata["dof_names"] = list(sim.dof_names)
        if hasattr(sim, "body_names"):
            self._metadata["body_names"] = list(sim.body_names)

        # Static robot properties
        robot_cfg = env.robot_config
        self._metadata["effort_limits"] = list(robot_cfg.dof_effort_limit_list)
        self._metadata["dof_pos_lower_limits"] = list(robot_cfg.dof_pos_lower_limit_list)
        self._metadata["dof_pos_upper_limits"] = list(robot_cfg.dof_pos_upper_limit_list)
        self._metadata["velocity_limits"] = list(robot_cfg.dof_vel_limit_list)
        asset_cfg = robot_cfg.asset
        self._metadata["urdf_path"] = str(Path(asset_cfg.asset_root) / asset_cfg.urdf_file)

        channel_names = [
            "dof_pos_target",
            "dof_pos",
            "dof_vel",
            "torques",
            "torques_substep",
            "dof_pos_substep",
            "dof_vel_substep",
            "actions",
            "root_pos",
            "root_quat_xyzw",
            "root_lin_vel",
            "root_ang_vel",
            "body_pos_w",
            "body_quat_xyzw",
            "commanded_velocity",
        ]
        for name in channel_names:
            self._buffers[name] = []

        logger.info(f"EvalRecordingCallback: recording env_id={self.env_id}, output={self.output_path}")

    def on_post_eval_env_step(self, actor_state: dict) -> dict:
        env = self._get_env()
        sim = env.simulator
        eid = self.env_id

        def _to_np(t: torch.Tensor) -> np.ndarray:
            return t.detach().cpu().numpy().copy()

        self._buffers["dof_pos"].append(_to_np(sim.dof_pos[eid]))  # post_eval_env_step, so after 4 decimation
        self._buffers["dof_vel"].append(_to_np(sim.dof_vel[eid]))
        self._buffers["torques"].append(
            _to_np(self._extract_torques(env, eid))
        )  # pre_eval_env_step, so the torques is the last decimation

        # robot_root_states: [num_envs, 13] = pos(3), quat_xyzw(4), lin_vel(3), ang_vel(3)
        root = sim.robot_root_states[eid]
        self._buffers["root_pos"].append(_to_np(root[:3]))
        self._buffers["root_quat_xyzw"].append(_to_np(root[3:7]))
        self._buffers["root_lin_vel"].append(_to_np(root[7:10]))
        self._buffers["root_ang_vel"].append(_to_np(root[10:13]))

        self._buffers["body_pos_w"].append(_to_np(sim._rigid_body_pos[eid]))
        self._buffers["body_quat_xyzw"].append(_to_np(sim._rigid_body_rot[eid]))

        # substep tensors: [decimation, num_dof] — one row per physics sub-step
        torques_substep, dof_pos_substep, dof_vel_substep = self._extract_substep_data(env, eid)
        self._buffers["torques_substep"].append(_to_np(torques_substep))
        self._buffers["dof_pos_substep"].append(_to_np(dof_pos_substep))
        self._buffers["dof_vel_substep"].append(_to_np(dof_vel_substep))

        if "actions" in actor_state and actor_state["actions"] is not None:
            self._buffers["actions"].append(_to_np(actor_state["actions"][eid]))

        # Record desired target joint positions (PD setpoint)
        self._buffers["dof_pos_target"].append(_to_np(self._extract_dof_pos_target(env, eid)))

        # Record commanded velocity [lin_vel_x, lin_vel_y, ang_vel_yaw]
        if hasattr(env, "command_manager") and env.command_manager is not None:
            try:
                self._buffers["commanded_velocity"].append(_to_np(env.command_manager.commands[eid]))
            except (AttributeError, IndexError):
                pass

        self._step_count += 1
        return actor_state

    def _extract_dof_pos_target(self, env: Any, env_id: int) -> torch.Tensor:
        """Extract desired target joint positions from the action manager's joint control term.

        The PD target is: actions_after_delay * action_scales + default_dof_pos.
        Returns shape [num_dof].
        """
        for _term_name, term in env.action_manager.iter_terms():
            if hasattr(term, "_actions_after_delay") and hasattr(term, "action_scales"):
                return term._actions_after_delay[env_id] * term.action_scales + env.default_dof_pos[env_id]
        raise RuntimeError("No action term with _actions_after_delay found")

    def _extract_torques(self, env: Any, env_id: int) -> torch.Tensor:
        """Extract torques from the action manager's joint control term.

        Returns torques, shape [num_dof].
        """
        for _term_name, term in env.action_manager.iter_terms():
            if hasattr(term, "torques"):
                return term.torques[env_id]
        raise RuntimeError("No action term with torques found")

    def _extract_substep_data(self, env: Any, env_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract sub-step torques, dof_pos, and dof_vel from the action manager's joint control term.

        Returns (torques_substep, dof_pos_substep, dof_vel_substep), each shape [decimation, num_dof].
        """
        for _term_name, term in env.action_manager.iter_terms():
            if hasattr(term, "torques_substep"):
                return term.torques_substep[env_id], term.dof_pos_substep[env_id], term.dof_vel_substep[env_id]
        raise RuntimeError("No action term with torques_substep found")

    def on_post_evaluate_policy(self) -> None:
        self._save()
