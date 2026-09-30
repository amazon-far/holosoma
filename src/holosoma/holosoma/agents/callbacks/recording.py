"""Single recorder for all eval callback data.

Records all condition envs to NPZ with shape [T, num_conditions, ...].
When no sweep axes are registered, num_conditions defaults to 1
(equivalent to old single-env mode).

Other callbacks interact via the public API:
    - ``condition_manager`` — register sweep axes
    - ``ensure_finalized()`` — trigger deferred setup (idempotent)
    - ``register_buffer_key(name)`` — reserve a recording channel
    - ``append_buffer(name, data)`` — append one step of data
    - ``set_metadata(key, value)`` — add metadata to the NPZ
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from holosoma.agents.callbacks.base_callback import RLEvalCallback
from holosoma.agents.callbacks.grid_conditions import GridConditionManager
from holosoma.config_types.eval_callback import RecordingConfig
from holosoma.utils.safe_torch_import import torch


class EvalRecordingCallback(RLEvalCallback):
    """Records per-step data during evaluation and saves to .npz on completion.

    Owns buffers, metadata, condition manager, and NPZ output.
    """

    def __init__(
        self,
        config: RecordingConfig,
        training_loop: Any = None,
    ):
        super().__init__(config, training_loop)

        output_path = config.output_path
        if not output_path.endswith(".npz"):
            output_path += ".npz"
        if training_loop is not None and hasattr(training_loop, "log_dir"):
            output_path = str(Path(training_loop.log_dir) / output_path)
        self.output_path = output_path

        self._buffers: dict[str, list[np.ndarray]] = {}
        self._metadata: dict[str, Any] = {}
        self._step_count = 0

        self.condition_manager = GridConditionManager()
        self._setup_done = False

        self._original_check_termination: Any = None

    @property
    def num_conditions(self) -> int:
        return self.condition_manager.num_conditions

    @staticmethod
    def _to_np(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy().copy()

    def register_buffer_key(self, name: str) -> None:
        """Reserve a recording channel."""
        if name not in self._buffers:
            self._buffers[name] = []

    def append_buffer(self, name: str, data: np.ndarray) -> None:
        """Append one step of data to a named channel."""
        self._buffers[name].append(data)

    def set_metadata(self, key: str, value: Any) -> None:
        """Set a metadata entry in the NPZ."""
        self._metadata[key] = value

    def on_pre_evaluate_policy(self) -> None:
        env = self._get_env()
        self._metadata.update(self._collect_sim_metadata(env))
        if self.config.note:
            self._metadata["note"] = self.config.note

        for name in [
            "dof_pos_target",
            "dof_pos",
            "dof_vel",
            "torques",
            "actions",
            "root_pos",
            "root_quat_xyzw",
            "root_lin_vel",
            "root_ang_vel",
            "body_pos_w",
            "body_quat_xyzw",
            "commanded_velocity",
        ]:
            self.register_buffer_key(name)

        if self.config.record_motion_ref:
            for name in [
                "motion_time_steps",
                "motion_root_pos",
                "motion_root_quat_xyzw",
                "motion_root_lin_vel",
                "motion_root_ang_vel",
                "motion_dof_pos",
                "motion_dof_vel",
            ]:
                self.register_buffer_key(name)

        logger.info(f"EvalRecordingCallback: output={self.output_path}")

    def ensure_finalized(self) -> None:
        """Ensure the condition manager is finalized and recording is ready.

        Other callbacks must call this at the start of their own deferred
        setup, before reading ``condition_manager.conditions``.  Safe to
        call multiple times — only the first call does work.
        """
        if not self._setup_done:
            self._deferred_setup()

    def _deferred_setup(self) -> None:
        """Finalize grid conditions on the first env step.

        Called after all callbacks have registered their sweep axes.
        When no axes are registered, defaults to 1 condition.
        """
        env = self._get_env()
        self.condition_manager.finalize(env.num_envs)
        self._metadata.update(self.condition_manager.get_metadata())

        # --- Suppress termination/resets so fallen robots stay down ---
        self._original_check_termination = env._check_termination

        def _no_termination():
            env.reset_buf[:] = 0
            env.time_out_buf[:] = 0

        env._check_termination = _no_termination

        for name in ["torques_substep", "dof_pos_substep", "dof_vel_substep"]:
            self.register_buffer_key(name)

        self._setup_done = True
        logger.info(f"EvalRecordingCallback: {self.condition_manager.num_conditions} conditions, disabled env resets")

    def on_pre_eval_env_step(self, actor_state: dict) -> dict:
        self.ensure_finalized()
        return actor_state

    def on_post_eval_env_step(self, actor_state: dict) -> dict:
        env = self._get_env()
        sim = env.simulator
        _to_np = self._to_np
        num_c = self.condition_manager.num_conditions
        s = slice(num_c)

        self._buffers["dof_pos"].append(_to_np(sim.dof_pos[s]))
        self._buffers["dof_vel"].append(_to_np(sim.dof_vel[s]))

        term = env.action_manager.get_term("joint_control")
        self._buffers["torques"].append(_to_np(term.torques[s]))

        root = sim.robot_root_states[s]
        self._buffers["root_pos"].append(_to_np(root[:, :3]))
        self._buffers["root_quat_xyzw"].append(_to_np(root[:, 3:7]))
        self._buffers["root_lin_vel"].append(_to_np(root[:, 7:10]))
        self._buffers["root_ang_vel"].append(_to_np(root[:, 10:13]))

        self._buffers["body_pos_w"].append(_to_np(sim._rigid_body_pos[s]))
        self._buffers["body_quat_xyzw"].append(_to_np(sim._rigid_body_rot[s]))

        self._buffers["torques_substep"].append(_to_np(term.torques_substep[s]))
        self._buffers["dof_pos_substep"].append(_to_np(term.dof_pos_substep[s]))
        self._buffers["dof_vel_substep"].append(_to_np(term.dof_vel_substep[s]))

        if "actions" in actor_state and actor_state["actions"] is not None:
            self._buffers["actions"].append(_to_np(actor_state["actions"][s]))

        self._buffers["dof_pos_target"].append(
            _to_np(term._actions_after_delay[s] * term.action_scales + env.default_dof_pos[s])
        )

        if hasattr(env, "command_manager") and env.command_manager is not None:
            try:
                self._buffers["commanded_velocity"].append(_to_np(env.command_manager.commands[s]))
            except (AttributeError, IndexError):
                pass

            if self.config.record_motion_ref:
                try:
                    mc = env.command_manager.get_state("motion_command")
                    self._buffers["motion_time_steps"].append(mc.time_steps[s].detach().cpu().numpy().copy())
                    self._buffers["motion_root_pos"].append(_to_np(mc.root_pos_w[s]))
                    self._buffers["motion_root_quat_xyzw"].append(_to_np(mc.root_quat_w[s]))
                    self._buffers["motion_root_lin_vel"].append(_to_np(mc.root_lin_vel_w[s]))
                    self._buffers["motion_root_ang_vel"].append(_to_np(mc.root_ang_vel_w[s]))
                    self._buffers["motion_dof_pos"].append(_to_np(mc.joint_pos[s]))
                    self._buffers["motion_dof_vel"].append(_to_np(mc.joint_vel[s]))
                except (AttributeError, KeyError):
                    pass

        self._step_count += 1
        return actor_state

    def on_post_evaluate_policy(self) -> None:
        if self._original_check_termination is not None:
            env = self._get_env()
            env._check_termination = self._original_check_termination
            self._original_check_termination = None

        log_prefix = (
            f"EvalRecordingCallback: {self._step_count} steps x {self.condition_manager.num_conditions} conditions"
        )
        self._save_npz(
            self._buffers,
            self._metadata,
            self._step_count,
            self.output_path,
            log_prefix,
        )

    @staticmethod
    def _collect_sim_metadata(env: Any) -> dict[str, Any]:
        """Collect standard simulator metadata (timing, DOF names, limits, URDF path)."""
        sim = env.simulator
        meta: dict[str, Any] = {
            "dt": float(env.dt),
            "fps": round(1.0 / float(env.dt)),
            "sim_dt": float(env.sim_dt),
            "sim_fps": round(1.0 / float(env.sim_dt)),
            "control_decimation": sim.simulator_config.sim.control_decimation_steps,
        }
        if hasattr(sim, "dof_names"):
            meta["dof_names"] = list(sim.dof_names)
        if hasattr(sim, "body_names"):
            meta["body_names"] = list(sim.body_names)
        robot_cfg = env.robot_config
        meta["effort_limits"] = list(robot_cfg.dof_effort_limit_list)
        meta["dof_pos_lower_limits"] = list(robot_cfg.dof_pos_lower_limit_list)
        meta["dof_pos_upper_limits"] = list(robot_cfg.dof_pos_upper_limit_list)
        meta["velocity_limits"] = list(robot_cfg.dof_vel_limit_list)
        asset_cfg = robot_cfg.asset
        meta["urdf_path"] = str(Path(asset_cfg.asset_root) / asset_cfg.urdf_file)
        return meta

    @staticmethod
    def _save_npz(
        buffers: dict[str, list[np.ndarray]],
        metadata: dict[str, Any],
        step_count: int,
        output_path: str,
        log_prefix: str,
    ) -> None:
        """Stack buffers and write compressed NPZ."""
        if step_count == 0:
            return
        arrays: dict[str, np.ndarray] = {}
        for name, values in buffers.items():
            if values:
                arrays[name] = np.stack(values, axis=0)
        arrays["_metadata_json"] = np.array(json.dumps(metadata))
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(path), **arrays)
        summary = ", ".join(f"{k}{list(v.shape)}" for k, v in arrays.items() if k != "_metadata_json")
        logger.info(f"{log_prefix}: saved {step_count} steps to {path}\n  Channels: {summary}")
