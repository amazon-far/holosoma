from __future__ import annotations

import json
import sys

import numpy as np
import onnx
import onnxruntime
from loguru import logger
from termcolor import colored

from holosoma_inference.config.config_types.inference import InferenceConfig
from holosoma_inference.policies import BasePolicy
from holosoma_inference.policies.tracking_source import NullTrackingSource, TrackingSource
from holosoma_inference.policies.wbt_utils import MotionClockUtil, PinocchioRobot, TimestepUtil
from holosoma_inference.utils.clock import ClockSub
from holosoma_inference.utils.math.quat import (
    matrix_from_quat,
    quat_mul,
    quat_rotate_inverse,
    quat_to_rpy,
    rpy_to_quat,
    subtract_frame_transforms,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)


class _InlinePolicyOutputShmWriter:
    """Inlined writer for the `holosoma_policy_output` SHM segment.

    Mirrors holosoma_ext_viser.skeleton_shm.PolicyOutputShmWriter byte-for-
    byte. Inlined here so the bazel-built inference binary publishes the
    (q_target, dof_pos) stream without a dep on holosoma_ext_viser, which
    is pip-only. Layout must stay in sync with _PolicyOutputShmReader in
    policy_output_viewer.py.
    """

    _HEADER_SIZE = 8

    def __init__(self, num_dofs: int = 29, shm_name: str = "holosoma_policy_output") -> None:
        import struct as _struct
        from multiprocessing import shared_memory

        self._struct = _struct
        self._shm_name = shm_name
        self._num_dofs = num_dofs
        joint_size = num_dofs * 8
        self._total_size = self._HEADER_SIZE + 2 * joint_size

        # One WBT policy per shm_name. If two WBT policies start
        # concurrently with the same name, the second silently unlinks
        # the first's SHM and the viewer starts seeing a mix of writes
        # that correspond to neither. Callers that need multi-policy
        # output should pass distinct shm_name values (e.g. suffixed
        # with os.getpid()). See 2026-05-05 code review #13.
        try:
            old = shared_memory.SharedMemory(name=shm_name, create=False)
            old.close()
            old.unlink()
        except FileNotFoundError:
            pass

        self._shm = shared_memory.SharedMemory(name=shm_name, create=True, size=self._total_size)
        off = self._HEADER_SIZE
        self._q_target = np.ndarray((num_dofs,), dtype=np.float64, buffer=self._shm.buf[off : off + joint_size])
        off += joint_size
        self._dof_pos = np.ndarray((num_dofs,), dtype=np.float64, buffer=self._shm.buf[off : off + joint_size])
        self._q_target[:] = 0.0
        self._dof_pos[:] = 0.0

    def write(self, q_target: np.ndarray, dof_pos: np.ndarray) -> None:
        import time as _time

        self._q_target[:] = q_target
        self._dof_pos[:] = dof_pos
        self._struct.pack_into("d", self._shm.buf, 0, _time.monotonic())

    def close(self) -> None:
        try:
            self._shm.close()
        except Exception as exc:
            logger.debug("SHM close failed: {}", exc)
        try:
            self._shm.unlink()
        except Exception as exc:
            logger.debug("SHM unlink failed: {}", exc)


class WholeBodyTrackingPolicy(BasePolicy):
    def __init__(self, config: InferenceConfig, tracking_source: TrackingSource | None = None):
        self.config = config
        # Transport-agnostic external tracking input. Default ``NullTrackingSource``
        # makes this an exact no-op — behavior is byte-identical to the
        # pre-change policy unless a concrete source is injected at construction.
        # See ``holosoma_inference.policies.tracking_source`` for the contract.
        self._tracking_source: TrackingSource = tracking_source or NullTrackingSource()

        # Lazy retargeter state. Populated on first non-None payload so that
        # mink/mujoco are NOT imported at policy construction when no tracking
        # source is in play — keeps NullTrackingSource byte-identical to the
        # pre-change policy (no new dependency load path).
        self._retargeter = None  # type: ignore[var-annotated]
        self._retargeter_init_failed = False
        self._retargeter_runtime_warned = False
        # 2026-05-05 code review #6: track how many frames have fallen
        # through after the first WARN, and re-warn periodically so an
        # operator who misses the one-shot log line still sees that the
        # retargeter is no-op'ing (the 2026-05-02 analysis found a 30-
        # minute silent retargeter OOM this would have caught).
        self._retargeter_runtime_err_count = 0
        self._retargeter_runtime_last_warn_count = 0
        # Re-WARN every ~500 frames (= ~10 s at 50 Hz).
        self._retargeter_runtime_rewarn_every = 500

        # initialize motion state
        self.motion_clip_progressing = False
        self.curr_motion_timestep = config.task.motion_start_timestep
        self.motion_command_t = None
        self.ref_quat_xyzw_t = None
        self.motion_command_0 = None
        self.ref_quat_xyzw_0 = None

        # Initialize clock for sim-time synchronization
        clock_sub = ClockSub()
        clock_sub.start()
        clock_util = MotionClockUtil(clock_sub)
        self.timestep_util = TimestepUtil(
            clock=clock_util,
            interval_ms=1000.0 / config.task.rl_rate,
            start_timestep=config.task.motion_start_timestep,
        )

        # Read use_sim_time from config
        self.use_sim_time = config.task.use_sim_time

        self._stiff_hold_active = True
        self.robot_yaw_offset = 0.0
        self.motion_yaw_offset = 0.0
        self.per_joint_policy_action_scale: np.ndarray | None = None

        super().__init__(config)
        self._configure_action_scales()

        # Load stiff startup parameters from robot config
        if config.robot.stiff_startup_pos is not None:
            self._stiff_hold_q = np.array(config.robot.stiff_startup_pos, dtype=np.float32).reshape(1, -1)
        else:
            # Fallback to default_dof_angles if not specified
            self._stiff_hold_q = np.array(config.robot.default_dof_angles, dtype=np.float32).reshape(1, -1)

        if config.robot.stiff_startup_kp is not None:
            self._stiff_hold_kp = np.array(config.robot.stiff_startup_kp, dtype=np.float32)
        else:
            raise ValueError("Robot config must specify stiff_startup_kp for WBT policy")

        if config.robot.stiff_startup_kd is not None:
            self._stiff_hold_kd = np.array(config.robot.stiff_startup_kd, dtype=np.float32)
        else:
            raise ValueError("Robot config must specify stiff_startup_kd for WBT policy")

        if self._stiff_hold_q.shape[1] != self.num_dofs:
            raise ValueError("Stiff startup pose dimension mismatch with robot DOFs")

        # Prompt user before entering stiff mode (only if stdin is available)
        def _show_warning():
            logger.warning(
                colored(
                    "⚠️  Non-interactive mode detected - cannot prompt for stiff mode confirmation!",
                    "red",
                    attrs=["bold"],
                )
            )

        if hasattr(self, "_shared_hardware_source"):
            logger.info(colored("Skipping stiff hold prompt (secondary policy)", "yellow"))
        elif sys.stdin.isatty():
            logger.info(colored("\n⚠️  Ready to enter stiff hold mode", "yellow", attrs=["bold"]))
            logger.info(colored("Press Enter to continue...", "yellow"))
            try:
                input()
                logger.info(colored("✓ Entering stiff hold mode", "green"))
            except EOFError:
                # [drockyd] seems like in some cases, input() will raise EOFError even in interactive mode.
                _show_warning()
        else:
            _show_warning()

    def _get_ref_body_orientation_in_world(self, robot_state_data):
        # Create configuration for pinocchio robot
        # Note:
        # 1. pinocchio quaternion is in xyzw format, robot_state_data is in wxyz format
        # 2. joint sequences in pinocchio robot and real robot are different

        # free base pos, does not matter
        root_pos = robot_state_data[0, :3]

        # free base ori, wxyz -> xyzw
        root_ori_xyzw = wxyz_to_xyzw(robot_state_data[:, 3:7])[0]

        # dof pos in real robot -> pinocchio robot
        num_dofs = self.num_dofs
        dof_pos_in_real = robot_state_data[0, 7 : 7 + num_dofs]
        dof_pos_in_pinocchio = dof_pos_in_real[self.pinocchio_robot.real2pinocchio_index]

        configuration = np.concatenate([root_pos, root_ori_xyzw, dof_pos_in_pinocchio], axis=0)

        ref_ori_xyzw = self.pinocchio_robot.fk_and_get_ref_body_orientation_in_world(configuration)
        return xyzw_to_wxyz(ref_ori_xyzw)

    def setup_policy(self, model_path):
        from holosoma_inference.policies.base import _build_ort_session_options

        self.onnx_policy_session = onnxruntime.InferenceSession(model_path, sess_options=_build_ort_session_options())
        self.onnx_input_names = [inp.name for inp in self.onnx_policy_session.get_inputs()]
        self.onnx_output_names = [out.name for out in self.onnx_policy_session.get_outputs()]

        # Extract KP/KD from ONNX metadata (same as base class)
        onnx_model = onnx.load(model_path)
        metadata = {}
        for prop in onnx_model.metadata_props:
            metadata[prop.key] = json.loads(prop.value)

        # Extract URDF text from ONNX metadata
        assert "robot_urdf" in metadata, "Robot urdf text not found in ONNX metadata"
        self.pinocchio_robot = PinocchioRobot(self.config.robot, metadata["robot_urdf"])

        self.onnx_kp = np.array(metadata["kp"]) if "kp" in metadata else None
        self.onnx_kd = np.array(metadata["kd"]) if "kd" in metadata else None

        if self.onnx_kp is not None:
            from pathlib import Path

            logger.info(f"Loaded KP/KD from ONNX metadata: {Path(model_path).name}")

        # get initial command and ref quat xyzw at the configured start timestep
        time_step = np.array([[self.config.task.motion_start_timestep]], dtype=np.float32)

        # Use configured observation dimensions (including history) instead of a hard-coded value.
        actor_obs_template = self.obs_buf_dict.get("actor_obs")
        if actor_obs_template is None:
            raise ValueError("Observation group 'actor_obs' must be configured for WBT policy.")
        obs = actor_obs_template.copy()
        input_feed = {"obs": obs, "time_step": time_step}
        outputs = self.onnx_policy_session.run(["joint_pos", "joint_vel", "ref_quat_xyzw"], input_feed)

        # motion_command_t/ref_quat_xyzw_t will be used in get_current_obs_buffer_dict
        self.motion_command_t = np.concatenate(outputs[0:2], axis=1)  # (1, 58)
        self.ref_quat_xyzw_t = outputs[2]
        # duplicate, will be used in _get_init_target and _handle_stop_policy
        self.motion_command_0 = self.motion_command_t.copy()
        self.ref_quat_xyzw_0 = self.ref_quat_xyzw_t.copy()

        def policy_act(input_feed):
            output = self.onnx_policy_session.run(["actions", "joint_pos", "joint_vel", "ref_quat_xyzw"], input_feed)
            action = output[0]
            motion_command = np.concatenate(output[1:3], axis=1)
            ref_quat_xyzw = output[3]
            return action, motion_command, ref_quat_xyzw

        self.policy = policy_act

        # Shared-memory publisher for the side-by-side policy_output_viewer.
        # Best-effort — the policy runs unchanged if shm creation fails.
        self._policy_shm_writer = None
        try:
            self._policy_shm_writer = _InlinePolicyOutputShmWriter(num_dofs=self.num_dofs)
            import atexit as _atexit

            _atexit.register(self._cleanup_policy_shm)
            logger.info("Policy-output shared memory publisher started (holosoma_policy_output)")
        except Exception as e:
            logger.warning("Failed to start policy-output shared memory writer: {}", e)

    def _cleanup_policy_shm(self):
        w = getattr(self, "_policy_shm_writer", None)
        if w is not None:
            try:
                w.close()
            except Exception as exc:
                logger.debug("policy-output SHM close failed: {}", exc)
            self._policy_shm_writer = None

    def _capture_policy_state(self):
        state = super()._capture_policy_state()
        state.update(
            {
                "motion_command_0": self.motion_command_0.copy(),
                "ref_quat_xyzw_0": self.ref_quat_xyzw_0.copy(),
                "per_joint_policy_action_scale": self.per_joint_policy_action_scale.copy()
                if self.per_joint_policy_action_scale is not None
                else None,
            }
        )
        return state

    def _restore_policy_state(self, state):
        super()._restore_policy_state(state)
        self.motion_command_0 = state["motion_command_0"].copy()
        self.ref_quat_xyzw_0 = state["ref_quat_xyzw_0"].copy()
        saved = state["per_joint_policy_action_scale"]
        self.per_joint_policy_action_scale = saved.copy() if saved is not None else None
        self.motion_clip_progressing = False
        self.timestep_util.reset(start_timestep=0)
        self.curr_motion_timestep = self.timestep_util.timestep
        self.robot_yaw_offset = 0.0
        self.motion_yaw_offset = 0.0

    def _on_policy_switched(self, model_path: str):
        super()._on_policy_switched(model_path)
        self.motion_command_t = self.motion_command_0.copy()
        self.ref_quat_xyzw_t = self.ref_quat_xyzw_0.copy()
        self.motion_clip_progressing = False
        self.timestep_util.reset(start_timestep=0)
        self.curr_motion_timestep = self.timestep_util.timestep
        self._stiff_hold_active = True
        self.robot_yaw_offset = 0.0
        self.motion_yaw_offset = 0.0
        self._configure_action_scales()

    def get_init_target(self, robot_state_data):
        """Get initialization target joint positions."""
        dof_pos = robot_state_data[:, 7 : 7 + self.num_dofs]
        if self.get_ready_state:
            # Interpolate from current dof_pos to first pose in motion command
            target_dof_pos = self.motion_command_0[:, : self.num_dofs]

            q_target = dof_pos + (target_dof_pos - dof_pos) * (self.init_count / 500)
            self.init_count += 1
            return q_target
        return dof_pos

    def get_current_obs_buffer_dict(self, robot_state_data):
        current_obs_buffer_dict = {}

        # base_quat (used below for projected_gravity; not included in the
        # ONNX obs itself for WBT).
        current_obs_buffer_dict["base_quat"] = robot_state_data[:, 3:7]

        # motion_command
        current_obs_buffer_dict["motion_command"] = self.motion_command_t

        # motion_ref_ori_b
        motion_ref_ori = xyzw_to_wxyz(self.ref_quat_xyzw_t)  # wxyz
        motion_ref_ori = self._remove_yaw_offset(motion_ref_ori, self.motion_yaw_offset)

        # robot_ref_ori
        robot_ref_ori = self._get_ref_body_orientation_in_world(robot_state_data)  #  wxyz
        robot_ref_ori = self._remove_yaw_offset(robot_ref_ori, self.robot_yaw_offset)

        motion_ref_ori_b = matrix_from_quat(subtract_frame_transforms(robot_ref_ori, motion_ref_ori))
        current_obs_buffer_dict["motion_ref_ori_b"] = motion_ref_ori_b[..., :2].reshape(1, -1)

        # base_ang_vel
        current_obs_buffer_dict["base_ang_vel"] = robot_state_data[:, 7 + self.num_dofs + 3 : 7 + self.num_dofs + 6]

        # dof_pos
        current_obs_buffer_dict["dof_pos"] = robot_state_data[:, 7 : 7 + self.num_dofs] - self.default_dof_angles

        # dof_vel
        current_obs_buffer_dict["dof_vel"] = robot_state_data[
            :, 7 + self.num_dofs + 6 : 7 + self.num_dofs + 6 + self.num_dofs
        ]

        # actions
        current_obs_buffer_dict["actions"] = self.last_policy_action

        # projected_gravity — mirrors BasePolicy.get_current_obs_buffer_dict's
        # three-way selection (force-upright debug > interface-provided >
        # compute from quat). Only consumed by dense-tracker observation
        # presets (e.g. wbt-dense); the stock wbt preset's obs_dict doesn't
        # reference it, so the extra entry is harmless.
        expected_len = 7 + self.num_dofs + 6 + self.num_dofs
        if self.config.task.debug.force_upright_imu:
            current_obs_buffer_dict["projected_gravity"] = np.array([[0.0, 0.0, -1.0]])
        elif robot_state_data.shape[1] == expected_len + 3:
            current_obs_buffer_dict["projected_gravity"] = robot_state_data[:, expected_len : expected_len + 3]
        else:
            v = np.array([[0, 0, -1]])
            current_obs_buffer_dict["projected_gravity"] = quat_rotate_inverse(current_obs_buffer_dict["base_quat"], v)

        return current_obs_buffer_dict

    def rl_inference(self, robot_state_data):
        # prepare obs, run policy inference
        import os as _os
        import time as _time

        _dbg = _os.environ.get("HOLOSOMA_INFERENCE_TIMING", "0") in ("1", "true", "True")
        _ts: list[tuple[str, float]] = []
        if _dbg:
            _ts.append(("start", _time.perf_counter()))

        # Non-blocking poll of the external tracking source. On
        # NullTrackingSource (default) this returns None and the policy
        # falls through to the ONNX-clip motion_command_t path unchanged.
        # On a concrete source, translate the SMPLH payload into a
        # ``(1, 58)`` motion_command_t via SMPLRetargeter and substitute
        # it into ``self.motion_command_t`` BEFORE the ONNX policy runs,
        # so the policy conditions on the teleop target instead of the
        # clip. If retargeting fails (bad quats, IK divergence, missing
        # URDF), we log once and fall through to the ONNX-clip path —
        # the driver's sticky-fault contract expects the policy to keep
        # producing commands under partial sensor failures.
        external = self._tracking_source.get_latest()
        if _dbg:
            _ts.append(("tracking_poll", _time.perf_counter()))
        if external is not None:
            # Payload log was firing on every inference tick (50-100 Hz),
            # drowning legitimate WARN/ERROR in DEBUG mode. Throttle to
            # one line per 200 payloads so a pathological source (quality
            # flipping, device swap) still shows up without the volume.
            if not hasattr(self, "_external_log_counter"):
                self._external_log_counter = 0
            self._external_log_counter += 1
            if self._external_log_counter % 200 == 1:
                logger.debug(
                    f"WBT tracking_source: payload #{self._external_log_counter} "
                    f"device={external.device_type!r} mode={external.mode} "
                    f"body_joints={len(external.joint_names)} "
                    f"hand_joints={len(external.hand_joint_names)} "
                    f"quality={external.tracking_quality}"
                )
            retargeted = self._retarget_payload_to_motion_command(external)
            if retargeted is not None:
                retargeted_motion_command, retargeted_ref_quat_xyzw = retargeted
                self.motion_command_t = retargeted_motion_command
                # Also substitute the reference orientation. Without this,
                # motion_ref_ori_b in the obs stays anchored at the ONNX
                # clip's baseline pose while joint targets come from live
                # teleop, which caused the on-robot T-pose under-reach on
                # 2026-05-05 (code review finding #5). None means the
                # retargeter returned an invalid root quat; hold the last
                # value rather than feed NaN into the obs.
                if retargeted_ref_quat_xyzw is not None:
                    self.ref_quat_xyzw_t = retargeted_ref_quat_xyzw
        if _dbg:
            _ts.append(("retarget", _time.perf_counter()))

        if not self.motion_clip_progressing:
            # Keep motion index pinned at the configured start while waiting to trigger the clip.
            self.timestep_util.reset(start_timestep=self.config.task.motion_start_timestep)
            self.curr_motion_timestep = self.timestep_util.timestep

        obs = self.prepare_obs_for_rl(robot_state_data)
        if _dbg:
            _ts.append(("prepare_obs", _time.perf_counter()))
        if self.config.task.print_observations:
            self._print_observations(obs)

        input_feed = {"time_step": np.array([[self.curr_motion_timestep]], dtype=np.float32), "obs": obs["actor_obs"]}
        policy_action, self.motion_command_t, self.ref_quat_xyzw_t = self.policy(input_feed)
        if _dbg:
            _ts.append(("policy", _time.perf_counter()))

        # clip policy action
        policy_action = np.clip(policy_action, -100, 100)
        # store last policy action
        self.last_policy_action = policy_action.copy()
        # scale policy action
        if self.per_joint_policy_action_scale is None:
            self.scaled_policy_action = policy_action * self.policy_action_scale
        else:
            self.scaled_policy_action = policy_action * self.per_joint_policy_action_scale

        # Publish (q_target, dof_pos) to shared memory for policy_output_viewer.
        # q_target is the commanded set-point pre-Dampener; dof_pos is the
        # robot readback. Both in the policy's joint order (matches the ONNX
        # dof_names). Best-effort; shape mismatches bail silently.
        if getattr(self, "_policy_shm_writer", None) is not None:
            try:
                q_target_pub = (self.scaled_policy_action + self.default_dof_angles).reshape(-1).astype(np.float64)
                dof_pos_pub = robot_state_data[0, 7 : 7 + self.num_dofs].astype(np.float64)
                self._policy_shm_writer.write(q_target_pub, dof_pos_pub)
            except Exception as _e:
                logger.debug("policy-output shm write failed: {}", _e)

        # update motion timestep
        self._set_motion_timestep()
        if _dbg and len(_ts) > 1:
            parts = []
            for i in range(1, len(_ts)):
                dt_ms = (_ts[i][1] - _ts[i - 1][1]) * 1000.0
                parts.append(f"{_ts[i][0]}={dt_ms:.2f}ms")
            logger.info("[inference_timing] " + " ".join(parts))

        return self.scaled_policy_action

    def _retargeter_warn(self, reason: str) -> None:
        """Log a retargeter-fallthrough WARN once, then throttled re-WARNs.

        First call prints the reason + intent to suppress. Every
        `_retargeter_runtime_rewarn_every` calls after that, we log
        again with the accumulated count so silent-failure stretches
        are visible in logs.
        """
        self._retargeter_runtime_err_count += 1
        if not self._retargeter_runtime_warned:
            logger.warning(
                "WBT tracking_source: {}: falling through to ONNX-clip "
                "motion_command. Subsequent failures suppressed; a periodic "
                "re-WARN will fire every {} frames.",
                reason,
                self._retargeter_runtime_rewarn_every,
            )
            self._retargeter_runtime_warned = True
            self._retargeter_runtime_last_warn_count = 1
            return
        delta = self._retargeter_runtime_err_count - self._retargeter_runtime_last_warn_count
        if delta >= self._retargeter_runtime_rewarn_every:
            logger.warning(
                "WBT tracking_source: retargeter still failing. {} total fallthroughs so far (last reason: {}).",
                self._retargeter_runtime_err_count,
                reason,
            )
            self._retargeter_runtime_last_warn_count = self._retargeter_runtime_err_count

    def _reset_retargeter_runtime_state(self) -> None:
        """Clear the one-shot WARN latch + error counter.

        Called on policy stop/start transitions so a transient tracker
        glitch at the start of a session doesn't silence every future
        failure in the process lifetime (2026-05-05 code review #6).
        """
        self._retargeter_runtime_warned = False
        self._retargeter_runtime_err_count = 0
        self._retargeter_runtime_last_warn_count = 0

    def _retarget_payload_to_motion_command(self, payload) -> tuple[np.ndarray, np.ndarray] | None:
        """Translate a ``TrackingPayload`` into (motion_command_t, ref_quat_xyzw_t).

        Returns a tuple of two arrays on success, or ``None`` when retargeting
        is not usable — the caller falls through to the ONNX-clip path in
        that case.

        Return tuple:
          * ``motion_command_t``: ``(1, 58)`` float32 (29 joint_pos then 29 joint_vel).
          * ``ref_quat_xyzw_t``: ``(1, 4)`` float32 xyzw root orientation from
            the retargeter's IK-solved freejoint. Feeds ``motion_ref_ori_b``
            in ``get_current_obs_buffer_dict`` so the policy's orientation cue
            tracks live teleop instead of the frozen ONNX-clip baseline.

        Failure modes handled here (all fall through, none raise):
          * ``robot.urdf_path`` is unset → cannot construct the retargeter.
          * SMPLRetargeter construction fails (missing deps, bad URDF).
          * ``joint_transforms`` has the wrong size for (24, 7) reshape.
          * ``retarget()`` raises (bad quaternions, IK divergence, etc).

        The first retargeter construction is lazy — mink/mujoco are not
        imported until a non-None payload arrives. This keeps
        ``NullTrackingSource`` behavior byte-identical to today.
        """
        # Initial construction: tried at most once. If it fails, we stay
        # in ONNX-clip mode for the remainder of this policy's lifetime.
        #
        # 2026-05-05 code review #7: mink's IK solver mutates
        # self._config.q across frames, and when the MuJoCo interface
        # backend is also active we end up with two MjModel instances
        # loaded from the same XML in the same process (interface +
        # retargeter). This is documented-safe for a single policy but
        # would silently alias state if a secondary policy shared this
        # retargeter through _shared_hardware_source. We don't support
        # that today; the assert below is a tripwire.
        if self._retargeter is None and not self._retargeter_init_failed:
            if getattr(self, "_shared_hardware_source", False):
                assert not hasattr(self, "_retargeter_is_shared"), (
                    "WBT retargeter would be shared across policies via "
                    "_shared_hardware_source; mink.solve_ik keeps per-frame "
                    "state in self._config.q which would alias. Not supported."
                )

            urdf_path = getattr(self.config.robot, "urdf_path", None)
            if not urdf_path:
                self._retargeter_init_failed = True
                logger.warning(
                    "WBT tracking_source: received payload but robot.urdf_path is unset; "
                    "cannot construct SMPLRetargeter: falling through to ONNX-clip motion_command."
                )
                return None
            try:
                from holosoma_retargeting.src.realtime_smpl_retargeter import SMPLRetargeter

                dt = 1.0 / float(self.config.task.rl_rate)
                import os as _os

                ik_iters = int(_os.environ.get("HOLOSOMA_RETARGETER_IK_ITERS", "4") or 4)
                self._retargeter = SMPLRetargeter(urdf_path=urdf_path, dt=dt, max_ik_iters=ik_iters)
            except Exception as exc:
                self._retargeter_init_failed = True
                logger.warning(
                    "WBT tracking_source: failed to construct SMPLRetargeter "
                    f"(urdf_path={urdf_path!r}): {exc!r}: falling through to ONNX-clip motion_command."
                )
                return None

        if self._retargeter is None:
            return None

        # Reshape flat transforms into (24, 7). TrackingPayload documents
        # joint_transforms as a flattened (N, 7) array; we expect the SMPL
        # 24-joint body ordering here.
        try:
            transforms = np.asarray(payload.joint_transforms, dtype=np.float32).reshape(24, 7)
        except (ValueError, AttributeError) as exc:
            self._retargeter_warn(f"joint_transforms reshape to (24, 7) failed ({exc!r})")
            return None

        try:
            q_joints, dq_joints, root_orn_wxyz = self._retargeter.retarget(transforms)
        except Exception as exc:
            self._retargeter_warn(f"SMPLRetargeter.retarget raised {exc!r}")
            return None

        q_joints = np.asarray(q_joints, dtype=np.float32).reshape(-1)
        dq_joints = np.asarray(dq_joints, dtype=np.float32).reshape(-1)
        if q_joints.size != self.num_dofs or dq_joints.size != self.num_dofs:
            self._retargeter_warn(
                f"retargeter returned unexpected dim (q={q_joints.size}, dq={dq_joints.size}, expected={self.num_dofs})"
            )
            return None

        motion_command = np.concatenate([q_joints, dq_joints]).reshape(1, -1)

        # Convert the retargeter's wxyz root quaternion into xyzw and shape
        # it to match how get_current_obs_buffer_dict consumes
        # ref_quat_xyzw_t: (1, 4) so self.ref_quat_xyzw_t[0] indexes cleanly
        # through xyzw_to_wxyz + _remove_yaw_offset. If the retargeter hands
        # back something non-finite (rare — only seen after a failed IK
        # solve that nonetheless returned), fall through; motion_command_t
        # still gets the joint targets but ref_quat_xyzw_t stays at its
        # last valid value rather than propagating garbage into the obs.
        root_orn_wxyz = np.asarray(root_orn_wxyz, dtype=np.float32).reshape(-1)
        if root_orn_wxyz.size != 4 or not np.all(np.isfinite(root_orn_wxyz)):
            self._retargeter_warn(
                f"retargeter root_orn_wxyz invalid (size={root_orn_wxyz.size}); keeping previous ref_quat_xyzw_t"
            )
            return motion_command, None

        # wxyz -> xyzw: (w, x, y, z) -> (x, y, z, w).
        ref_quat_xyzw = np.array(
            [root_orn_wxyz[1], root_orn_wxyz[2], root_orn_wxyz[3], root_orn_wxyz[0]],
            dtype=np.float32,
        ).reshape(1, 4)
        return motion_command, ref_quat_xyzw

    def _configure_action_scales(self) -> None:
        """Configure action scales, prioritising ONNX metadata over config fallbacks.

        Resolution order:
        1. ONNX metadata ``action_scale`` (scalar or per-joint list)
        2. ``robot.default_per_joint_action_scale`` when
           ``task.action_scales_by_effort_limit_over_p_gain`` is True
        3. Fall back to the scalar ``task.policy_action_scale``
        """
        raw_metadata = dict(self.onnx_policy_session.get_modelmeta().custom_metadata_map)
        onnx_action_scale = self._parse_action_scale_metadata(raw_metadata.get("action_scale"))

        if onnx_action_scale is not None:
            scales = onnx_action_scale.astype(np.float32, copy=False).reshape(-1)
        elif self.config.task.action_scales_by_effort_limit_over_p_gain:
            fallback = self.config.robot.default_per_joint_action_scale
            if fallback is None:
                raise ValueError(
                    "task.action_scales_by_effort_limit_over_p_gain=True requires ONNX metadata key "
                    "'action_scale' (scalar or per-joint list) or "
                    "robot.default_per_joint_action_scale."
                )
            scales = np.asarray(fallback, dtype=np.float32).reshape(-1)
            logger.warning("ONNX metadata 'action_scale' missing; using robot.default_per_joint_action_scale.")
        else:
            self.per_joint_policy_action_scale = None
            return

        if scales.size == 1:
            scales = np.full(self.num_dofs, scales.item(), dtype=np.float32)
        elif scales.size != self.num_dofs:
            raise ValueError(f"Action scale must contain 1 or {self.num_dofs} values, got {scales.size}.")

        self.per_joint_policy_action_scale = scales.reshape(1, -1)

    @staticmethod
    def _parse_action_scale_metadata(raw_value: str | None) -> np.ndarray | None:
        """Parse action_scale metadata from JSON-serialized or CSV string formats."""
        if raw_value is None:
            return None

        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError:
            parsed = raw_value

        if isinstance(parsed, (int, float)):
            return np.array([float(parsed)], dtype=np.float32)
        if isinstance(parsed, str):
            values = [float(token.strip()) for token in parsed.split(",") if token.strip()]
            if not values:
                raise ValueError("ONNX metadata action_scale is an empty string.")
            return np.array(values, dtype=np.float32)

        values = np.asarray(parsed, dtype=np.float32).reshape(-1)
        if values.size == 0:
            raise ValueError("ONNX metadata action_scale is empty.")
        return values

    def _get_manual_command(self, robot_state_data):
        # TODO: instead of adding kp/kd_override in def _set_motor_command,
        # just use the motor_kp/motor_kd when calling it in _fill_motor_commands
        if not self._stiff_hold_active:
            return None
        return {
            "q": self._stiff_hold_q.copy(),
            "kp": self._stiff_hold_kp,
            "kd": self._stiff_hold_kd,
        }

    def _handle_start_policy(self):
        super()._handle_start_policy()
        self._stiff_hold_active = False
        self._capture_robot_yaw_offset()
        self._capture_motion_yaw_offset(self.ref_quat_xyzw_0)
        # 2026-05-05 code review #6: clear the retargeter one-shot WARN latch
        # so a transient tracker glitch at the start of a session doesn't
        # silence every future failure in the process lifetime.
        self._reset_retargeter_runtime_state()

    def _set_motion_timestep(self):
        if self.motion_clip_progressing:
            prev = self.curr_motion_timestep

            if self.use_sim_time:
                self.curr_motion_timestep = self.timestep_util.get_timestep(log=self.logger)
            else:
                self.curr_motion_timestep += 1

            if self.curr_motion_timestep != prev:
                self.logger.info(f"Motion timestep: {prev} → {self.curr_motion_timestep}")  # noqa: G004

            # Stop motion clip at configured end timestep (keep policy running at final pose)
            if (end := self.config.task.motion_end_timestep) and self.curr_motion_timestep >= end:
                self.logger.info(colored(f"Reached end timestep {end}, stopping motion clip", "yellow"))
                self.motion_clip_progressing = False
                self.curr_motion_timestep = end

    def _handle_stop_policy(self):
        """Handle stop policy action."""
        self.use_policy_action = False
        self.get_ready_state = False
        self._stiff_hold_active = True
        self.logger.info("Actions set to stiff startup command")
        if hasattr(self.interface, "no_action"):
            self.interface.no_action = 0

        self.motion_clip_progressing = False
        self.timestep_util.reset(start_timestep=0)
        self.curr_motion_timestep = self.timestep_util.timestep
        self.ref_quat_xyzw_t = self.ref_quat_xyzw_0.copy()
        self.motion_command_t = self.motion_command_0.copy()
        self.robot_yaw_offset = 0.0
        self.motion_yaw_offset = 0.0
        # 2026-05-05 code review #6: clear retargeter WARN latch on stop so a
        # restart of the policy doesn't stay silent across transient faults.
        self._reset_retargeter_runtime_state()

    def _handle_start_motion_clip(self):
        """Handle start motion clip action."""
        self.timestep_util.reset(start_timestep=self.config.task.motion_start_timestep)
        self.curr_motion_timestep = self.timestep_util.timestep
        self.motion_clip_progressing = True

        if self.config.task.motion_start_timestep > 0 or self.config.task.motion_end_timestep is not None:
            start_str = str(self.config.task.motion_start_timestep)
            end_str = str(self.config.task.motion_end_timestep) if self.config.task.motion_end_timestep else "end"
            self.logger.info(colored(f"Starting motion clip from timestep {start_str} to {end_str}", "blue"))
        else:
            self.logger.info(colored("Starting motion clip", "blue"))

    def _dispatch_command(self, cmd):
        from holosoma_inference.inputs.api.commands import StateCommand

        if cmd == StateCommand.START_MOTION_CLIP:
            self._handle_start_motion_clip()
        else:
            super()._dispatch_command(cmd)

    def _capture_robot_yaw_offset(self):
        """Capture robot yaw when policy starts to use as reference offset."""
        robot_state_data = self.interface.get_low_state()
        if robot_state_data is None:
            self.robot_yaw_offset = 0.0
            self.logger.warning("Unable to capture robot yaw offset - missing robot state.")
            return

        robot_ref_ori = self._get_ref_body_orientation_in_world(robot_state_data)  # wxyz
        yaw = self._quat_yaw(robot_ref_ori)
        self.robot_yaw_offset = yaw
        self.logger.info(colored(f"Robot yaw offset captured at {np.degrees(yaw):.1f} deg", "blue"))

    def _capture_motion_yaw_offset(self, ref_quat_xyzw_0: np.ndarray) -> float:
        """Capture motion yaw when policy starts to use as reference offset."""
        self.motion_yaw_offset = self._quat_yaw(xyzw_to_wxyz(ref_quat_xyzw_0))
        self.logger.info(colored(f"Motion yaw offset captured at {np.degrees(self.motion_yaw_offset):.1f} deg", "blue"))

    def _remove_yaw_offset(self, quat_wxyz: np.ndarray, yaw_offset: float) -> np.ndarray:
        """Remove stored yaw offset from robot orientation quaternion."""
        if abs(yaw_offset) < 1e-6:
            return quat_wxyz
        yaw_quat = rpy_to_quat((0.0, 0.0, -yaw_offset)).reshape(1, 4)
        yaw_quat = np.broadcast_to(yaw_quat, quat_wxyz.shape)
        return quat_mul(yaw_quat, quat_wxyz)

    @staticmethod
    def _quat_yaw(quat_wxyz: np.ndarray) -> float:
        """Extract yaw angle from quaternion array of shape (1, 4)."""
        quat_flat = quat_wxyz.reshape(-1, 4)[0]
        _, _, yaw = quat_to_rpy(quat_flat)
        return float(yaw)
