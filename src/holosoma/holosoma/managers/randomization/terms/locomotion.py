"""Randomization terms for locomotion environments."""

from __future__ import annotations

from typing import Any, Literal, Sequence

import torch
from loguru import logger

from holosoma.config_types.randomization import BaseComRange
from holosoma.managers.action.terms.joint_control import JointPositionActionTerm
from holosoma.managers.randomization.base import RandomizationTermBase
from holosoma.managers.randomization.exceptions import RandomizerNotSupportedError
from holosoma.managers.randomization.terms._shared import (
    _MATERIAL_NUM_BUCKETS,
    _ensure_env_ids_tensor,
)
from holosoma.simulator import mujoco_required_field
from holosoma.simulator.shared.field_decorators import MUJOCO_FIELD_ATTR
from holosoma.utils.sampler import DistributionLike, DistributionSpec, TermSampler, quantiles
from holosoma.utils.simulator_config import SimulatorType


def _get_joint_action_term(env: Any) -> JointPositionActionTerm | None:
    """Return the joint-position action term registered with the action manager."""
    action_manager = getattr(env, "action_manager", None)
    if action_manager is None:
        return None

    get_term = getattr(action_manager, "get_term", None)
    if callable(get_term):
        term = get_term("joint_control")
        if isinstance(term, JointPositionActionTerm):
            return term

    iter_terms = getattr(action_manager, "iter_terms", None)
    if callable(iter_terms):
        for _, term in iter_terms():
            if isinstance(term, JointPositionActionTerm):
                return term

    return None


class PushRandomizerState(RandomizationTermBase):
    """Stateful randomizer that owns push scheduling buffers and counters."""

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        params = cfg.params or {}
        interval = params.get("push_interval_s", [5, 16])
        self.push_interval_range: Sequence[float] = [float(interval[0]), float(interval[1])]
        vector_max = params.get("max_push_vel")
        if vector_max is None:
            raise ValueError("PushRandomizerState requires `max_push_vel` to be specified.")
        self._max_push_vel_tensor = torch.empty(0, dtype=torch.float32, device=env.device)
        self._set_max_push_tensor(vector_max)
        self.enabled: bool = bool(params.get("enabled", True))
        logger.info(
            f"[Randomization] PushRandomizerState initialized (enabled={self.enabled}, "
            f"max_push_vel={self._max_push_vel_tensor.tolist()}, "
            f"interval_s={self.push_interval_range})",
        )

        self.push_interval_s: torch.Tensor | None = None
        self.push_robot_counter: torch.Tensor | None = None
        self.push_robot_plot_counter: torch.Tensor | None = None

    def setup(self, sampler: TermSampler) -> None:
        env = self.env
        device = env.device
        num_envs = env.num_envs

        self.push_interval_s = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.push_robot_counter = torch.zeros(num_envs, dtype=torch.int, device=device)
        self.push_robot_plot_counter = torch.zeros(num_envs, dtype=torch.int, device=device)

        all_ids = torch.arange(num_envs, device=device, dtype=torch.long)
        self._resample_intervals(all_ids, sampler)

    def reset(self, env_ids: torch.Tensor | None, sampler: TermSampler) -> None:
        if self.push_robot_counter is None or self.push_robot_plot_counter is None:
            return
        idx = _ensure_env_ids_tensor(self.env, env_ids)
        if idx.numel() == 0:
            return
        self.push_robot_counter[idx] = 0
        self.push_robot_plot_counter[idx] = 0

    def step(self, sampler: TermSampler) -> None:
        if not self.enabled:
            return
        if self.push_robot_counter is None or self.push_robot_plot_counter is None:
            return
        self.push_robot_counter += 1
        self.push_robot_plot_counter += 1

    # ------------------------------------------------------------------ #
    # Public helpers for other randomization hooks
    # ------------------------------------------------------------------ #

    def configure(
        self,
        *,
        enabled: bool | None = None,
        push_interval_s: Sequence[float] | None = None,
        max_push_vel: Sequence[float] | None = None,
    ) -> None:
        if enabled is not None:
            self.enabled = bool(enabled)
        if push_interval_s is not None:
            self.push_interval_range = [float(push_interval_s[0]), float(push_interval_s[1])]
        if max_push_vel is not None:
            self._set_max_push_tensor(max_push_vel)

    def resample(self, env_ids: torch.Tensor | None, sampler: TermSampler) -> None:
        idx = _ensure_env_ids_tensor(self.env, env_ids)
        if idx.numel() == 0:
            return
        self._resample_intervals(idx, sampler)

    def due_envs(self, dt: float) -> torch.Tensor:
        if not self.enabled:
            return torch.empty(0, device=self.env.device, dtype=torch.long)
        if self.push_interval_s is None or self.push_robot_counter is None:
            return torch.empty(0, device=self.env.device, dtype=torch.long)
        interval_steps = (self.push_interval_s / dt).to(torch.int)
        return (self.push_robot_counter == interval_steps).nonzero(as_tuple=False).flatten()

    def zero_counters(self, env_ids: torch.Tensor) -> None:
        if self.push_robot_counter is None or self.push_robot_plot_counter is None:
            return
        self.push_robot_counter[env_ids] = 0
        self.push_robot_plot_counter[env_ids] = 0

    @property
    def max_push_vel(self) -> torch.Tensor:
        return self._max_push_vel_tensor

    def _resample_intervals(self, env_ids: torch.Tensor, sampler: TermSampler) -> None:
        if self.push_interval_s is None:
            return
        low, high = self.push_interval_range
        low_i = max(1, int(low))
        high_i = max(low_i + 1, int(high))
        # Keyed via the bound sampler. Drawn on the env device, one value per env -> [n_env].
        samples = sampler.draw([float(low_i), float(high_i)], env_ids=env_ids, device=self.env.device)
        self.push_interval_s[env_ids] = samples

    def _set_max_push_tensor(self, values: Sequence[float]) -> None:
        tensor = torch.as_tensor(values, dtype=torch.float32, device=self.env.device).flatten()
        if tensor.numel() == 0:
            raise ValueError("max_push_vel must contain at least one value.")
        self._max_push_vel_tensor = tensor.clone()


class ActuatorRandomizerState(RandomizationTermBase):
    """Stateful actuator randomizer managing PD gain and RFI scales."""

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        params = cfg.params or {}

        kp_range = params.get("kp_range", [1.0, 1.0])
        kd_range = params.get("kd_range", [1.0, 1.0])
        rfi_lim_range = params.get("rfi_lim_range", [1.0, 1.0])

        self.enable_pd_gain = bool(params.get("enable_pd_gain", True))
        self.enable_rfi_lim = bool(params.get("enable_rfi_lim", False))

        # Ranges are config leaves ([lo, hi] pair, uniform; or a spec dict) kept as-is so an explicit
        # spec survives to the sampler.
        self.kp_range: DistributionLike = kp_range
        self.kd_range: DistributionLike = kd_range
        self.rfi_lim_range: DistributionLike = rfi_lim_range

        self.rfi_lim = float(params.get("rfi_lim", 0.1))

        self.kp_scale: torch.Tensor | None = None
        self.kd_scale: torch.Tensor | None = None
        self.rfi_lim_scale: torch.Tensor | None = None

    # Per-quantity axis tags so kp / kd / rfi draw from independent keyed streams.
    _AXIS_KP, _AXIS_KD, _AXIS_RFI = 0, 1, 2

    def setup(self, sampler: TermSampler) -> None:
        env = self.env
        device = env.device
        num_envs = env.num_envs
        num_dof = env.num_dof

        self.kp_scale = torch.ones(num_envs, num_dof, dtype=torch.float32, device=device)
        self.kd_scale = torch.ones(num_envs, num_dof, dtype=torch.float32, device=device)
        self.rfi_lim_scale = torch.ones(num_envs, num_dof, dtype=torch.float32, device=device)

        term = _get_joint_action_term(env)
        if term is not None:
            term.attach_actuator_scales(self.kp_scale, self.kd_scale, self.rfi_lim_scale)
        else:
            logger.debug(
                "JointPositionActionTerm not ready during ActuatorRandomizerState.setup(); "
                "the term will attach shared actuator scales once its setup() runs."
            )

    def reset(self, env_ids: torch.Tensor | None, sampler: TermSampler) -> None:
        if self.kp_scale is None or self.kd_scale is None or self.rfi_lim_scale is None:
            raise RuntimeError("ActuatorRandomizerState.setup() must be called before reset().")

        idx = _ensure_env_ids_tensor(self.env, env_ids)
        if idx.numel() == 0:
            return

        device = self.env.device
        n_dof = self.env.num_dof
        # Per-(env, dof) draws: dof indices as a [1, n_dof] coord -> [n_env, n_dof]; the _AXIS_* ints
        # are the per-quantity STREAM coords keeping kp/kd/rfi mutually independent.
        dof_ids = torch.arange(n_dof)[None, :]

        if self.enable_pd_gain:
            self.kp_scale[idx] = sampler.draw(
                self.kp_range,
                env_ids=idx,
                coords=(self._AXIS_KP, dof_ids),
                device=device,
            )
            self.kd_scale[idx] = sampler.draw(
                self.kd_range,
                env_ids=idx,
                coords=(self._AXIS_KD, dof_ids),
                device=device,
            )
        else:
            self.kp_scale[idx] = 1.0
            self.kd_scale[idx] = 1.0

        if self.enable_rfi_lim:
            self.rfi_lim_scale[idx] = sampler.draw(
                self.rfi_lim_range,
                env_ids=idx,
                coords=(self._AXIS_RFI, dof_ids),
                device=device,
            )
        else:
            self.rfi_lim_scale[idx] = 1.0

    def step(self, sampler: TermSampler) -> None:
        """No per-step behaviour required."""

    @property
    def kp_scale_tensor(self) -> torch.Tensor:
        if self.kp_scale is None:
            raise RuntimeError("ActuatorRandomizerState.setup() has not been called yet.")
        return self.kp_scale

    @property
    def kd_scale_tensor(self) -> torch.Tensor:
        if self.kd_scale is None:
            raise RuntimeError("ActuatorRandomizerState.setup() has not been called yet.")
        return self.kd_scale

    @property
    def rfi_lim_scale_tensor(self) -> torch.Tensor:
        if self.rfi_lim_scale is None:
            raise RuntimeError("ActuatorRandomizerState.setup() has not been called yet.")
        return self.rfi_lim_scale


def setup_action_delay_buffers(
    env, sampler: TermSampler, *, ctrl_delay_step_range: Sequence[int], enabled: bool = True, **_
) -> None:
    """Initialize action delay index buffer during setup.

    Note: The action_queue itself is managed by the action manager.
    This only sets up the delay index that determines which queued action to use.
    """
    env._randomize_ctrl_delay = bool(enabled)
    env._ctrl_delay_step_range = list(ctrl_delay_step_range)

    if not enabled:
        return

    # Keyed discrete draw (per-env, reproducible) of the delay index in [lo, hi] inclusive -> [n_env].
    env.action_delay_idx = sampler.draw_int(
        int(ctrl_delay_step_range[0]),
        int(ctrl_delay_step_range[1]),
        env_ids=torch.arange(env.num_envs, device="cpu"),
    ).to(env.device)


def setup_torque_rfi(env, sampler: TermSampler, *, enabled: bool = False, rfi_lim: float = 0.1, **_) -> None:
    """Configure torque RFI at startup."""
    term = _get_joint_action_term(env)
    env._pending_torque_rfi = (bool(enabled), float(rfi_lim))
    if term is None:
        return
    term.configure_torque_rfi(enabled=env._pending_torque_rfi[0], rfi_lim=env._pending_torque_rfi[1])


def setup_dof_pos_bias(
    env,
    sampler: TermSampler,
    *,
    dof_pos_bias_range: DistributionLike,
    enabled: bool = False,
    **_,
) -> None:
    """Apply startup DOF position bias randomization.

    ``dof_pos_bias_range`` is a config range value — a ``[lo, hi]`` pair (uniform) or a spec dict, drawn via
    the bound sampler. A gaussian bias gives a smoother initial-pose jitter than uniform.
    """
    env._randomize_dof_pos_bias = bool(enabled)
    env._dof_pos_bias_range = dof_pos_bias_range

    if not enabled:
        return

    # Per-(env, dof): dof indices as a [1, n_dof] coord -> [n_env, n_dof].
    dof_ids = torch.arange(env.num_dof)[None, :]
    default_dof_pos_bias = sampler.draw(
        dof_pos_bias_range,
        env_ids=torch.arange(env.num_envs, device="cpu"),
        coords=(dof_ids,),
        device=env.device,
    )
    env.default_dof_pos = env.default_dof_pos_base + default_dof_pos_bias


def randomize_push_schedule(
    env,
    env_ids,
    *,
    sampler: TermSampler,
    push_interval_s: Sequence[float] | None = None,
    enabled: bool | None = None,
    max_push_vel: Sequence[float] | None = None,
    **_,
) -> None:
    """Resample push intervals for selected environments."""
    state = env.randomization_manager.get_state("push_randomizer_state")
    if state is None:
        raise AttributeError("PushRandomizerState is not registered with the randomization manager.")

    state.configure(enabled=enabled, push_interval_s=push_interval_s, max_push_vel=max_push_vel)
    env._randomize_push_robots = state.enabled
    env._max_push_vel = state.max_push_vel.clone()

    if not state.enabled:
        return

    idx = _ensure_env_ids_tensor(env, env_ids)
    if idx.numel() == 0:
        return

    state.zero_counters(idx)
    state.resample(idx, sampler)


def randomize_pd_gains(
    env,
    env_ids,
    *,
    sampler: TermSampler,
    kp_range: DistributionLike,
    kd_range: DistributionLike,
    enabled: bool = True,
    **_,
):
    """Randomize proportional and derivative gain scales.

    ``kp_range``/``kd_range`` are config leaves ([lo, hi] pair, uniform; or a spec dict), drawn via
    the bound sampler. When the actuator state is registered this delegates to it (passing the
    sampler through); otherwise it draws directly.
    """
    state = env.randomization_manager.get_state("actuator_randomizer_state")
    term = _get_joint_action_term(env)
    if state is None:
        if term is None:
            logger.warning("JointPositionActionTerm not found; PD gain randomization skipped.")
            return

        idx = _ensure_env_ids_tensor(env, env_ids)
        if idx.numel() == 0:
            return

        if not enabled:
            kp_scale, kd_scale = term.get_pd_scale_tensors()
            term.update_pd_scales(idx, torch.ones_like(kp_scale[idx]), torch.ones_like(kd_scale[idx]))
            return

        dof_ids = torch.arange(env.num_dof)[None, :]  # [1, n_dof] -> [n_env, n_dof]; ints 0/1 = streams
        kp_samples = sampler.draw(kp_range, env_ids=idx, coords=(0, dof_ids), device=env.device)
        kd_samples = sampler.draw(kd_range, env_ids=idx, coords=(1, dof_ids), device=env.device)
        term.update_pd_scales(idx, kp_samples, kd_samples)
        return

    state.enable_pd_gain = bool(enabled)
    state.kp_range = kp_range
    state.kd_range = kd_range
    state.reset(env_ids, sampler)


def randomize_rfi_limits(
    env,
    env_ids,
    *,
    sampler: TermSampler,
    rfi_lim_range: DistributionLike,
    enabled: bool = True,
    **_,
) -> None:
    """Randomize residual force injection limits.

    ``rfi_lim_range`` is a config range value — a ``[lo, hi]`` pair (uniform) or a spec dict, drawn via the
    bound sampler.
    """
    state = env.randomization_manager.get_state("actuator_randomizer_state")
    term = _get_joint_action_term(env)
    if state is None:
        if term is None:
            logger.warning("JointPositionActionTerm not found; RFI randomization skipped.")
            return

        idx = _ensure_env_ids_tensor(env, env_ids)
        if idx.numel() == 0:
            return

        if not enabled:
            term.update_rfi_scales(idx, torch.ones_like(term.get_rfi_scale_tensor()[idx]))
            return

        dof_ids = torch.arange(env.num_dof)[None, :]  # [1, n_dof] -> [n_env, n_dof]; int 2 = stream
        rfi_samples = sampler.draw(rfi_lim_range, env_ids=idx, coords=(2, dof_ids), device=env.device)
        term.update_rfi_scales(idx, rfi_samples)
        return

    state.enable_rfi_lim = bool(enabled)
    state.rfi_lim_range = rfi_lim_range
    state.reset(env_ids, sampler)


def randomize_action_delay(
    env,
    env_ids,
    *,
    sampler: TermSampler,
    ctrl_delay_step_range: Sequence[int] | None = None,
    enabled: bool | None = None,
    **_,
) -> None:
    """Randomize control delay indices.

    If ``ctrl_delay_step_range``/``enabled`` are omitted the values captured during
    ``setup_action_delay_buffers`` are reused.
    """
    if enabled is not None:
        env._randomize_ctrl_delay = bool(enabled)
    elif not hasattr(env, "_randomize_ctrl_delay"):
        raise AttributeError(
            "randomize_action_delay() requires setup_action_delay_buffers to run before it can infer 'enabled'."
        )

    if ctrl_delay_step_range is not None:
        env._ctrl_delay_step_range = list(ctrl_delay_step_range)
    elif not hasattr(env, "_ctrl_delay_step_range"):
        raise AttributeError(
            "randomize_action_delay() requires setup_action_delay_buffers "
            "to run before it can infer ctrl_delay_step_range."
        )

    if not env._randomize_ctrl_delay:
        return

    idx = _ensure_env_ids_tensor(env, env_ids)
    if idx.numel() == 0:
        return

    # Reset action queue in the action manager
    if hasattr(env.action_manager, "action_queue"):
        env.action_manager.action_queue[idx] *= 0.0

    delay_low = int(env._ctrl_delay_step_range[0])
    delay_high = int(env._ctrl_delay_step_range[1])

    # Keyed discrete delay index per env (reproducible per (term, env, episode)) -> [n_env].
    # Bounds ordering is validated inside draw_int (single source of truth).
    env.action_delay_idx[idx] = sampler.draw_int(delay_low, delay_high, env_ids=idx).to(env.device)


def randomize_dof_state(
    env,
    env_ids,
    *,
    sampler: TermSampler,
    joint_pos_scale_range: DistributionLike,
    joint_pos_bias_range: DistributionLike,
    joint_vel_range: DistributionLike,
    randomize_dof_pos_bias: bool = False,
    **_,
) -> None:
    """Randomize DOF positions and velocities.

    Each range is a config range value — a ``[lo, hi]`` pair (uniform) or a spec dict, drawn via the bound
    sampler. The three quantities (pos scale, pos bias, vel) draw on independent axes so they don't
    share a stream.
    """
    env._joint_pos_scale_range = joint_pos_scale_range
    env._joint_pos_bias_range = joint_pos_bias_range
    env._joint_vel_range = joint_vel_range
    env._randomize_dof_pos_bias = bool(randomize_dof_pos_bias)

    idx = _ensure_env_ids_tensor(env, env_ids)
    if idx.numel() == 0:
        return

    n_dof = env.num_dof
    dof_ids = torch.arange(n_dof)[None, :]  # [1, n_dof] -> [n_env, n_dof]; ints 0/1/2 = streams
    scale_factor = sampler.draw(joint_pos_scale_range, env_ids=idx, coords=(0, dof_ids), device=env.device)
    if randomize_dof_pos_bias:
        bias_offset = sampler.draw(joint_pos_bias_range, env_ids=idx, coords=(1, dof_ids), device=env.device)
    else:
        bias_offset = torch.zeros((idx.shape[0], n_dof), device=env.device)

    env.simulator.dof_pos[idx] = env.default_dof_pos[idx] * scale_factor + bias_offset
    env.simulator.dof_vel[idx] = sampler.draw(joint_vel_range, env_ids=idx, coords=(2, dof_ids), device=env.device)


@mujoco_required_field("body_ipos")
def randomize_base_com_startup(
    env,
    env_ids: Sequence[int] | torch.Tensor | None = None,
    *,
    sampler: TermSampler,
    base_com_range: BaseComRange | dict[str, DistributionLike],
    enabled: bool = True,
    **_,
) -> None:
    """Randomize base (torso) center of mass.

    Note: Uses ADDITION operation to offset CoM position (e.g., x: [-0.01, 0.01] m).

    ``base_com_range`` is a :class:`BaseComRange` (or an equivalent ``{x, y, z}`` dict); each axis is a
    ``[lo, hi]`` pair (uniform) or a ``{kind, low, high, mean, std}`` spec dict —
    honored identically on ALL backends (IsaacGym per-axis loop, MuJoCo ``randomize_field``, IsaacSim
    project-owned ``randomize_body_com``), all via the shared keyed
    :meth:`holosoma.utils.sampler.TermSampler.draw`. A gaussian spec is a truncated normal on
    ``[lo, hi]``; log_uniform requires positive bounds.
    """
    if isinstance(base_com_range, BaseComRange):
        base_com_range = {"x": base_com_range.x, "y": base_com_range.y, "z": base_com_range.z}
    env._randomize_base_com = bool(enabled)
    env._base_com_range = base_com_range
    if not enabled:
        return

    logger.info(
        f"[Randomization] Base CoM: "
        f"x={base_com_range.get('x', [0, 0])}, "
        f"y={base_com_range.get('y', [0, 0])}, "
        f"z={base_com_range.get('z', [0, 0])} (operation=add)"
    )

    simulator = env.simulator

    idx = _ensure_env_ids_tensor(env, env_ids)
    if idx.numel() == 0:
        return

    if simulator.get_simulator_type() == SimulatorType.ISAACGYM:
        gym = simulator.gym
        torso_name = env.robot_config.torso_name
        if not hasattr(simulator, "_base_com_bias"):
            simulator._base_com_bias = torch.zeros(
                env.num_envs, 3, dtype=torch.float, device=env.device, requires_grad=False
            )

        # Per-axis (x/y/z) draw for all envs at once, keyed per env; index per actor in the loop. CoM
        # offsets are commonly signed (e.g. x: [-0.025, 0.025]); gaussian truncates to the range,
        # log_uniform (correctly) raises on a non-positive bound at spec construction.
        # Draw on CPU: the values are consumed per-actor via .item() below (IsaacGym props are CPU),
        # matching the sibling friction/mass draws and avoiding a pointless GPU round-trip.
        axis_draws = [
            sampler.draw(base_com_range[ax], env_ids=idx, coords=(k,), device="cpu")
            for k, ax in enumerate(("x", "y", "z"))
        ]  # each [n_env]; stack the three per-axis streams into (n_env, 3)
        bias_all = torch.stack(axis_draws, dim=-1)  # (n_env, 3)
        for offset, env_id in enumerate(idx.tolist()):
            env_ptr = simulator.envs[env_id]
            actor = simulator.robot_handles[env_id]
            body_props = gym.get_actor_rigid_body_properties(env_ptr, actor)
            body_index = gym.find_actor_rigid_body_handle(env_ptr, actor, torso_name)
            if body_index < 0:
                raise RuntimeError(f"Body '{torso_name}' not found when randomizing base COM.")

            bias = bias_all[offset]
            simulator._base_com_bias[env_id] = bias
            body_props[body_index].com.x += bias[0].item()
            body_props[body_index].com.y += bias[1].item()
            body_props[body_index].com.z += bias[2].item()
            # recomputeInertia=False: a CoM shift must NOT rewrite the inertia tensor from geometry —
            # that would discard the authored inertia and diverge from IsaacSim (randomize_body_com
            # writes coms only) and MuJoCo (writes body_ipos only), both of which leave body_inertia
            # (the principal moments about the body CoM) unchanged.
            gym.set_actor_rigid_body_properties(env_ptr, actor, body_props, recomputeInertia=False)
    elif simulator.get_simulator_type() == SimulatorType.ISAACSIM:
        try:
            from isaaclab.managers import SceneEntityCfg
        except ImportError as exc:  # pragma: no cover - dependency optional
            raise RuntimeError("IsaacSim base COM randomization requires isaaclab.") from exc
        from holosoma.simulator.isaacsim.events import randomize_body_com

        torso_name = env.robot_config.torso_name
        env_ids_cpu = idx.to(device="cpu", dtype=torch.long)

        # One spec per axis (x, y, z).
        specs = [DistributionSpec.parse(base_com_range[axis]) for axis in ("x", "y", "z")]
        asset_cfg = SceneEntityCfg("robot", body_names=[torso_name])
        asset_cfg.resolve(simulator.scene)  # Required to avoid applying randomization to all bodies
        randomize_body_com(
            simulator,
            env_ids_cpu,
            asset_cfg,
            specs,
            operation="add",
            sampler=sampler,
        )
    elif simulator.get_simulator_type() == SimulatorType.MUJOCO:
        from holosoma.simulator.mujoco.backends.randomization import randomize_field

        # convert xyz -> axis index 0/1/2, passing each range (a [lo, hi] pair or spec dict) through
        # unchanged so an explicit per-axis spec survives to the sampler.
        base_com_range_remapped = {"xyz".index(key): value for key, value in base_com_range.items()}
        randomize_field(
            simulator,
            field=getattr(randomize_base_com_startup, MUJOCO_FIELD_ATTR),
            ranges=base_com_range_remapped,
            sampler=sampler,
            env_ids=idx,
            entity_names=[env.robot_config.torso_name],
            entity_type="body",
            operation="add",
        )

    else:  # pragma: no cover - defensive
        raise RandomizerNotSupportedError(
            f"Unsupported simulator type '{type(simulator).__name__}' for base COM randomization."
        )


@mujoco_required_field("body_mass")
def randomize_mass_startup(
    env,
    env_ids: Sequence[int] | torch.Tensor | None = None,
    *,
    sampler: TermSampler,
    enable_link_mass: bool = True,
    link_mass_range: DistributionLike = (1.0, 1.0),
    enable_base_mass: bool = True,
    added_mass_range: DistributionLike = (0.0, 0.0),
    enabled: bool = True,
    recompute_inertia: bool = True,
    **_,
) -> None:
    """Randomize link and base masses at startup.

    Note: link_mass_range uses SCALING (e.g., 0.9-1.2 = 90-120% of original),
          added_mass_range uses ADDITION (e.g., -1.0 to 3.0 kg offset).

    ``recompute_inertia`` (default True): each body's inertia is rescaled by the SAME factor its mass
    changed by (``m_after / m_before``), identically on all backends (explicit multiplicative ratio —
    NOT a from-geometry recompute — so it commutes with a separate inertia DR term). This mirrors the
    object mass term's contract. When False, mass changes and inertia is left untouched.

    Each range is a config range value — a ``[lo, hi]`` pair (uniform) or a ``{kind, low, high, mean, std}``
    spec dict — honored identically on ALL backends (IsaacGym per-actor loop, MuJoCo
    ``randomize_field``, IsaacSim via the project-owned ``randomize_rigid_body_mass``), all via the
    shared keyed :meth:`holosoma.utils.sampler.TermSampler.draw`. A gaussian spec is a truncated normal
    on ``[lo, hi]``; log_uniform requires positive bounds (so it suits the scale operation,
    ``link_mass_range``, and by design RAISES on a signed ``added_mass_range`` rather than silently
    mis-sampling).
    """
    if not enabled:
        return

    logger.info(
        f"[Randomization] Mass: "
        f"link_mass={link_mass_range} (operation=scale, enabled={enable_link_mass}), "
        f"base_mass={added_mass_range} (operation=add, enabled={enable_base_mass})"
    )

    simulator = env.simulator
    idx = _ensure_env_ids_tensor(env, env_ids)
    if idx.numel() == 0:
        return

    env._randomize_link_mass = bool(enable_link_mass)
    env._randomize_base_mass = bool(enable_base_mass)

    if simulator.get_simulator_type() == SimulatorType.ISAACGYM:
        gym = simulator.gym
        body_names = list(env.robot_config.randomize_link_body_names or [])
        torso_name = env.robot_config.torso_name
        if idx.numel() > 0:
            sample_env = idx[0].item()
            sample_env_ptr = simulator.envs[sample_env]
            sample_actor = simulator.robot_handles[sample_env]
            sample_props = gym.get_actor_rigid_body_properties(sample_env_ptr, sample_actor)
            if enable_link_mass and body_names:
                link_masses = [
                    float(sample_props[simulator._body_list.index(name)].mass)
                    for name in body_names
                    if name in simulator._body_list
                ]
                if link_masses:
                    logger.debug(
                        "[randomize_mass_startup][IsaacGym] default link mass range: "
                        f"min={min(link_masses):.6f}, max={max(link_masses):.6f}"
                    )
            if enable_base_mass and torso_name in simulator._body_list:
                base_mass = float(sample_props[simulator._body_list.index(torso_name)].mass)
                logger.debug(f"[randomize_mass_startup][IsaacGym] default torso mass: {base_mass:.6f}")
        # Pre-draw keyed tensors for all envs at once, then index per actor in the loop. Link scale is
        # per (env, body) keyed by the stable body index (stream 0, body ids on the trailing dim ->
        # [n_env, n_link]); base-mass add is per env (stream 1 -> [n_env]).
        # Warn on configured link bodies absent from this robot rather than silently dropping them:
        # MuJoCo (resolve_entity_ids) and IsaacSim (SceneEntityCfg.resolve) both RAISE on an unknown
        # body name, so silence here would be an inconsistent, backend-specific drop of an explicit request.
        if enable_link_mass:
            missing = [n for n in body_names if n not in simulator._body_list]
            if missing:
                logger.warning(
                    f"[randomize_mass_startup][IsaacGym] link body name(s) {missing} not found on the "
                    f"robot (body list: {simulator._body_list}); link-mass randomization skipped for them."
                )
        link_body_ids = [simulator._body_list.index(n) for n in body_names if n in simulator._body_list]
        link_scales = (
            sampler.draw(link_mass_range, env_ids=idx, coords=(0, torch.tensor(link_body_ids)[None, :]), device="cpu")
            if enable_link_mass and link_body_ids
            else None
        )
        base_deltas = (
            sampler.draw(added_mass_range, env_ids=idx, coords=(1,), device="cpu")
            if enable_base_mass and torso_name in simulator._body_list
            else None
        )
        for offset, env_id in enumerate(idx.tolist()):
            env_ptr = simulator.envs[env_id]
            actor = simulator.robot_handles[env_id]
            body_props = gym.get_actor_rigid_body_properties(env_ptr, actor)
            # Snapshot mass per touched body so inertia can be scaled by the exact ratio THIS write
            # produced (explicit m_after/m_before — NOT recomputeInertia=True, which recomputes from
            # geometry, a different result that would also clobber a prior inertia-DR scale).
            touched: dict[int, float] = {}
            if link_scales is not None:
                for col, body_index in enumerate(link_body_ids):
                    touched.setdefault(body_index, body_props[body_index].mass)
                    body_props[body_index].mass *= float(link_scales[offset, col])  # scale by factor
            if base_deltas is not None:
                base_index = simulator._body_list.index(torso_name)
                touched.setdefault(base_index, body_props[base_index].mass)
                body_props[base_index].mass += float(base_deltas[offset])  # add operation: offset
            if recompute_inertia:
                for body_index, m_before in touched.items():
                    if m_before > 0.0:
                        r = body_props[body_index].mass / m_before
                        inertia = body_props[body_index].inertia  # gymapi.Mat33, symmetric
                        inertia.x.x *= r
                        inertia.y.y *= r
                        inertia.z.z *= r
                        inertia.x.y *= r
                        inertia.y.x *= r
                        inertia.y.z *= r
                        inertia.z.y *= r
                        inertia.x.z *= r
                        inertia.z.x *= r
            # recomputeInertia=False: inertia is set explicitly above (or left untouched).
            gym.set_actor_rigid_body_properties(env_ptr, actor, body_props, recomputeInertia=False)
    elif simulator.get_simulator_type() == SimulatorType.ISAACSIM:
        try:
            from isaaclab.managers import SceneEntityCfg
        except ImportError as exc:  # pragma: no cover - defensive
            raise RuntimeError("IsaacSim mass randomization requires isaaclab.") from exc

        from holosoma.simulator.isaacsim.events import randomize_rigid_body_mass

        env_ids_cpu = idx.to(device="cpu", dtype=torch.long)

        if enable_link_mass:
            asset_cfg = SceneEntityCfg("robot", body_names=env.robot_config.randomize_link_body_names)
            asset_cfg.resolve(simulator.scene)  # Required to avoid applying randomization to all bodies
            randomize_rigid_body_mass(
                simulator,
                env_ids_cpu,
                asset_cfg,
                link_mass_range,
                operation="scale",
                sampler=sampler,
                recompute_inertia=recompute_inertia,
            )

        if enable_base_mass:
            asset_cfg = SceneEntityCfg("robot", body_names=[env.robot_config.torso_name])
            asset_cfg.resolve(simulator.scene)  # Required to avoid applying randomization to all bodies
            randomize_rigid_body_mass(
                simulator,
                env_ids_cpu,
                asset_cfg,
                added_mass_range,
                operation="add",
                sampler=sampler,
                axis=1,  # base-add stream, distinct from link-scale (axis 0) so a body in both lists decorrelates
                recompute_inertia=recompute_inertia,
            )
    elif simulator.get_simulator_type() == SimulatorType.MUJOCO:
        from holosoma.simulator.mujoco.backends.randomization import (
            _field_view,
            randomize_field,
            resolve_entity_ids,
            scale_inertia_by_mass_ratio,
        )

        mass_field = getattr(randomize_mass_startup, MUJOCO_FIELD_ATTR)

        def _mass_write(
            names: Sequence[str], ranges, *, operation: Literal["add", "scale", "abs"], axis_base: int
        ) -> None:
            # Resolve to body ids ourselves so the inertia rescale (which has no name API) uses the
            # same bodies, and snapshot mass BEFORE so the ratio matches the exact write this produced
            # (mirrors the object mass term + the IsaacGym/IsaacSim explicit m_after/m_before scale).
            body_ids = torch.tensor(
                resolve_entity_ids(simulator.backend.model, list(names), "body"),
                device=simulator.sim_device,
                dtype=torch.long,
            )
            mass_before = _field_view(simulator, "body_mass")[:, body_ids].clone() if recompute_inertia else None
            randomize_field(
                simulator,
                field=mass_field,
                ranges=ranges,
                sampler=sampler,
                env_ids=idx,
                entity_ids=body_ids,
                operation=operation,
                axis_base=axis_base,
            )
            if recompute_inertia:
                scale_inertia_by_mass_ratio(simulator, body_ids, mass_before)

        # each range is a pair or spec dict, passed to randomize_field which builds the spec.
        if enable_link_mass:
            _mass_write(
                env.robot_config.randomize_link_body_names, link_mass_range, operation="scale", axis_base=0
            )  # link-scale stream
        if enable_base_mass:
            _mass_write(
                [env.robot_config.torso_name], added_mass_range, operation="add", axis_base=1
            )  # base-add stream (distinct from link-scale)

    else:  # pragma: no cover - defensive
        raise RandomizerNotSupportedError(
            f"Mass randomization not supported for simulator type '{type(simulator).__name__}'."
        )


def _draw_bucketed_per_env(
    sampler: TermSampler,
    spec: DistributionLike,
    env_ids: torch.Tensor,
    num_buckets: int | None,
) -> torch.Tensor:
    """Draw one friction value per env, bucketed or continuous, on the PhysX (IsaacGym) path.

    ``num_buckets`` an int: fill that many buckets with the distribution's quantile values (keyed
    permute so the staircase is reproducible-but-shuffled per seed) and pick one bucket per env via
    the keyed selection — bounds the unique-material count at ``num_buckets`` regardless of env count.
    ``None``: one continuous keyed draw per env (~``num_envs`` unique values). Mirrors the IsaacSim
    material writer's bucket mechanism so both PhysX backends share one quantization policy.
    """
    if num_buckets is None:
        return sampler.draw(spec, env_ids=env_ids)  # [n_env], continuous
    column = quantiles(DistributionSpec.parse(spec), num_buckets, "cpu")[sampler.permute(num_buckets, (0,))]
    bucket_ids = sampler.draw_int(0, num_buckets - 1, env_ids=env_ids, coords=(1,))
    return column[bucket_ids]


@mujoco_required_field("geom_friction")
def randomize_friction_startup(
    env,
    env_ids: Sequence[int] | torch.Tensor | None = None,
    *,
    sampler: TermSampler,
    friction_range: DistributionLike,
    num_buckets: int | None = _MATERIAL_NUM_BUCKETS,
    enabled: bool = True,
    **_,
) -> None:
    """Randomize contact friction coefficients for robot rigid shapes.

    Note: Uses ABSOLUTE operation to set friction values (e.g., [0.5, 1.5]).

    ``friction_range`` is a config range value — a ``[lo, hi]`` pair (uniform) or a spec dict.

    ``num_buckets`` controls VALUE QUANTIZATION uniformly across all three backends (default 64):
    friction here draws through the shared keyed sampler, and an int quantizes each draw onto
    ``num_buckets`` stratified quantile values (an ``n``-atom staircase), while ``None`` draws
    continuously. Bucketing exists to respect PhysX's per-scene material cap (~64k); the default keeps
    that guard on the PhysX backends (IsaacGym + IsaacSim) and, for cross-backend value consistency,
    applies the same staircase on MuJoCo (which has no cap and could otherwise be continuous). Set
    ``None`` for a continuous marginal where the material count is safely under the cap. ``uniform`` is
    exact when continuous; ``gaussian``/``log_uniform`` match up to the bucket quantization.

    GRANULARITY is unified too: this term draws ONE friction per env (per robot) on every backend —
    IsaacGym sets it on each shape, IsaacSim via the material writer's ``per_env=True``, MuJoCo via
    ``randomize_field``'s ``shared_across_entities=True``. So a single robot contacting the floor has a
    single contact-friction coefficient (the legged_gym / IsaacGym convention), not a different value
    per link/geom.
    """
    env._randomize_friction = bool(enabled)
    env._friction_range = friction_range
    if not enabled:
        return

    logger.info(f"[Randomization] Friction: range={friction_range} (operation=abs, num_buckets={num_buckets})")

    idx = _ensure_env_ids_tensor(env, env_ids)
    if idx.numel() == 0:
        return

    simulator = env.simulator

    if simulator.get_simulator_type() == SimulatorType.ISAACGYM:
        # IsaacGym sets prop.friction directly. One friction per env -> ~num_envs unique PhysX
        # materials, so bucket (default) to bound that count under the cap, or draw continuous per env
        # when num_buckets is None. Same quantization policy as the IsaacSim branch below.
        gym = simulator.gym
        idx_cpu = idx.to(device="cpu", dtype=torch.long)
        friction_samples = _draw_bucketed_per_env(sampler, friction_range, idx_cpu, num_buckets)  # [n_env]
        for offset, env_id in enumerate(idx_cpu.tolist()):
            env_ptr = simulator.envs[env_id]
            actor = simulator.robot_handles[env_id]
            shape_props = gym.get_actor_rigid_shape_properties(env_ptr, actor)
            friction_value = float(friction_samples[offset])
            for prop in shape_props:
                prop.friction = friction_value
            gym.set_actor_rigid_shape_properties(env_ptr, actor, shape_props)
    elif simulator.get_simulator_type() == SimulatorType.ISAACSIM:
        try:
            from isaaclab.managers import SceneEntityCfg
        except ImportError as exc:  # pragma: no cover - defensive
            raise RuntimeError("IsaacSim friction randomization requires isaaclab.") from exc
        env_ids_cpu = idx.to(device="cpu", dtype=torch.long)

        from holosoma.simulator.isaacsim.events import randomize_rigid_body_material

        asset_cfg = SceneEntityCfg("robot", body_names=".*")
        asset_cfg.resolve(simulator.scene)  # Not stricly required, but a good practice

        randomize_rigid_body_material(
            simulator,
            env_ids_cpu,
            asset_cfg,
            static_friction=friction_range,
            dynamic_friction=friction_range,
            restitution=None,  # leave restitution at each shape's spawned value (friction-only term);
            # matches the IsaacGym/MuJoCo branches, which never touch restitution here.
            num_buckets=num_buckets,  # same quantization policy as the IsaacGym branch
            per_env=True,  # one friction per robot (all shapes share), matching IsaacGym/MuJoCo
            sampler=sampler,
        )

    elif simulator.get_simulator_type() == SimulatorType.MUJOCO:
        # MuJoCo writes geom_friction directly (no material cap). num_buckets matches the PhysX
        # backends' quantization; shared_across_entities gives the whole robot ONE friction (per-env
        # granularity) instead of an independent per-geom draw, matching IsaacGym/IsaacSim.
        from holosoma.simulator.mujoco.backends.randomization import randomize_field

        randomize_field(
            simulator,
            field=getattr(randomize_friction_startup, MUJOCO_FIELD_ATTR),
            ranges={0: friction_range},
            sampler=sampler,
            env_ids=idx,
            operation="abs",
            num_buckets=num_buckets,
            shared_across_entities=True,  # one friction per robot (all geoms share), matching IsaacGym
        )

    else:  # pragma: no cover - defensive
        raise RandomizerNotSupportedError(
            f"Unsupported simulator type '{type(simulator).__name__}' for friction randomization."
        )


def randomize_robot_rigid_body_material_startup(
    env,
    env_ids: Sequence[int] | torch.Tensor | None = None,
    *,
    sampler: TermSampler,
    static_friction_range: DistributionLike,
    dynamic_friction_range: DistributionLike,
    restitution_range: DistributionLike,
    enabled: bool = True,
    **_,
) -> None:
    """Randomize robot rigid body material properties (friction, restitution). IsaacSim-only.

    Each range is a config range value — a ``[lo, hi]`` pair (uniform) or a spec dict — honored via the
    quantile-bucket scheme: IsaacSim MUST bucket (PhysX caps unique materials per scene), so the
    per-shape marginal is a 64-atom staircase approximation of the requested distribution. ``uniform``
    is exact; ``gaussian``/``log_uniform`` are approximate.
    """
    if not enabled:
        return

    idx = _ensure_env_ids_tensor(env, env_ids)
    if idx.numel() == 0:
        return

    simulator = env.simulator
    if simulator.get_simulator_type() != SimulatorType.ISAACSIM:
        raise RandomizerNotSupportedError(
            f"randomize_robot_rigid_body_material_startup only supports IsaacSim, got {type(simulator).__name__}"
        )

    try:
        from isaaclab.managers import SceneEntityCfg
    except ImportError as exc:  # pragma: no cover - defensive
        raise RuntimeError("IsaacSim material randomization requires isaaclab.") from exc
    from holosoma.simulator.isaacsim.events import randomize_rigid_body_material

    env_ids_cpu = idx.to(device="cpu", dtype=torch.long)

    asset_cfg = SceneEntityCfg("robot", body_names=".*")
    asset_cfg.resolve(simulator.scene)

    randomize_rigid_body_material(
        simulator,
        env_ids_cpu,
        asset_cfg,
        static_friction=static_friction_range,
        dynamic_friction=dynamic_friction_range,
        restitution=restitution_range,
        num_buckets=_MATERIAL_NUM_BUCKETS,
        sampler=sampler,
    )


def configure_torque_rfi(
    env,
    env_ids,
    *,
    sampler: TermSampler,
    enabled: bool | None = None,
    rfi_lim: float | None = None,
    **_,
) -> None:
    """Toggle torque RFI injection flag."""
    prev_enabled, prev_lim = env._pending_torque_rfi
    enabled_flag = prev_enabled if enabled is None else bool(enabled)
    rfi_limit = prev_lim if rfi_lim is None else float(rfi_lim)
    env._pending_torque_rfi = (enabled_flag, rfi_limit)

    state = env.randomization_manager.get_state("actuator_randomizer_state")
    if state is not None:
        state.enable_rfi_lim = enabled_flag
    term = _get_joint_action_term(env)
    if term is not None:
        term.configure_torque_rfi(enabled=enabled_flag, rfi_lim=rfi_limit)


def apply_pushes(
    env,
    *,
    sampler: TermSampler,
    enabled: bool | None = None,
    push_interval_s: Sequence[float] | None = None,
    max_push_vel: Sequence[float] | None = None,
    **_,
) -> None:
    """Apply random pushes based on the current schedule."""
    state = env.randomization_manager.get_state("push_randomizer_state")
    if state is None:
        raise AttributeError("PushRandomizerState is not registered with the randomization manager.")

    state.configure(enabled=enabled, push_interval_s=push_interval_s, max_push_vel=max_push_vel)
    env._push_robots_enabled = state.enabled

    if env.is_evaluating or not state.enabled:
        return

    push_robot_env_ids = state.due_envs(env.dt)
    if push_robot_env_ids.numel() == 0:
        return

    state.zero_counters(push_robot_env_ids)
    state.resample(push_robot_env_ids, sampler)
    env._max_push_vel = state.max_push_vel.clone()
    env._push_robots(push_robot_env_ids)
