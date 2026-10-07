"""World-frame external-force accumulator behind ``BaseSimulator.apply_external_force``.

Internal helper composed onto the simulator as ``self._external_wrench`` (like ``SimulatorBridge`` /
``VirtualGantry``). ``apply`` ADDS into a per-actor ``[num_envs, n_bodies, 6]`` world-frame buffer
(so multiple callers per substep compose); ``flush`` SETS native buffers to it via the backend's
``_write_external_wrench_native`` hook, then zeroes. Hence one contract on every backend regardless of
native force semantics: a force lasts one substep, re-apply each step to sustain it. See
``BaseSimulator.apply_external_force`` for the user-facing API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from holosoma.utils.safe_torch_import import torch

if TYPE_CHECKING:
    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator
    from holosoma.simulator.types import EnvIds

# The actor name reserved for the robot articulation across all backends.
ROBOT_ACTOR_NAME = "robot"


@dataclass
class WrenchTarget:
    """One actor's live world-frame wrench accumulator, passed to the backend flush hook."""

    actor_name: str
    is_robot: bool
    """Robot articulation vs scene object: backends flush the two through different handles."""
    body_names: list[str]
    """One per column of ``wrench`` (holosoma body order); a single entry for objects."""
    # wrench is mutated in place (never rebound) so it survives the warp backend's graph capture.
    wrench: torch.Tensor
    """``[num_envs, len(body_names), 6]`` (force[:3] + torque[3:] at CoM), on the sim device."""
    touched_cols: set[int] = field(default_factory=set)
    """Columns that received a force this substep."""
    # write_cols = touched this substep UNION last substep, so a released body gets one zero-write while
    # bodies we never touched are left alone (another writer's forces on them are not clobbered).
    write_cols: list[int] = field(default_factory=list)
    """Columns the backend writes this flush; set by ``flush`` before calling the backend."""
    prev_touched: set[int] = field(default_factory=set)
    """Columns touched last substep; managed by ``flush``."""


class ExternalWrenchAccumulator:
    """World-frame wrench accumulator for one simulator (see module docstring).

    Per-actor buffers are allocated lazily, so a sim that never applies a force pays nothing.
    """

    def __init__(self, sim: BaseSimulator) -> None:
        self._sim = sim
        self._targets: dict[str, WrenchTarget] = {}

    def apply(
        self,
        actor_name: str,
        *,
        forces: torch.Tensor | None = None,
        torques: torch.Tensor | None = None,
        body_names: list[str] | None = None,
        env_ids: EnvIds | None = None,
    ) -> None:
        """Add a world-frame force and/or torque to a body for this substep (see ``apply_external_force``)."""
        if forces is None and torques is None:
            raise ValueError("apply_external_force: provide forces, torques, or both.")

        target = self._resolve_target(actor_name)
        env_ids_t = self._resolve_env_ids(env_ids)
        cols = self._resolve_body_columns(target, body_names)
        n_env = int(env_ids_t.numel())
        n_body = len(cols)

        # Broadcast whichever component(s) were given; zero-fill the omitted half to the same shape.
        force_block = _broadcast_wrench_component(forces, n_env, n_body, "forces") if forces is not None else None
        torque_block = _broadcast_wrench_component(torques, n_env, n_body, "torques") if torques is not None else None
        if force_block is None:
            force_block = torch.zeros_like(torque_block)
        if torque_block is None:
            torque_block = torch.zeros_like(force_block)

        # .to(): input may be on a different device than the sim buffer (IsaacGym's CPU pipeline
        # has device="cpu" but sim_device="cuda:0").
        wrench_block = torch.cat([force_block, torque_block], dim=-1).to(target.wrench.device)  # [N, B, 6]

        # index_put_ with an explicit [N, B] grid handles non-contiguous env+body selections and
        # accumulates (vs overwrites).
        col_idx = torch.as_tensor(cols, device=target.wrench.device, dtype=torch.long)
        env_grid = env_ids_t.view(n_env, 1).expand(n_env, n_body)
        col_grid = col_idx.view(1, n_body).expand(n_env, n_body)
        target.wrench.index_put_((env_grid, col_grid), wrench_block, accumulate=True)

        target.touched_cols.update(cols)

    def flush(self) -> None:
        """Write accumulated wrenches to native buffers (via the backend hook), then zero them.

        Called once per substep by ``BaseSimulator.flush_external_wrench``. Targets with nothing to
        write (no touch this or last substep) are skipped.
        """
        if not self._targets:
            return

        targets = []
        for target in self._targets.values():
            write_cols = target.touched_cols | target.prev_touched
            if not write_cols:
                continue
            target.write_cols = sorted(write_cols)
            targets.append(target)

        if targets:
            self._sim._write_external_wrench_native(targets)

        for target in targets:
            target.wrench.zero_()
            target.prev_touched = target.touched_cols
            target.touched_cols = set()

    def _resolve_target(self, actor_name: str) -> WrenchTarget:
        """Return (allocating on first use) the wrench accumulator for ``actor_name``."""
        target = self._targets.get(actor_name)
        if target is not None:
            return target

        sim = self._sim
        is_robot = actor_name == ROBOT_ACTOR_NAME
        if is_robot:
            body_names = list(sim.body_names)
        else:
            # The registry is the cross-backend source of truth for actor names (same names as
            # get/set_actor_states); an unknown name fails here with the available set.
            registered = sim.object_registry.list_all_objects()
            if actor_name not in registered:
                raise ValueError(
                    f"apply_external_force: unknown actor '{actor_name}'. Known actors: {sorted(registered)}."
                )
            body_names = [actor_name]  # objects are single-body on every backend

        wrench = torch.zeros(sim.num_envs, len(body_names), 6, dtype=torch.float32, device=sim.sim_device)
        target = WrenchTarget(actor_name=actor_name, is_robot=is_robot, body_names=body_names, wrench=wrench)
        self._targets[actor_name] = target
        return target

    def _resolve_body_columns(self, target: WrenchTarget, body_names: list[str] | None) -> list[int]:
        """Map requested body names to column indices in ``target.wrench`` (``None`` => all bodies)."""
        if body_names is None:
            return list(range(len(target.body_names)))
        name_to_col = {name: i for i, name in enumerate(target.body_names)}
        cols: list[int] = []
        for name in body_names:
            if name not in name_to_col:
                raise ValueError(
                    f"apply_external_force: body '{name}' is not a body of actor '{target.actor_name}'. "
                    f"Available: {target.body_names}"
                )
            cols.append(name_to_col[name])
        return cols

    def _resolve_env_ids(self, env_ids: EnvIds | None) -> torch.Tensor:
        """Normalise ``env_ids`` to a 1-D long tensor on the sim device (``None`` => all envs).

        Bounds are validated here so a bad env id fails with a clear ValueError instead of an
        out-of-bounds ``index_put_`` (a device-side fault on CUDA).
        """
        device = self._sim.sim_device
        if env_ids is None:
            return torch.arange(self._sim.num_envs, device=device, dtype=torch.long)
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=device, dtype=torch.long)
        env_ids = env_ids.to(device=device, dtype=torch.long)
        if env_ids.numel():
            lo, hi = env_ids.aminmax()
            if int(lo) < 0 or int(hi) >= self._sim.num_envs:
                raise ValueError(
                    f"apply_external_force: env_ids must be in [0, {self._sim.num_envs}), "
                    f"got range [{int(lo)}, {int(hi)}]."
                )
        return env_ids


def _broadcast_wrench_component(comp: torch.Tensor, n_env: int, n_body: int, what: str) -> torch.Tensor:
    """Broadcast a force/torque argument to ``[n_env, n_body, 3]`` float32.

    Accepts ``[3]`` (same on every env/body), ``[N, 3]`` (per-env, broadcast over bodies),
    ``[B, 3]`` (per-body, broadcast over envs), or ``[N, B, 3]`` (fully specified). When ``N == B``
    the 2-D forms are indistinguishable, so a 2-D input is REJECTED (pass an explicit ``[N, B, 3]``)
    rather than silently guessing an axis.
    """
    if comp.dim() == 1:
        if comp.shape[0] != 3:
            raise ValueError(f"apply_external_force: 1-D {what} must have shape [3], got {tuple(comp.shape)}")
        block = comp.view(1, 1, 3).expand(n_env, n_body, 3)
    elif comp.dim() == 2:
        if comp.shape[1] != 3:
            raise ValueError(
                f"apply_external_force: 2-D {what} must have shape [N,3] or [B,3], got {tuple(comp.shape)}"
            )
        rows = comp.shape[0]
        if n_env == n_body and rows == n_env and n_env != 1:
            raise ValueError(
                f"apply_external_force: 2-D {what} of shape {tuple(comp.shape)} is ambiguous when "
                f"n_env == n_body == {n_env} (per-env vs per-body). Pass an explicit [N,B,3] tensor."
            )
        if rows == n_env:
            block = comp.view(n_env, 1, 3).expand(n_env, n_body, 3)
        elif rows == n_body:
            block = comp.view(1, n_body, 3).expand(n_env, n_body, 3)
        else:
            raise ValueError(
                f"apply_external_force: 2-D {what} rows ({rows}) must equal n_env ({n_env}) or n_body ({n_body})"
            )
    elif comp.dim() == 3:
        if tuple(comp.shape[1:]) == (n_body, 3) and comp.shape[0] == n_env:
            block = comp
        else:
            raise ValueError(
                f"apply_external_force: 3-D {what} must have shape [{n_env},{n_body},3], got {tuple(comp.shape)}"
            )
    else:
        raise ValueError(f"apply_external_force: {what} must be 1/2/3-D, got {comp.dim()}-D")
    return block.to(dtype=torch.float32)
