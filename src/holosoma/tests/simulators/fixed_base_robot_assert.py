"""Headless cross-backend assertion harness for a FIXED-BASE robot (``asset.fix_base_link=True``).

Builds a sim for a chosen backend with the ``test-arm-fixed`` preset (3-DOF arm, welded base; see
``tests/simulators/_robot_presets.py``) plus one static pillar clear of the arm (for the mixed robot+object
write) and asserts the fixed-robot state contract:
the base is handled like a static (welded) scene object -- constant spawn pose, zero velocity, and one
root-write rule on every backend (a write whose pose equals the current pose is a no-op with the
velocity columns ignored; a pose change is a kinematic teleport on MuJoCo/IsaacSim and refused with
NotImplementedError on IsaacGym) -- while the arm's DOFs stay live (gravity, torques, DOF-state writes).

  1. SPAWN       robot_root_states == [origin + init.position, init quat xyzw, zeros(6)] per env;
                 all_root_states[robot rows], get_actor_states(["robot"]), base_quat and the base body
                 row of rigid_body_pos_w agree with it; base_angular_vel / base_linear_acc are zero
                 (required on MuJoCo, which exposes both on a fixed base; IsaacGym/IsaacSim expose
                 base_linear_acc only with the bridge enabled and no base_angular_vel: skipped, printed).
  2. STAYS PUT   1 s of zero torques: base pose unchanged (atol 1e-6); the tip (slider) drops by
                 > 1 cm under gravity and dof_pos changes -- the joints are dynamic, the base is not.
  3. SAME-POSE   set_actor_root_state_tensor_robots(env_ids, current pose with the quaternion NEGATED
                 (-q is the same rotation) + a NON-ZERO velocity) is accepted on every backend (IsaacGym
                 included) and moves nothing: root (zero velocity, the current sign), base body and tip
                 read back unchanged.
  4. TELEPORT    set_actor_root_state_tensor_robots(env_ids, +0.5 m z): base pose reads back at the
                 target, velocity stays zero, dof_pos is preserved, the tip moves by the same +0.5 m.
                 IsaacGym MUST refuse a fixed-base pose change with the documented NotImplementedError
                 (PhysX keeps a welded root where it spawned); a silent accept is a failure there.
  4b. MIXED      set_actor_states(["robot", pillar], ...) moving BOTH by +0.5 m z in one write: IsaacGym
                 raises the documented NotImplementedError before anything is written (all-or-nothing:
                 the pillar reads back unchanged); MuJoCo/IsaacSim apply both poses (velocity ignored).
  5. UNIFIED     set_actor_states(["robot"], ...) to the spawn pose with a NON-ZERO velocity column:
                 the pose is applied (a same-pose no-op on IsaacGym, where TELEPORT was refused), the
                 velocity is ignored (reads back zero), exactly as for a static scene object. A raise
                 is a failure on every backend.
  6. DOF WRITE   set_dof_state_tensor_robots to [0.3, -0.3, 0.02] reads back within 5e-2 after one
                 step; the base pose is untouched.
  7. PRISMATIC   +2 N on slide_joint for 0.3 s drives dof_pos[:, 2] into (0.01, 0.055]; -2 N for
                 0.3 s brings it back down (the joint range is [0, 0.05]).
  8. PER-ENV     (num_envs > 1) distinct env origins give distinct base positions.

Exits 0 on success, 1 with FAIL lines otherwise; writes "OK" to ``--result-file`` on success.
Mirrors static_move_assert.py / all_root_states_unified_assert.py.

Usage:
  python fixed_base_robot_assert.py --simulator mujoco                 # ClassicBackend, 1 env, cpu
  python fixed_base_robot_assert.py --simulator mjwarp   --num-envs 4  # WarpBackend, cuda
  python fixed_base_robot_assert.py --simulator isaacgym --num-envs 4  # IsaacGym, cuda
  python fixed_base_robot_assert.py --simulator isaacsim --num-envs 4  # IsaacSim, cuda
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from typing import TYPE_CHECKING, Any, Callable

# tests/simulators/ has an isaacsim/ subpackage that would shadow the real IsaacSim package if it
# lands on sys.path[0] when run as a script -- drop it (mirrors the sibling harnesses).
if sys.path and sys.path[0].endswith("tests/simulators"):
    sys.path.pop(0)

from holosoma.config_types.scene import RigidObjectConfig, SceneConfig
from holosoma.utils.sim_utils import setup_simulation_environment
from tests.simulators import _robot_presets
from tests.simulators._sim_harness import build_run_sim_config, run_and_hard_exit, steps_for_seconds

if TYPE_CHECKING:
    import torch

# Autotools-convention skip exit code; kept in sync with ``_run_harness.SKIP_EXIT_CODE``.
SKIP_EXIT_CODE = 77

ROBOT_KEY = "test-arm-fixed"
ENV_SPACING = 5.0  # metres between env origins along x (num_envs > 1)
# One static box per env for the MIXED check, on the world -x side of the base. The arm (base at z 0.5,
# yawed 90 deg) reaches ~0.63 m along +y and droops downward, so nothing touches the pillar.
PILLAR = "pillar"
PILLAR_POSITION = [-0.6, 0.0, 0.15]
_SMALL_BOX = "holosoma/data/scene_objects/boxes/small_box.urdf"
_SMALL_BOX_USD = "holosoma/data/scene_objects/boxes/small_box.usda"
PILLAR_SCENE = SceneConfig(
    rigid_objects={
        PILLAR: RigidObjectConfig(urdf_file=_SMALL_BOX, usd_file=_SMALL_BOX_USD, position=PILLAR_POSITION, fixed=True)
    }
)
MIXED_DZ = 0.5

POSE_ATOL = 1e-5  # agreement between the spawn pose and every read path
STAY_ATOL = 1e-6  # base pose drift over the zero-torque window
TIP_STILL_ATOL = 5e-3  # tip creep allowed across a same-pose write (IsaacSim steps physics to refresh reads)
FREEFALL_SECONDS = 1.0
TIP_DROP_MIN = 0.01  # metres the tip must fall in FREEFALL_SECONDS
TELEPORT_DZ = 0.5
DOF_TARGET = [0.3, -0.3, 0.02]  # [shoulder rad, elbow rad, slide m]
DOF_ATOL = 5e-2
SLIDE_FORCE = 2.0  # N on slide_joint
SLIDE_SECONDS = 0.3
SLIDE_MIN, SLIDE_MAX = 0.01, 0.055  # joint range is [0, 0.05]; 5 mm soft-limit slack
HOLD_KP, HOLD_KD = 20.0, 1.0  # harness-side PD hold on the hinge joints while the slider is driven
# IsaacSim refreshes its read cache from PhysX on a sim step; MuJoCo/IsaacGym reflect writes at once.
ISAACSIM_REFRESH_STEPS = 2


def _build(simulator: str, num_envs: int) -> tuple[Any, Any]:
    """Build the sim (bridge + gantry off) with the fixed arm and the static pillar. Returns ``(sim, config)``."""
    _robot_presets.register()
    config = build_run_sim_config(simulator, "empty", ROBOT_KEY, "terrain_locomotion_plane")
    device = "cpu" if simulator == "mujoco" else "cuda:0"
    config = dataclasses.replace(
        config,
        device=device,
        scene=PILLAR_SCENE,
        training=dataclasses.replace(config.training, num_envs=num_envs),
    )
    env, device, _app = setup_simulation_environment(config, device=device)
    import torch

    sim = env.sim
    sim.set_headless(True)
    sim.setup()
    sim.setup_terrain()
    sim.load_assets()
    origins = torch.zeros(num_envs, 3, device=device)
    if num_envs > 1:
        origins[:, 0] = torch.arange(num_envs, device=device, dtype=torch.float32) * ENV_SPACING
    base_init = torch.tensor(config.robot.init_state.root_state_list(), device=device, dtype=torch.float32)
    sim.create_envs(num_envs, origins, base_init)
    sim.prepare_sim()
    sim.install_plugins()
    return sim, config


def _fmt(t: torch.Tensor) -> list[float]:
    return [round(float(v), 4) for v in t.reshape(-1)]


class _Arm:
    """Live-sim accessors shared by the checks: expected spawn state, root/DOF/tip reads, torque loops."""

    def __init__(self, sim: Any, config: Any) -> None:
        import torch

        self.torch = torch
        self.sim = sim
        self.num_envs = int(sim.num_envs)
        self.num_dof = int(sim.num_dof)
        self.env_ids = torch.arange(self.num_envs, device=sim.sim_device)
        self.base_idx = sim.find_rigid_body_indice("base_link")
        self.tip_idx = sim.find_rigid_body_indice("slider")
        self.is_isaacsim = sim.get_simulator_type().value == "isaacsim"
        self.is_isaacgym = sim.get_simulator_type().value == "isaacgym"
        self.effort = sim.get_dof_limits_properties()[2].to(device=sim.sim_device, dtype=torch.float32)
        init = config.robot.init_state
        # Expected [num_envs, 13]: world position (the backend's live env origins: IsaacSim lays envs out
        # on its own grid and ignores the passed ones), xyzw quaternion, zero velocity.
        origins = torch.as_tensor(sim.env_origins, device=sim.sim_device, dtype=torch.float32)[: self.num_envs]
        self.spawn = torch.zeros(self.num_envs, 13, device=sim.sim_device)
        self.spawn[:, :3] = torch.tensor(init.position, device=sim.sim_device) + origins
        self.spawn[:, 3:7] = torch.tensor(init.orientation_xyzw, device=sim.sim_device)

    def root(self) -> torch.Tensor:
        """``robot_root_states`` rows for every env, ``[num_envs, 13]``."""
        return self.sim.robot_root_states[self.env_ids].clone()

    def dof_pos(self) -> torch.Tensor:
        return self.sim.dof_pos[self.env_ids].clone()

    def body_pos(self, body_idx: int) -> torch.Tensor:
        """World position of one robot body per env after a tensor refresh, ``[num_envs, 3]``."""
        self.sim.refresh_sim_tensors()
        return self.sim.rigid_body_pos_w[self.env_ids, body_idx].clone()

    def zero_torques(self) -> torch.Tensor:
        return self.torch.zeros(self.num_envs, self.num_dof, device=self.sim.sim_device)

    def hold_torques(self, hold_q: torch.Tensor, slide_force: float) -> torch.Tensor:
        """PD-hold the two hinge joints at ``hold_q`` and push the slider with ``slide_force`` (N)."""
        q = self.sim.dof_pos[self.env_ids]
        qd = self.sim.dof_vel[self.env_ids]
        torques = self.zero_torques()
        torques[:, :2] = -HOLD_KP * (q[:, :2] - hold_q) - HOLD_KD * qd[:, :2]
        torques[:, 2] = slide_force
        return torques.clamp(-self.effort, self.effort)

    def run_steps(self, n_steps: int, torques: Callable[[], torch.Tensor]) -> None:
        """Advance ``n_steps`` physics steps, applying ``torques()`` before each; refresh at the end."""
        for _ in range(n_steps):
            self.sim.refresh_sim_tensors()
            self.sim.apply_torques_at_dof(torques())
            self.sim.simulate_at_each_physics_step()
        self.sim.refresh_sim_tensors()

    def run(self, seconds: float, torques: Callable[[], torch.Tensor]) -> None:
        self.run_steps(steps_for_seconds(self.sim, seconds), torques)

    def settle_reads(self) -> None:
        """IsaacSim only: step so the PhysX read cache reflects a write (see ISAACSIM_REFRESH_STEPS)."""
        if self.is_isaacsim:
            self.run_steps(ISAACSIM_REFRESH_STEPS, self.zero_torques)

    def write_dof_pos(self, target: torch.Tensor) -> None:
        """``set_dof_state_tensor_robots`` with zero velocity, in each backend's expected layout."""
        states = self.torch.zeros(self.num_envs, self.num_dof, 2, device=self.sim.sim_device)
        states[..., 0] = target
        if self.is_isaacsim:
            self.sim.set_dof_state_tensor_robots(self.env_ids, states)  # 3D [num_envs, num_dof, 2]
        else:
            self.sim.set_dof_state_tensor_robots(self.env_ids, states.reshape(-1, 2))  # [num_envs*num_dof, 2]


def _base_unchanged(arm: _Arm, root_before: torch.Tensor, base_before: torch.Tensor, action: str) -> list[str]:
    """Both the root-state mirror and the base body's physics position must still equal the ``*_before`` reads."""
    torch = arm.torch
    fails: list[str] = []
    if not torch.allclose(arm.root(), root_before, atol=STAY_ATOL):
        fails.append(f"{action} moved the base: {_fmt(root_before)} -> {_fmt(arm.root())}")
    # robot_root_states is a mirror on MuJoCo; the base body's physics position is the real check.
    base_after = arm.body_pos(arm.base_idx)
    if not torch.allclose(base_after, base_before, atol=STAY_ATOL):
        fails.append(f"{action} moved the base body: {_fmt(base_before)} -> {_fmt(base_after)}")
    return fails


# The documented IsaacGym refusal (isaacgym/root_state.py fixed_base_robot_rows); any other NotImplementedError
# is a failure.
ISAACGYM_REFUSAL = "cannot move a fixed-base robot"


def _root_write_refused(arm: _Arm, label: str, exc: NotImplementedError) -> list[str]:
    """IsaacGym documents refusing a fixed-base root write that changes the pose (PhysX would ignore it);
    elsewhere, or with any other message, a raise is a failure."""
    if arm.is_isaacgym and ISAACGYM_REFUSAL in str(exc):
        print(f"  {label} skipped (fixed-base pose change refused: {exc})")
        return []
    return [f"{label} raised NotImplementedError: {exc}"]


def _check_same_pose(arm: _Arm) -> list[str]:
    """Writing the current pose back (quaternion negated: the same rotation; with a velocity the welded base
    must ignore) is a no-op everywhere."""
    torch, sim = arm.torch, arm.sim
    root_before, base_before, tip_before = arm.root(), arm.body_pos(arm.base_idx), arm.body_pos(arm.tip_idx)
    target = root_before.clone()
    target[:, 3:7] = -root_before[:, 3:7]  # -q == q; root_pose_changed must treat it as the same pose
    target[:, 7:10] = torch.tensor([0.3, 0.0, 0.0], device=sim.sim_device)
    try:
        sim.set_actor_root_state_tensor_robots(arm.env_ids, target)
    except NotImplementedError as exc:
        return [f"same-pose set_actor_root_state_tensor_robots raised NotImplementedError: {exc}"]
    arm.settle_reads()
    sim.refresh_sim_tensors()  # read the physics pose back (IsaacGym: the raw buffer held the registered spawn)
    fails = _base_unchanged(arm, root_before, base_before, "same-pose write")
    tip = arm.body_pos(arm.tip_idx)
    # settle_reads steps IsaacSim briefly, so the free-hanging tip may creep under gravity; a teleport is not small.
    if not torch.allclose(tip, tip_before, atol=TIP_STILL_ATOL):
        fails.append(f"same-pose write moved the tip: {_fmt(tip_before)} -> {_fmt(tip)}")
    print(f"  SAME-POSE root={_fmt(arm.root()[0, :3])} vel={_fmt(arm.root()[0, 7:])} ok={not fails}")
    return fails


def _check_spawn(arm: _Arm) -> list[str]:
    torch, sim, spawn = arm.torch, arm.sim, arm.spawn
    fails: list[str] = []
    root = arm.root()
    if tuple(root.shape) != (arm.num_envs, 13):
        return [f"robot_root_states[env_ids] shape {tuple(root.shape)} != ({arm.num_envs}, 13)"]
    if not torch.allclose(root, spawn, atol=POSE_ATOL):
        fails.append(f"robot_root_states {_fmt(root)} != spawn {_fmt(spawn)}")
    unified = sim.all_root_states[sim.get_actor_indices(["robot"], arm.env_ids)]
    if not torch.allclose(unified, spawn, atol=POSE_ATOL):
        fails.append(f"all_root_states[robot] {_fmt(unified)} != spawn")
    by_name = sim.get_actor_states(["robot"], arm.env_ids)
    if not torch.allclose(by_name, spawn, atol=POSE_ATOL):
        fails.append(f"get_actor_states(['robot']) {_fmt(by_name)} != spawn")
    base_quat = sim.base_quat[arm.env_ids]
    if not torch.allclose(base_quat, spawn[:, 3:7], atol=POSE_ATOL):
        fails.append(f"base_quat {_fmt(base_quat)} != spawn quat {_fmt(spawn[:, 3:7])}")
    base_body = arm.body_pos(arm.base_idx)
    if not torch.allclose(base_body, spawn[:, :3], atol=POSE_ATOL):
        fails.append(f"rigid_body_pos_w[base_link] {_fmt(base_body)} != spawn position {_fmt(spawn[:, :3])}")
    for name in ("base_angular_vel", "base_linear_acc"):
        view = getattr(sim, name, None)
        if view is None:
            if arm.is_isaacgym or arm.is_isaacsim:
                print(f"  SPAWN {name} not exposed on this backend, skipped")
                continue
            fails.append(f"{name} missing: MuJoCo exposes it on a fixed base")
            continue
        values = view[arm.env_ids]
        if float(values.abs().max()) > POSE_ATOL:
            fails.append(f"{name} {_fmt(values)} != 0 for a welded base")
    print(f"  SPAWN root={_fmt(root[0])} envs={arm.num_envs} ok={not fails}")
    return fails


def _check_stays_put(arm: _Arm) -> list[str]:
    root0, base0, tip0, q0 = arm.root(), arm.body_pos(arm.base_idx), arm.body_pos(arm.tip_idx), arm.dof_pos()
    arm.run(FREEFALL_SECONDS, arm.zero_torques)
    tip1, q1 = arm.body_pos(arm.tip_idx), arm.dof_pos()
    drop = tip0[:, 2] - tip1[:, 2]
    dof_delta = (q1 - q0).abs().max(dim=1).values
    fails = _base_unchanged(arm, root0, base0, "zero torque")
    if not bool((drop > TIP_DROP_MIN).all()):
        fails.append(f"tip did not fall under gravity: drop {_fmt(drop)} (need > {TIP_DROP_MIN})")
    if not bool((dof_delta > 1e-3).all()):
        fails.append(f"dof_pos did not change under gravity: {_fmt(q0)} -> {_fmt(q1)}")
    print(f"  STAYS-PUT tip_drop={_fmt(drop)} dof={_fmt(q1[0])} ok={not fails}")
    return fails


def _check_teleport(arm: _Arm) -> list[str]:
    torch, sim = arm.torch, arm.sim
    q_before, tip_before, root_before = arm.dof_pos(), arm.body_pos(arm.tip_idx), arm.root()
    target = root_before.clone()
    target[:, 2] += TELEPORT_DZ
    target[:, 7:] = 0.0
    try:
        sim.set_actor_root_state_tensor_robots(arm.env_ids, target)
    except NotImplementedError as exc:
        return _root_write_refused(arm, "TELEPORT set_actor_root_state_tensor_robots", exc)
    if arm.is_isaacgym:
        # The documented contract: PhysX keeps a welded root where it spawned, so IsaacGym must refuse.
        return ["TELEPORT set_actor_root_state_tensor_robots did not raise NotImplementedError on IsaacGym"]
    arm.settle_reads()
    root, tip = arm.root(), arm.body_pos(arm.tip_idx)
    fails: list[str] = []
    if not torch.allclose(root[:, :7], target[:, :7], atol=POSE_ATOL):
        fails.append(f"teleported base pose {_fmt(root[:, :7])} != target {_fmt(target[:, :7])}")
    if float(root[:, 7:].abs().max()) > POSE_ATOL:
        fails.append(f"teleported base reports velocity {_fmt(root[:, 7:])}")
    if not torch.allclose(arm.dof_pos(), q_before, atol=DOF_ATOL):
        fails.append(f"teleport changed dof_pos: {_fmt(q_before)} -> {_fmt(arm.dof_pos())}")
    want = torch.tensor([0.0, 0.0, TELEPORT_DZ], device=sim.sim_device).expand(arm.num_envs, 3)
    if not torch.allclose(tip - tip_before, want, atol=2e-3):
        fails.append(f"tip displacement {_fmt(tip - tip_before)} != {_fmt(want[0])}")
    print(f"  TELEPORT base_z={_fmt(root[:, 2])} tip_dz={_fmt(tip[:, 2] - tip_before[:, 2])} ok={not fails}")
    return fails


def _check_mixed_write(arm: _Arm) -> list[str]:
    """One ``set_actor_states`` over the robot AND the static pillar, moving both by +MIXED_DZ: IsaacGym
    refuses the robot row before writing anything (all-or-nothing, the pillar stays), MuJoCo/IsaacSim apply
    both poses and ignore the velocity columns."""
    torch, sim = arm.torch, arm.sim
    names = ["robot", PILLAR]
    n = arm.num_envs
    before = sim.get_actor_states(names, arm.env_ids).clone()  # [2n, 13], name-major: robot rows then pillar rows
    tip_before = arm.body_pos(arm.tip_idx)
    target = before.clone()
    target[:, 2] += MIXED_DZ
    target[:n, 7:10] = torch.tensor([1.0, 0.0, 0.0], device=sim.sim_device)  # robot rows: a velocity to ignore
    try:
        sim.set_actor_states(names, arm.env_ids, target)
    except NotImplementedError as exc:
        fails = _root_write_refused(arm, "MIXED set_actor_states(['robot', pillar])", exc)
        after = sim.get_actor_states(names, arm.env_ids)
        if not torch.allclose(after, before, atol=POSE_ATOL):
            fails.append(f"refused mixed write still changed actor states: {_fmt(before)} -> {_fmt(after)}")
        if not torch.allclose(arm.body_pos(arm.tip_idx), tip_before, atol=TIP_STILL_ATOL):
            fails.append("refused mixed write moved the tip")
        print(f"  MIXED refused, pillar z={_fmt(after[n:, 2])} ok={not fails}")
        return fails
    if arm.is_isaacgym:
        return ["MIXED set_actor_states(['robot', pillar]) did not raise NotImplementedError on IsaacGym"]
    arm.settle_reads()
    after = sim.get_actor_states(names, arm.env_ids)
    fails = []
    if not torch.allclose(after[:, :7], target[:, :7], atol=POSE_ATOL):
        fails.append(f"mixed write poses {_fmt(after[:, :7])} != target {_fmt(target[:, :7])}")
    if float(after[:n, 7:].abs().max()) > POSE_ATOL:
        fails.append(f"mixed write left a velocity on the welded base: {_fmt(after[:n, 7:])}")
    if not torch.allclose(arm.root()[:, :7], target[:n, :7], atol=POSE_ATOL):
        fails.append(f"robot_root_states {_fmt(arm.root()[:, :7])} != mixed-write robot pose")
    if not torch.allclose(arm.body_pos(arm.base_idx), target[:n, :3], atol=POSE_ATOL):
        fails.append(f"rigid_body_pos_w[base_link] {_fmt(arm.body_pos(arm.base_idx))} != mixed-write robot position")
    want = torch.tensor([0.0, 0.0, MIXED_DZ], device=sim.sim_device).expand(n, 3)
    if not torch.allclose(arm.body_pos(arm.tip_idx) - tip_before, want, atol=2e-3):
        fails.append(f"tip displacement {_fmt(arm.body_pos(arm.tip_idx) - tip_before)} != {_fmt(want[0])}")
    print(f"  MIXED base_z={_fmt(after[:n, 2])} pillar_z={_fmt(after[n:, 2])} ok={not fails}")
    return fails


def _check_unified_write(arm: _Arm) -> list[str]:
    torch, sim, spawn = arm.torch, arm.sim, arm.spawn
    target = spawn.clone()  # back to the spawn pose ...
    target[:, 7:10] = torch.tensor([1.0, 0.0, 0.0], device=sim.sim_device)  # ... with a velocity that must
    target[:, 10:13] = torch.tensor([0.0, 0.0, 0.5], device=sim.sim_device)  # be ignored for a welded base
    try:
        sim.set_actor_states(["robot"], arm.env_ids, target)
    except NotImplementedError as exc:
        # The spawn pose is the current pose on IsaacGym (TELEPORT was refused), so a raise is a failure everywhere.
        return [f"set_actor_states(['robot']) to the spawn pose raised NotImplementedError: {exc}"]
    arm.settle_reads()
    fails: list[str] = []
    reads = (("get_actor_states", sim.get_actor_states(["robot"], arm.env_ids)), ("robot_root_states", arm.root()))
    for label, state in reads:
        if not torch.allclose(state[:, :7], spawn[:, :7], atol=POSE_ATOL):
            fails.append(f"{label} pose {_fmt(state[:, :7])} != written pose {_fmt(spawn[:, :7])}")
        if float(state[:, 7:].abs().max()) > POSE_ATOL:
            fails.append(f"{label} kept the written velocity {_fmt(state[:, 7:])}; a welded base must read zero")
    base_body = arm.body_pos(arm.base_idx)
    if not torch.allclose(base_body, spawn[:, :3], atol=POSE_ATOL):
        fails.append(f"rigid_body_pos_w[base_link] {_fmt(base_body)} != written position")
    print(f"  UNIFIED-WRITE base={_fmt(arm.root()[0, :3])} vel={_fmt(arm.root()[0, 7:])} ok={not fails}")
    return fails


def _check_dof_write(arm: _Arm) -> list[str]:
    torch, sim = arm.torch, arm.sim
    root_before, base_before = arm.root(), arm.body_pos(arm.base_idx)
    target = torch.tensor(DOF_TARGET, device=sim.sim_device).expand(arm.num_envs, arm.num_dof)
    arm.write_dof_pos(target)
    arm.run_steps(1, arm.zero_torques)
    q = arm.dof_pos()
    fails: list[str] = []
    if not torch.allclose(q, target, atol=DOF_ATOL):
        fails.append(f"dof_pos {_fmt(q)} != written {DOF_TARGET} (atol {DOF_ATOL})")
    fails += _base_unchanged(arm, root_before, base_before, "DOF write")
    print(f"  DOF-WRITE dof={_fmt(q[0])} ok={not fails}")
    return fails


def _check_prismatic(arm: _Arm) -> list[str]:
    torch, sim = arm.torch, arm.sim
    root_before, base_before = arm.root(), arm.body_pos(arm.base_idx)
    hold_q = torch.tensor(DOF_TARGET[:2], device=sim.sim_device)
    arm.run(SLIDE_SECONDS, lambda: arm.hold_torques(hold_q, SLIDE_FORCE))
    q_push = arm.dof_pos()[:, 2]
    arm.run(SLIDE_SECONDS, lambda: arm.hold_torques(hold_q, -SLIDE_FORCE))
    q_pull = arm.dof_pos()[:, 2]
    fails: list[str] = []
    if not bool(((q_push > SLIDE_MIN) & (q_push <= SLIDE_MAX)).all()):
        fails.append(f"+{SLIDE_FORCE} N left slide_joint at {_fmt(q_push)}, expected in ({SLIDE_MIN}, {SLIDE_MAX}]")
    if not bool(((q_push - q_pull) > SLIDE_MIN).all()):
        fails.append(f"-{SLIDE_FORCE} N did not retract slide_joint: {_fmt(q_push)} -> {_fmt(q_pull)}")
    fails += _base_unchanged(arm, root_before, base_before, "slider drive")
    print(f"  PRISMATIC push={_fmt(q_push)} pull={_fmt(q_pull)} ok={not fails}")
    return fails


def _check_per_env(arm: _Arm) -> list[str]:
    """Every env's base sits at its own origin (the backend's live layout) and the origins are distinct."""
    torch = arm.torch
    if arm.num_envs < 2:
        print("  PER-ENV n/a (single env)")
        return []
    base = arm.body_pos(arm.base_idx)  # [N, 3] from physics
    expected = arm.spawn[:, :3]  # init position + the backend's live env origins
    fails: list[str] = []
    if not torch.allclose(base, expected, atol=POSE_ATOL):
        fails.append(f"base positions {_fmt(base)} != origin + init position {_fmt(expected)}")
    xy = torch.round(base[:, :2] * 1000.0) / 1000.0
    if torch.unique(xy, dim=0).shape[0] != arm.num_envs:
        fails.append(f"base xy positions are not distinct across envs: {_fmt(xy)}")
    print(f"  PER-ENV base_xy={_fmt(xy)} ok={not fails}")
    return fails


_CHECKS: list[tuple[str, Callable[[_Arm], list[str]]]] = [
    ("SPAWN", _check_spawn),
    ("STAYS-PUT", _check_stays_put),
    ("SAME-POSE", _check_same_pose),
    ("TELEPORT", _check_teleport),
    ("MIXED", _check_mixed_write),
    ("UNIFIED-WRITE", _check_unified_write),
    ("DOF-WRITE", _check_dof_write),
    ("PRISMATIC", _check_prismatic),
    ("PER-ENV", _check_per_env),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-backend fixed-base robot assertion harness.")
    parser.add_argument("--simulator", required=True, choices=["mujoco", "mjwarp", "isaacgym", "isaacsim"])
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--result-file", default=None, help="write 'OK' here after PASS (teardown-robust)")
    args = parser.parse_args()

    if args.simulator == "mujoco" and args.num_envs > 1:
        print("SKIP: MuJoCo ClassicBackend is single-env (use mjwarp for --num-envs > 1)")
        return SKIP_EXIT_CODE

    sim, config = _build(args.simulator, args.num_envs)
    print(
        f"[{args.simulator}] robot={config.robot.asset.robot_type} fixed_base={config.robot.asset.fix_base_link} "
        f"dof={sim.dof_names} bodies={sim.body_names} num_envs={args.num_envs} sim_dt={sim.sim_dt}"
    )
    fails: list[str] = []
    if not config.robot.asset.fix_base_link:
        fails.append(f"robot '{ROBOT_KEY}' resolved to a floating-base config (did _robot_presets.register() run?)")
    else:
        arm = _Arm(sim, config)
        for name, check in _CHECKS:
            fails.extend(f"{name}: {msg}" for msg in check(arm))

    if args.result_file:
        with open(args.result_file, "w") as fh:
            fh.write("OK" if not fails else "FAIL\n" + "\n".join(fails))
    if fails:
        for msg in fails:
            print(f"FAIL: {msg}")
        return 1
    print(f"FIXED-BASE OK: {args.simulator} ({args.num_envs} env(s)) -- welded base, live joints")
    return 0


if __name__ == "__main__":
    run_and_hard_exit(main)
