"""Headless cross-backend domain-randomization (DR) assertion harness.

Builds ONE real, fully-managed locomotion env for a chosen backend (real action_manager +
randomization_manager — NO shims), then runs the full matrix of robot + object DR terms against
it and asserts each took effect, reading the mutated physics-model values back through that
backend's native API. Exits 0 on success, non-zero with a message on failure — so it works as a
live integration test under each backend's launcher (MuJoCo venv, IsaacSim
DISPLAY/EULA), the same way scene_spawn_assert.py does for spawning.

Usage:
  python dr_matrix_assert.py --simulator mujoco                 # ClassicBackend, 1 env, cpu
  python dr_matrix_assert.py --simulator mjwarp   --num-envs 4  # WarpBackend, cuda
  python dr_matrix_assert.py --simulator isaacsim --num-envs 4  # IsaacSim, cuda
  python dr_matrix_assert.py --simulator isaacsim --num-envs 4  # IsaacSim, cuda

The DR term calls are shared across backends (in _dr_matrix.py); only the read-back of the
mutated model field differs, so each backend supplies a small reader object below.
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from tests.simulators._dr_matrix import ModelReader

# This file lives in tests/simulators/, which contains an ``isaacsim/`` subpackage. Run as a
# script, that dir lands on sys.path[0] and shadows the real IsaacSim ``isaacsim`` package. Drop
# it (mirrors scene_spawn_assert.py); the package is imported via PYTHONPATH=src/holosoma instead.
# Only when run as a script — when imported (e.g. by the in-process classic test) sys.path[0] is
# the caller's, which must not be mutated.
if __name__ == "__main__" and sys.path and sys.path[0].endswith("tests/simulators"):
    sys.path.pop(0)


# ---------------------------------------------------------------------------------------------
# Backend-specific physics-model read-back. Each exposes the ModelReader interface used by
# _dr_matrix (body_mass / geom_friction_for_body / body_inertia / body_com), reading env 0.
# ---------------------------------------------------------------------------------------------
class _MujocoReader:
    """MuJoCo classic (float64 numpy on backend.model) or Warp (per-world warp_model_bridge)."""

    # MuJoCo writes CoM straight into ``body_ipos`` and reads it back from the same field, so the
    # base-CoM band assertion reads the integrated value and is trusted.
    COM_READBACK_TRUSTED = True
    # MuJoCo material DR sets sliding friction only; restitution is not modelled the same way, so
    # the restitution band assertion is gated off for this backend (see run_object_dr).
    RESTITUTION_READBACK_TRUSTED = False

    def __init__(self, sim):
        import mujoco

        self._mj = mujoco
        self._sim = sim
        self._m = sim.backend.model  # CPU model for name->id resolution (both backends)
        self._warp = getattr(sim.backend, "warp_model_bridge", None)

    def _bid(self, body_name: str) -> int:
        # Robot body (possibly "robot_"-prefixed), or a registry object name whose MuJoCo body is
        # resolved via the object helper (e.g. "free0" -> "free0_baseLink").
        for cand in (body_name, "robot_" + body_name):
            bid = self._mj.mj_name2id(self._m, self._mj.mjtObj.mjOBJ_BODY, cand)
            if bid != -1:
                return bid
        from holosoma.managers.randomization.terms.objects import _mujoco_object_body_names

        obj_bodies = _mujoco_object_body_names(self._sim, body_name)
        assert obj_bodies, f"body '{body_name}' not found in model"
        return self._mj.mj_name2id(self._m, self._mj.mjtObj.mjOBJ_BODY, obj_bodies[0])

    def _geom0(self, body_name: str) -> int:
        bid = self._bid(body_name)
        geoms = [g for g in range(self._m.ngeom) if self._m.geom_bodyid[g] == bid]
        assert geoms, f"body '{body_name}' has no geoms"
        return geoms[0]

    def body_mass(self, body_name: str) -> float:
        bid = self._bid(body_name)
        return float(self._warp.body_mass[0, bid]) if self._warp is not None else float(self._m.body_mass[bid])

    def geom_friction_for_body(self, body_name: str) -> float:
        g = self._geom0(body_name)
        return (
            float(self._warp.geom_friction[0, g, 0]) if self._warp is not None else float(self._m.geom_friction[g, 0])
        )

    def body_inertia(self, body_name: str):
        bid = self._bid(body_name)
        src = self._warp.body_inertia[0, bid] if self._warp is not None else self._m.body_inertia[bid]
        return tuple(float(x) for x in src)

    def body_com(self, body_name: str):
        bid = self._bid(body_name)
        src = self._warp.body_ipos[0, bid] if self._warp is not None else self._m.body_ipos[bid]
        return tuple(float(x) for x in src)


class _IsaacSimReader:
    """IsaacSim physx-view read-back via the isaaclab assets (env 0)."""

    # PhysX get_coms reflects the CoM write, so the base-CoM band assertion is trusted.
    COM_READBACK_TRUSTED = True
    # IsaacSim is the only backend that models restitution (PhysX material slot 2), so the
    # restitution band assertion is enabled here (see run_object_dr).
    RESTITUTION_READBACK_TRUSTED = True

    def __init__(self, sim):
        self._sim = sim
        self._scene = sim.scene

    def _asset_and_body(self, body_name: str):
        """Return (asset, body_index). Object names (from the registry) resolve to the matching
        RigidObject (single-body, index 0); otherwise the name is a robot body on the 'robot'
        articulation. Objects are checked FIRST because Articulation.find_bodies RAISES (not
        returns empty) on a non-matching name."""
        for name in self._scene.rigid_objects:
            if body_name == name or body_name.startswith(name):
                return self._scene[name], 0
        robot = self._scene["robot"]
        ids, _ = robot.find_bodies([body_name])
        assert ids, f"could not resolve body '{body_name}' on the robot or any object"
        return robot, ids[0]

    @staticmethod
    def _row(view, bidx: int):
        """Per-body row for env 0. The robot Articulation view is [num_envs, num_bodies, k]; the
        single-body RigidObject view is [num_envs, k] (no body axis) — handle both."""
        return view[0, bidx] if view.ndim == 3 else view[0]

    def body_mass(self, body_name: str) -> float:
        asset, bidx = self._asset_and_body(body_name)
        masses = asset.root_physx_view.get_masses()  # [num_envs, num_bodies] or [num_envs]
        return float(masses[0, bidx] if masses.ndim == 2 else masses[0])

    def geom_friction_for_body(self, body_name: str) -> float:
        asset, _bidx = self._asset_and_body(body_name)
        # [num_envs, num_shapes, 3] = (static, dynamic, restitution)
        mats = asset.root_physx_view.get_material_properties()
        return float(mats[0, 0, 0])

    def dynamic_friction_for_body(self, body_name: str) -> float:
        asset, _bidx = self._asset_and_body(body_name)
        # [num_envs, num_shapes, 3] = (static, dynamic, restitution) -> dynamic friction at slot 1.
        mats = asset.root_physx_view.get_material_properties()
        return float(mats[0, 0, 1])

    def restitution_for_body(self, body_name: str) -> float:
        asset, _bidx = self._asset_and_body(body_name)
        # [num_envs, num_shapes, 3] = (static, dynamic, restitution) -> restitution at slot 2.
        mats = asset.root_physx_view.get_material_properties()
        return float(mats[0, 0, 2])

    def body_inertia(self, body_name: str):
        asset, bidx = self._asset_and_body(body_name)
        row = self._row(asset.root_physx_view.get_inertias(), bidx)  # 9 = row-major 3x3
        return (float(row[0]), float(row[4]), float(row[8]))  # Ixx, Iyy, Izz

    def body_com(self, body_name: str):
        asset, bidx = self._asset_and_body(body_name)
        row = self._row(asset.root_physx_view.get_coms(), bidx)  # [..,7] (pos+quat) or [..,3]
        return (float(row[0]), float(row[1]), float(row[2]))


_READERS = {
    "mujoco": _MujocoReader,
    "mjwarp": _MujocoReader,
    "isaacsim": _IsaacSimReader,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-backend DR matrix assertion harness.")
    parser.add_argument("--simulator", required=True, choices=list(_READERS))
    parser.add_argument("--num-envs", type=int, default=None, help="default: 1 for mujoco, 4 otherwise")
    parser.add_argument("--result-file", default=None, help="write 'OK' here after all checks pass")
    args = parser.parse_args()

    from holosoma.config_values import simulator as sim_values

    sim_cfg = getattr(sim_values, args.simulator)
    num_envs = args.num_envs if args.num_envs is not None else (1 if args.simulator == "mujoco" else 4)
    device = "cpu" if args.simulator == "mujoco" else "cuda:0"

    # Imported here (not at top): for isaacsim, build_full_env's training_context must
    # initialize the launcher first; _dr_matrix imports torch, which must come after that init.
    import traceback

    from tests.simulators import _dr_matrix as dr

    with dr.build_full_env(sim_cfg, num_envs=num_envs, device=device) as env:
        if args.simulator == "mjwarp":
            # Object inertia field isn't in the loco preset; expand every field the matrix touches.
            env.simulator.prepare_randomization_fields(["body_mass", "geom_friction", "body_ipos", "body_inertia"])
        reader = cast("ModelReader", _READERS[args.simulator](env.simulator))
        # Run each stage inside an explicit try so a failure is LOGGED with a traceback before the
        # with-block exits — IsaacSim's close_simulation_app (on __exit__) can hard-terminate the
        # process and swallow a propagating exception, leaving exit 0 and no diagnostic otherwise.
        try:
            skipped = dr.run_robot_dr(env, reader) or []
            dr.run_push_dr(env)
            dr.run_object_dr(env, reader)
            skipped += dr.run_distribution_dr(env, reader) or []
        except BaseException:
            print("DR MATRIX FAILED:\n" + traceback.format_exc(), flush=True)
            return 1
        # Emit the success sentinel HERE, INSIDE the with-block — anything printed after teardown
        # is unreliable on IsaacSim. The result-file is the authoritative success signal; the test
        # checks it, not the exit code.
        #
        # Scope the banner honestly: only claim "all verified" when nothing was skipped. Any check
        # the backend cannot read back live (e.g. a base CoM) is listed as NOT-verified so
        # the banner never claims a field it did not actually integrate-check.
        if skipped:
            msg = (
                f"DR MATRIX OK: {args.simulator} ({num_envs} env(s)) — robot + push + object DR "
                f"verified, EXCEPT not-verified on this backend: {', '.join(skipped)}"
            )
        else:
            msg = f"DR MATRIX OK: {args.simulator} ({num_envs} env(s)) — robot + push + object DR all verified"
        print(msg, flush=True)
        if args.result_file:
            with open(args.result_file, "w") as f:
                f.write("OK\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
