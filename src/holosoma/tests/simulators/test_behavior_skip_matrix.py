"""Guards for the behavior-suite SCENARIOS routing table (CPU, no sim spin-up).

Pin the cross-backend routing for ``link_physics`` and freejoint damping without booting a backend:

- ``damping-decay`` is Isaac-only (driven by the PhysX config-time ``linear_damping`` path); the
  MuJoCo wrappers statically skip it.
- ``dr-damping-governs`` (the runtime damping DR term): MuJoCo and IsaacSim supported, IsaacGym
  self-skipped.
"""

from __future__ import annotations

import pytest

# behavior_assert and the MuJoCo wrapper module import torch/mujoco at module top, so this file
# needs the MuJoCo env even though it never boots a sim — mark it into the mujoco lane (the no_sim
# lane's env has no mujoco module, where a no_sim mark would silently skip it in every lane).
pytest.importorskip("torch")
pytest.importorskip("mujoco")

from tests.simulators import behavior_assert
from tests.simulators.mujoco import test_behavior_mujoco as mj_wrapper

pytestmark = pytest.mark.mujoco_classic


def test_damping_decay_is_isaac_only():
    # (scene_key, assertion_fn, min_envs, mujoco_supported, unsupported_backends).
    scene_key, _fn, min_envs, mujoco_ok, unsupported = behavior_assert.SCENARIOS["damping-decay"]
    assert scene_key == "damping-decay"
    assert min_envs == 1
    assert mujoco_ok is False, "damping-decay must be Isaac-only (no static MuJoCo freejoint-damping config)"
    assert unsupported == (), "damping-decay is NOT self-skipped on any Isaac backend (it runs there)"


def test_dr_damping_governs_unchanged():
    # Runtime damping DR term: MuJoCo and IsaacSim supported, IsaacGym self-skipped.
    scene_key, _fn, min_envs, mujoco_ok, unsupported = behavior_assert.SCENARIOS["dr-damping-governs"]
    assert scene_key == "dr-damping-pair"
    assert min_envs == 1
    assert mujoco_ok is True
    assert unsupported == ("isaacgym",)


def test_mujoco_wrappers_statically_skip_damping_decay():
    # The classic and warp MuJoCo wrappers carry ``damping-decay`` as a pytest.mark.skip param with
    # an Isaac-only reason, so it never spawns a subprocess.
    assert "damping-decay" in mj_wrapper._ISAAC_ONLY
    for params in (mj_wrapper._CLASSIC_PARAMS, mj_wrapper._WARP_PARAMS):
        marked = [p for p in params if isinstance(p, type(pytest.param("x"))) and "damping-decay" in p.values]
        assert marked, "damping-decay must be a pytest.param (carrying a skip mark), not a bare string"
        mark_names = [m.name for m in marked[0].marks]
        assert "skip" in mark_names
        reason = " ".join(str(m.kwargs.get("reason", "")) for m in marked[0].marks)
        assert "Isaac-only" in reason


def test_damping_decay_runs_on_isaac():
    # An Isaac backend absent from the unsupported tuple runs damping-decay. Checked here without
    # importing the Isaac-only wrapper modules (which importorskip on isaacgym/isaaclab, absent
    # off-cluster).
    _scene, _fn, _min, _mj, unsupported = behavior_assert.SCENARIOS["damping-decay"]
    assert "isaacgym" not in unsupported
    assert "isaacsim" not in unsupported
