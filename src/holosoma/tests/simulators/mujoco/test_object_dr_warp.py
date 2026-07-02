"""GPU (WarpBackend, multi-env) tests for cross-backend object physics DR.

These prove the object DR terms (``randomize_object_rigid_body_mass_startup`` /
``..._material_startup``) mutate the live per-env MuJoCo Warp model fields (``body_mass`` /
``geom_friction``) for a registered free body, per env and per object. Skipped without
CUDA / the MuJoCo-Warp stack.

The terms rely on ``@mujoco_required_field`` field expansion, which in production is driven
by ``prepare_manager_fields`` scanning the RandomizationManager. The bare test sim has no
manager, so we expand the fields manually via ``prepare_fields`` (the same primitive the
manager path delegates to) before calling the term — exactly what a real run does.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")

# MuJoCo WarpBackend (CUDA) only.
pytestmark = pytest.mark.mujoco_warp

if not torch.cuda.is_available():
    pytest.skip("WarpBackend multi-env tests require a CUDA device", allow_module_level=True)

from holosoma.config_types.scene import RigidObjectConfig, SceneConfig  # noqa: E402
from holosoma.managers.randomization.terms.objects import (  # noqa: E402
    _mujoco_object_geom_ids,
    randomize_object_rigid_body_mass_startup,
    randomize_object_rigid_body_material_startup,
)
from tests.simulators._dr_matrix import _sampler  # noqa: E402
from tests.simulators.mujoco._build import build_warp_sim, env_shell, object_body_id  # noqa: E402

SMALL_BOX = "holosoma/data/scene_objects/boxes/small_box.urdf"
NUM_ENVS = 4


def test_object_mass_randomized_warp():
    sim = build_warp_sim(
        SceneConfig(rigid_objects={"box": RigidObjectConfig(urdf_file=SMALL_BOX, position=[0.4, 0.0, 0.6])}),
        seed=7,
    )
    # Expand the field the term needs (manager path does this via prepare_manager_fields).
    sim.prepare_randomization_fields(["body_mass"])

    bid = object_body_id(sim, "box")
    before = sim.backend.warp_model_bridge.body_mass[:, bid].clone()

    env = env_shell(sim, NUM_ENVS)
    randomize_object_rigid_body_mass_startup(
        env, torch.arange(NUM_ENVS, device=sim.sim_device), sampler=_sampler(env), mass_distribution_params=(5.0, 6.0)
    )

    after = sim.backend.warp_model_bridge.body_mass[:, bid]
    # Mass changed in every env (a CPU-only / no-op write would leave it at the URDF 0.1 kg).
    assert torch.all((after - before).abs() > 1e-3), f"mass unchanged: before={before.cpu()} after={after.cpu()}"
    # Additive offset within the configured band (per env).
    assert torch.all(after - before >= 5.0 - 1e-3) and torch.all(after - before <= 6.0 + 1e-3)
    # Per-env distinct (guards against a broadcast-one-value bug).
    assert len({round(float(v), 4) for v in after.cpu().tolist()}) > 1


def test_object_friction_randomized_warp():
    sim = build_warp_sim(
        SceneConfig(rigid_objects={"box": RigidObjectConfig(urdf_file=SMALL_BOX, position=[0.4, 0.0, 0.6])}),
        seed=7,
    )
    sim.prepare_randomization_fields(["geom_friction"])

    geom_ids = _mujoco_object_geom_ids(sim, "box")
    assert geom_ids, "object 'box' resolved to no geoms (unnamed-geom resolution failure)"
    g0 = geom_ids[0]
    before = sim.backend.warp_model_bridge.geom_friction[:, g0, 0].clone()

    env = env_shell(sim, NUM_ENVS)
    randomize_object_rigid_body_material_startup(
        env,
        torch.arange(NUM_ENVS, device=sim.sim_device),
        sampler=_sampler(env),
        material={"mujoco": {"sliding_friction": (0.2, 0.3)}},
    )

    after = sim.backend.warp_model_bridge.geom_friction[:, g0, 0]
    # operation="abs": every env now sits inside the requested band.
    assert torch.all(after >= 0.2 - 1e-4) and torch.all(after <= 0.3 + 1e-4), f"friction out of band: {after.cpu()}"
    # And it actually changed from the default (1.0) and is per-env distinct.
    assert torch.all((after - before).abs() > 1e-3)
    assert len({round(float(v), 4) for v in after.cpu().tolist()}) > 1


def test_object_mass_dr_multi_object_isolation_warp():
    """Randomizing one object's mass must not touch the other object's mass."""
    sim = build_warp_sim(
        SceneConfig(
            rigid_objects={
                "box_a": RigidObjectConfig(urdf_file=SMALL_BOX, position=[0.4, 0.0, 0.6]),
                "box_b": RigidObjectConfig(urdf_file=SMALL_BOX, position=[0.4, 0.3, 0.6]),
            }
        ),
        seed=7,
    )
    sim.prepare_randomization_fields(["body_mass"])

    bid_a, bid_b = object_body_id(sim, "box_a"), object_body_id(sim, "box_b")
    before_a = sim.backend.warp_model_bridge.body_mass[:, bid_a].clone()
    before_b = sim.backend.warp_model_bridge.body_mass[:, bid_b].clone()

    env = env_shell(sim, NUM_ENVS)
    randomize_object_rigid_body_mass_startup(
        env,
        torch.arange(NUM_ENVS, device=sim.sim_device),
        sampler=_sampler(env),
        mass_distribution_params=(5.0, 6.0),
        object_names=["box_a"],
    )

    after_a = sim.backend.warp_model_bridge.body_mass[:, bid_a]
    after_b = sim.backend.warp_model_bridge.body_mass[:, bid_b]
    assert torch.all((after_a - before_a).abs() > 1e-3), "box_a mass should have changed"
    assert torch.allclose(after_b, before_b), "box_b mass leaked — per-object targeting broken"


def test_object_dr_robot_only_scene_noop_warp():
    """A robot-only scene resolves to no free bodies -> the terms no-op without error."""
    sim = build_warp_sim(SceneConfig(), seed=7)
    sim.prepare_randomization_fields(["body_mass", "geom_friction"])
    env = env_shell(sim, NUM_ENVS)
    # Should not raise (RandomizerNotSupportedError or otherwise) and should change nothing.
    randomize_object_rigid_body_mass_startup(
        env, torch.arange(NUM_ENVS, device=sim.sim_device), sampler=_sampler(env), mass_distribution_params=(5.0, 6.0)
    )
    randomize_object_rigid_body_material_startup(
        env,
        torch.arange(NUM_ENVS, device=sim.sim_device),
        sampler=_sampler(env),
        material={"mujoco": {"sliding_friction": (0.2, 0.3)}},
    )
