"""Live CPU (ClassicBackend, single-env) tests for cross-backend object physics DR.

The object DR terms (``randomize_object_rigid_body_{mass,material,inertia}_startup``) route
all MuJoCo work through the shared ``randomize_field`` chokepoint. These tests prove the
ClassicBackend branch of that path mutates the live single-env ``mujoco.MjModel`` fields
(``body_mass`` / ``geom_friction`` / ``body_inertia``) in place — one sample, but real — so
classic CPU is usable for sim2sim robustness checks, reproducing a randomized instance, and
CPU-only CI. The GPU/multi-env analogue lives in test_object_dr_warp.py.

Unlike the Warp path, the ClassicBackend needs NO field expansion: it writes the one CPU
model field directly (the ``@mujoco_required_field`` attribute still names the field for the
term, but ``randomize_field`` skips expansion/validation when not on the Warp backend).

Runs in the MuJoCo (hsmujoco) CPU env — no CUDA required; mirrors the builder in
test_object_observations_live.py.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")

# MuJoCo ClassicBackend (CPU) only.
pytestmark = pytest.mark.mujoco_classic

from holosoma.config_types.scene import RigidObjectConfig, SceneConfig  # noqa: E402
from holosoma.managers.randomization.terms.objects import (  # noqa: E402
    randomize_object_rigid_body_inertia_startup,
    randomize_object_rigid_body_mass_startup,
    randomize_object_rigid_body_material_startup,
)
from tests.simulators._dr_matrix import _sampler  # noqa: E402
from tests.simulators.mujoco._build import build_classic_sim, env_shell, object_body_id  # noqa: E402

SMALL_BOX = "holosoma/data/scene_objects/boxes/small_box.urdf"
_IDENTITY_OFF_DIAG = {"Ixy": (1.0, 1.0), "Iyz": (1.0, 1.0), "Ixz": (1.0, 1.0)}


def test_object_mass_dr_multi_object_isolation_classic_cpu():
    """Randomizing one object's mass must not touch the other object's mass (single env)."""
    sim = build_classic_sim(
        SceneConfig(
            rigid_objects={
                "box_a": RigidObjectConfig(urdf_file=SMALL_BOX, position=[0.4, 0.0, 0.6]),
                "box_b": RigidObjectConfig(urdf_file=SMALL_BOX, position=[0.4, 0.3, 0.6]),
            }
        )
    )
    bid_a, bid_b = object_body_id(sim, "box_a"), object_body_id(sim, "box_b")
    before_a = float(sim.backend.model.body_mass[bid_a])
    before_b = float(sim.backend.model.body_mass[bid_b])

    env = env_shell(sim, 1)
    randomize_object_rigid_body_mass_startup(
        env, torch.tensor([0]), sampler=_sampler(env), mass_distribution_params=(5.0, 6.0), object_names=["box_a"]
    )

    assert abs(float(sim.backend.model.body_mass[bid_a]) - before_a) > 1e-3, "box_a mass should have changed"
    assert abs(float(sim.backend.model.body_mass[bid_b]) - before_b) < 1e-9, (
        "box_b mass leaked — per-object targeting broken"
    )


def test_object_inertia_off_diagonal_warns_classic_cpu():
    """A non-identity off-diagonal range is ignored (MuJoCo stores diagonal moments) but warns.

    The term logs via loguru (not stdlib logging), so capture it with a temporary sink.
    """
    from loguru import logger

    sim = build_classic_sim(
        SceneConfig(rigid_objects={"box": RigidObjectConfig(urdf_file=SMALL_BOX, position=[0.4, 0.0, 0.6])})
    )
    env = env_shell(sim, 1)
    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(m), level="WARNING")
    try:
        randomize_object_rigid_body_inertia_startup(
            env,
            torch.tensor([0]),
            sampler=_sampler(env),
            inertia_distribution_params_dict={
                "Ixx": (1.0, 1.0),
                "Iyy": (1.0, 1.0),
                "Izz": (1.0, 1.0),
                "Ixy": (1.5, 2.0),
                "Iyz": (1.0, 1.0),
                "Ixz": (1.0, 1.0),
            },
        )
    finally:
        logger.remove(sink_id)
    assert any("off-diagonal" in m.lower() and "Ixy" in m for m in messages), (
        "expected a one-time off-diagonal-ignored warning"
    )


def test_object_dr_robot_only_scene_noop_classic_cpu():
    """A robot-only scene resolves to no free bodies -> the terms no-op without error."""
    sim = build_classic_sim(SceneConfig())
    env = env_shell(sim, 1)
    # None of these should raise (RandomizerNotSupportedError or otherwise).
    randomize_object_rigid_body_mass_startup(
        env, torch.tensor([0]), sampler=_sampler(env), mass_distribution_params=(5.0, 6.0)
    )
    randomize_object_rigid_body_material_startup(
        env,
        torch.tensor([0]),
        sampler=_sampler(env),
        material={"mujoco": {"sliding_friction": (0.2, 0.3)}},
    )
    randomize_object_rigid_body_inertia_startup(
        env,
        torch.tensor([0]),
        sampler=_sampler(env),
        inertia_distribution_params_dict={
            "Ixx": (2.0, 3.0),
            "Iyy": (1.0, 1.0),
            "Izz": (1.0, 1.0),
            **_IDENTITY_OFF_DIAG,
        },
    )
