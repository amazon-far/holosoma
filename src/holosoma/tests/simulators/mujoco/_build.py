"""Shared MuJoCo sim-builder helpers for the live simulator tests.

Every live MuJoCo test stands up a real sim — bridge and virtual gantry disabled — through
the same ``setup -> setup_terrain -> load_assets -> create_envs -> prepare_sim`` sequence,
differing only in backend (Classic CPU single-env vs Warp GPU multi-env), the scene, and a
few knobs. That sequence lives here once.

Import this only AFTER the module-level ``pytest.importorskip("torch")`` / ``"mujoco"``
guards in the consuming test (this module imports torch at load), mirroring how
``dr_matrix_assert`` defers ``_dr_matrix``.
"""

from __future__ import annotations

import dataclasses
import types

import pytest
import torch

import holosoma.config_values.robot as robot_values
import holosoma.config_values.run_sim as run_sim_values
from holosoma.config_types.experiment import TrainingConfig
from holosoma.config_types.run_sim import RunSimConfig
from holosoma.config_types.scene import SceneConfig
from holosoma.config_types.simulator import BridgeConfig, MujocoBackend, VirtualGantryCfg


def _bridge_gantry_off(preset):
    """Return ``preset`` with bridge and virtual gantry disabled (every live test wants this)."""
    return dataclasses.replace(
        preset,
        config=dataclasses.replace(
            preset.config,
            bridge=BridgeConfig(enabled=False),
            virtual_gantry=VirtualGantryCfg(enabled=False),
        ),
    )


def _finish_build(env, num_envs, env_origins, base_init):
    """Run the shared headless ``setup -> ... -> prepare_sim`` tail and return the sim."""
    sim = env.sim
    sim.set_headless(True)
    sim.setup()
    sim.setup_terrain()
    sim.load_assets()
    sim.create_envs(num_envs, env_origins, base_init)
    sim.prepare_sim()
    return sim


def _robot_base_init():
    """The g1 robot's flattened initial root state ``[pos, rot, lin_vel, ang_vel]`` tensor."""
    init = robot_values.g1_29dof.init_state
    return torch.tensor(list(init.pos) + list(init.rot) + list(init.lin_vel) + list(init.ang_vel))


def build_classic_sim(scene: SceneConfig | None = None):
    """Build a ClassicBackend MuJoCo sim (CPU, single env) for ``scene`` (default: empty)."""
    from holosoma.utils.sim_utils import setup_simulation_environment

    sim_cfg = _bridge_gantry_off(run_sim_values.mujoco)
    config = dataclasses.replace(
        RunSimConfig(simulator=sim_cfg, robot=robot_values.g1_29dof, scene=scene or SceneConfig()),
        device="cpu",
    )
    env, device, _ = setup_simulation_environment(config, device="cpu")
    return _finish_build(env, 1, torch.zeros(1, 3, device=device), _robot_base_init())


def build_warp_sim(
    scene: SceneConfig | None = None,
    *,
    num_envs: int = 4,
    env_origins=None,
    seed: int = 42,
):
    """Build a WarpBackend MuJoCo sim (CUDA, multi-env) for ``scene`` (default: empty).

    Skips (not fails) when the Warp stack is unavailable or the ``mjwarp`` preset is not the
    Warp backend — but a genuine build error propagates as a real failure rather than being
    swallowed into a green skip. ``env_origins`` defaults to zeros; pass distinct origins to
    observe per-env spread.
    """
    pytest.importorskip("warp")
    pytest.importorskip("mujoco_warp")
    if getattr(run_sim_values.mjwarp.config, "mujoco_backend", None) != MujocoBackend.WARP:
        pytest.skip("mjwarp config is not the Warp backend")
    from holosoma.utils.sim_utils import setup_simulation_environment

    sim_cfg = _bridge_gantry_off(run_sim_values.mjwarp)
    config = dataclasses.replace(
        RunSimConfig(
            simulator=sim_cfg,
            robot=robot_values.g1_29dof,
            scene=scene or SceneConfig(),
            training=TrainingConfig(num_envs=num_envs, headless=True, seed=seed, torch_deterministic=False),
        ),
        device="cuda:0",
    )
    env, device, _ = setup_simulation_environment(config, device="cuda:0")
    if env_origins is None:
        env_origins = torch.zeros(num_envs, 3, device=device)
    return _finish_build(env, num_envs, env_origins, {})


def env_shell(sim, num_envs, **extra):
    """A minimal env stand-in (``simulator``/``num_envs``/``device`` + any ``extra`` attrs).

    Randomization terms only read these attributes off the env, so a SimpleNamespace stands in
    for a full task. ``extra`` carries term-specific fields, e.g. ``robot_config=...``.
    """
    return types.SimpleNamespace(simulator=sim, num_envs=num_envs, device=sim.sim_device, **extra)


def object_body_id(sim, name):
    """MuJoCo body id of a registered rigid object's root body (by ObjectRegistry name)."""
    import mujoco

    root_body = sim.scene_manager.rigid_object_root_bodies[name]
    return mujoco.mj_name2id(sim.backend.model, mujoco.mjtObj.mjOBJ_BODY, root_body)
