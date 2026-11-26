"""Exercise fixed-base contacts with reordered and omitted observation bodies."""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path
from typing import Any

if sys.path and sys.path[0].endswith("simulators"):
    sys.path.pop(0)
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from holosoma.hooks import Phase  # noqa: E402
from tests.simulators import _robot_presets  # noqa: E402
from tests.simulators._sim_harness import build_run_sim_config, launch_simulator, run_and_hard_exit  # noqa: E402
from tests.simulators.robot_name_mapping_assert import mapped_robot  # noqa: E402


def _build() -> Any:
    import torch

    from holosoma.config_types.scene import PhysicsConfig, RigidObjectConfig, SceneConfig
    from holosoma.config_types.sensor import ContactSensorConfig, ContactTargetConfig

    _robot_presets.register()
    actor = ContactTargetConfig(target_kind="actor", target="helper_probe")
    helper = ContactTargetConfig(target_kind="robot_link", target="link1")
    sensors = {
        "helper": ContactSensorConfig(sources=[helper], counterparts=[actor]),
        "helper_reaction": ContactSensorConfig(sources=[actor], counterparts=[helper]),
        "whole_robot_reaction": ContactSensorConfig(sources=[actor], counterparts=[ContactTargetConfig()]),
        "unfiltered": ContactSensorConfig(
            sources=[helper, ContactTargetConfig(), ContactTargetConfig(target_kind="robot_link", target="base_link")]
        ),
        "unfiltered_slow": ContactSensorConfig(sources=[helper], update_decimation=2),
    }
    config = build_run_sim_config("isaacsim", "empty", "test-arm-fixed", "terrain_locomotion_plane", sensors=sensors)
    box = RigidObjectConfig(
        urdf_file="holosoma/data/scene_objects/boxes/small_box.urdf",
        usd_file="holosoma/data/scene_objects/boxes/small_box.usda",
        physics=PhysicsConfig(mass=2),
    )
    config = dataclasses.replace(
        config,
        robot=mapped_robot(),
        scene=SceneConfig(
            rigid_objects={
                "base_probe": dataclasses.replace(box, position=[2, 0, 1]),
                "helper_probe": dataclasses.replace(box, position=[3, 0, 1]),
            }
        ),
        device="cuda:0",
        training=dataclasses.replace(config.training, num_envs=2, headless=True),
        simulator=dataclasses.replace(
            config.simulator, config=dataclasses.replace(config.simulator.config, debug_viz=False)
        ),
    )
    sim, device, _ = launch_simulator(config, device=config.device)
    sim.set_headless(True)
    sim.setup()
    sim.setup_terrain()
    sim.load_assets()
    sim.create_envs(
        2, torch.zeros(2, 3, device=device), torch.tensor(config.robot.init_state.root_state_list(), device=device)
    )
    sim.prepare_sim()
    return sim


def _assert_mapping(sim: Any) -> None:
    import torch

    from holosoma.utils.rotations import quat_apply

    assert sim.robot_config.asset.fix_base_link
    assert sim.body_names == ["slider", "base_link", "link2"]
    assert "link1" not in sim.body_names
    ids = torch.arange(sim.num_envs, device=sim.sim_device)
    # These names and geometry offsets come from the asset, independent of the canonical table.
    for actor, body, offset in (
        ("base_probe", "base_link", [-0.095, 0, 0.05]),
        ("helper_probe", "link1", [0.15, 0, 0.07]),
    ):
        body_id = sim._robot.body_names.index(body)
        position = sim._robot.data.body_link_pos_w[:, body_id]
        quaternion = sim._robot.data.body_link_quat_w[:, body_id]
        states = sim.get_actor_states([actor], ids).clone()
        states[:, :3] = position + quat_apply(
            quaternion, torch.tensor(offset, device=sim.sim_device).repeat(sim.num_envs, 1), w_last=False
        )
        states[:, 3:7] = quaternion[:, [1, 2, 3, 0]]
        states[:, 7:] = 0
        sim.set_actor_states([actor], ids, states)

    sim.hooks.emit(Phase.FRAME_BEGIN)
    sim.simulate_at_each_physics_step()
    native = sim.contact_sensor.data.net_forces_w
    expected = torch.stack(
        [native[:, sim.contact_sensor.body_names.index(name)] for name in ("slider", "base_link", "link2")], dim=1
    )
    # A nonzero base row makes reordered or omitted body mappings observable.
    assert torch.all(torch.linalg.vector_norm(expected[:, 1], dim=-1) > 1), expected
    torch.testing.assert_close(sim.contact_forces[:], expected)
    assert set(sim.sensor_manager.contact_names) == set(sim.sensor_config)
    torch.testing.assert_close(sim.contact_recorder.recorded_forces[:, 0], expected)
    helper_id = sim.contact_sensor.body_names.index("link1")
    unfiltered_expected = torch.stack([native[:, helper_id], native.sum(dim=1), expected[:, 1]], dim=1)
    unfiltered = sim.get_contact_data("unfiltered")
    torch.testing.assert_close(unfiltered, unfiltered_expected)
    assert torch.all((unfiltered[:, 1] - expected.sum(dim=1)).norm(dim=-1) > 1)
    assert sim.get_contact_data("unfiltered_slow").count_nonzero() == 0
    # Only the filtered helper body and its actor require additional views.
    assert len(sim.contact_views) == 2

    helper = sim.get_contact_data("helper")
    assert torch.all(torch.linalg.vector_norm(helper[:, 0], dim=-1) > 1), helper
    torch.testing.assert_close(sim.get_contact_data("helper_reaction"), -helper, atol=0.1, rtol=0.02)
    torch.testing.assert_close(
        sim.get_contact_data("whole_robot_reaction"), sim.get_contact_data("helper_reaction"), atol=0.1, rtol=0.02
    )
    sim.hooks.emit(Phase.FRAME_END)

    # Episode reset clears owned measurements. Rendering preserves legacy forces without a solver step.
    untouched = unfiltered[1].clone()
    legacy = sim.contact_forces[:].clone()
    sim.hooks.emit(Phase.EPISODE_START, ids[:1])
    sim.render_sensors(ids[:1], 1)
    torch.testing.assert_close(sim.contact_forces[:], legacy)
    assert sim.get_contact_data("unfiltered")[0].count_nonzero() == 0
    torch.testing.assert_close(sim.get_contact_data("unfiltered")[1], untouched)
    torch.testing.assert_close(sim.get_contact_data("unfiltered")[1], unfiltered_expected[1])
    torch.testing.assert_close(unfiltered, unfiltered_expected)
    sim.simulate_at_each_physics_step()
    native = sim.contact_sensor.data.net_forces_w
    torch.testing.assert_close(sim.get_contact_data("unfiltered_slow")[:, 0], native[:, helper_id])
    torch.testing.assert_close(sim.get_contact_data("unfiltered")[:, 1], native.sum(dim=1))
    torch.testing.assert_close(unfiltered, unfiltered_expected)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-file", type=Path, required=True)
    args = parser.parse_args()
    sim = _build()
    try:
        _assert_mapping(sim)
        args.result_file.write_text("OK\n")
        return 0
    finally:
        sim.hooks.emit(Phase.CLOSE)


if __name__ == "__main__":
    run_and_hard_exit(main)
