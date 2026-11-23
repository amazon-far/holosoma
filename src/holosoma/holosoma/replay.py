"""Play the motion a whole-body-tracking experiment tracks back on its robot, with the motion-playback plugin."""

from __future__ import annotations

from dataclasses import replace

from holosoma.config_types.command import MotionConfig
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_types.plugin import MotionPlaybackPluginConfig
from holosoma.config_types.run_sim import RunSimConfig
from holosoma.run_sim import run_simulation


def replay_config(experiment: ExperimentConfig) -> RunSimConfig:
    """A one-env simulation of ``experiment``'s world that plays its motion command's clip, then stops."""
    command = experiment.command
    motion_term = command.setup_terms.get("motion_command") if command is not None else None
    motion = motion_term.params.get("motion_config") if motion_term is not None else None
    if isinstance(motion, dict):
        motion = MotionConfig(**motion)
    if motion is None or not (motion.motion_file or motion.motion_dir):
        raise SystemExit("Replay needs an experiment whose motion command tracks a motion file, e.g. exp:g1-29dof-wbt.")
    files = [f"{motion.motion_dir.rstrip('/')}/*.npz"] if motion.motion_dir else [motion.motion_file]
    playback = MotionPlaybackPluginConfig(motion_files=files, objects=list(experiment.scene.rigid_objects)[:1])
    simulator = experiment.simulator
    sim = replace(simulator.config.sim, kinematic_playback=True)
    return RunSimConfig(
        simulator=replace(simulator, config=replace(simulator.config, sim=sim)),
        robot=experiment.robot,
        terrain=experiment.terrain,
        scene=experiment.scene,
        sensor=experiment.sensor,
        plugin={**experiment.plugin, "motion_playback": playback},
        training=replace(experiment.training, num_envs=1),
    )


def replay(experiment: ExperimentConfig) -> None:
    run_simulation(replay_config(experiment))


def main() -> None:
    from holosoma.config_values.experiment import get_annotated_experiment_config
    from holosoma.utils.config_registry import parse_config

    # Pass the factory uncalled so parse_config builds it after plugins load.
    tyro_cfg = parse_config(get_annotated_experiment_config)
    replay(tyro_cfg)


if __name__ == "__main__":
    main()
