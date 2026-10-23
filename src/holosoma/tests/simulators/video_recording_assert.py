"""Record actual nonblank frames and an MP4 from a headless Isaac simulation."""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

if sys.path and sys.path[0].endswith("tests/simulators"):
    sys.path.pop(0)

from tests.simulators._sim_harness import build_run_sim_config, run_and_hard_exit, step


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulator", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    from holosoma.utils.sim_utils import setup_simulation_environment

    config = build_run_sim_config(
        args.simulator,
        "empty",
        "g1-29dof",
        "terrain_locomotion_plane",
        record_dir=args.output_dir,
        show_command_overlay=False,
    )
    video = dataclasses.replace(config.logger.video, width=160, height=120, record_env_id=1, use_recording_thread=False)
    config = dataclasses.replace(
        config,
        device="cuda:0",
        training=dataclasses.replace(config.training, num_envs=2, headless=True),
        logger=dataclasses.replace(config.logger, video=video),
    )
    env, _, _ = setup_simulation_environment(config, device=config.device)
    sim = env.sim
    sim.set_headless(True)
    sim.setup()
    sim.setup_terrain()
    sim.load_assets()
    import torch

    sim.create_envs(2, sim.scene.env_origins, torch.zeros(13, device=config.device))
    sim.prepare_sim()
    recorder = sim.video_recorder
    assert recorder is not None
    recorder.start_recording(0)
    for _ in range(12):
        step(sim, 1)
        sim.sim.render()
        recorder.capture_frame(env_id=1)
    frames = list(recorder.video_frames)
    assert len(frames) >= 10, f"Only {len(frames)} frames captured"
    assert all(frame.shape == (120, 160, 3) for frame in frames)
    assert any(float(frame.max()) - float(frame.min()) > 10 for frame in frames), "Recording is blank"
    recorder.stop_recording()
    outputs = list(Path(args.output_dir).rglob("*.mp4"))
    assert outputs and all(path.stat().st_size > 1000 for path in outputs), "No encoded video"
    recorder.cleanup()
    print(f"PASS: {len(frames)} frames recorded from environment 1; encoded {outputs}")
    return 0


if __name__ == "__main__":
    run_and_hard_exit(main)
