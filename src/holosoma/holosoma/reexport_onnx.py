"""Re-export ONNX model with dynamic batch size.

The original ONNX was exported with a fixed batch dimension (e.g. 8192)
baked in from the training num_envs.  This script loads the .pt checkpoint,
reconstructs the environment just enough to call ``algo.export()``, and
writes a new ONNX file with ``dynamic_axes`` so inference can run at
batch=1.

Usage:
    source scripts/source_mujoco_setup.sh
    python src/holosoma/holosoma/reexport_onnx.py \
        --checkpoint /path/to/model_106000.pt \
        --motion_dir ./g1_npz \
        --split_file ./split/phuma_val.txt \
        --simulator mjwarp
"""

from __future__ import annotations

import argparse
import dataclasses
import os
from pathlib import Path

from loguru import logger

from holosoma.utils.eval_utils import (
    CheckpointConfig,
    init_eval_logging,
    load_saved_experiment_config,
)
from holosoma.utils.helpers import get_class
from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment


def main():
    init_eval_logging()

    parser = argparse.ArgumentParser(description="Re-export ONNX with dynamic batch size")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--motion_dir", default=None, help="Override motion_dir")
    parser.add_argument("--split_file", default=None, help="Override split_file")
    parser.add_argument("--num_envs", type=int, default=4,
                        help="Number of envs (only needed for env setup, keep small)")
    parser.add_argument("--simulator", default=None, choices=["mujoco", "mjwarp"],
                        help="Override simulator")
    parser.add_argument("--output", default=None,
                        help="Output ONNX path (default: same dir as checkpoint, with _dynamic suffix)")
    args = parser.parse_args()

    # Load saved config from checkpoint
    checkpoint_cfg = CheckpointConfig(checkpoint=args.checkpoint)
    saved_cfg, saved_wandb_path = load_saved_experiment_config(checkpoint_cfg)

    # Use minimal num_envs — we only need the env to reconstruct the model
    saved_cfg = dataclasses.replace(
        saved_cfg,
        eval_overrides=dataclasses.replace(
            saved_cfg.eval_overrides,
            num_envs=args.num_envs,
            headless=True,
        ),
    )

    # Override simulator if requested
    if args.simulator is not None:
        from holosoma.config_values import simulator as sim_presets
        sim_cfg = sim_presets.DEFAULTS[args.simulator]
        saved_cfg = dataclasses.replace(saved_cfg, simulator=sim_cfg)
        saved_cfg = dataclasses.replace(
            saved_cfg,
            randomization=dataclasses.replace(
                saved_cfg.randomization,
                ignore_unsupported=True,
            ),
        )

    eval_cfg = saved_cfg.get_eval_config()

    # Override motion_dir and split_file
    motion_config = eval_cfg.command.setup_terms["motion_command"].params["motion_config"]
    if isinstance(motion_config, dict):
        if args.motion_dir is not None:
            motion_config["motion_dir"] = args.motion_dir
        if args.split_file is not None:
            motion_config["split_file"] = args.split_file
    else:
        from dataclasses import asdict
        mc_dict = asdict(motion_config)
        if args.motion_dir is not None:
            mc_dict["motion_dir"] = args.motion_dir
        if args.split_file is not None:
            mc_dict["split_file"] = args.split_file
        eval_cfg.command.setup_terms["motion_command"].params["motion_config"] = mc_dict

    # Setup environment
    env, device, simulation_app = setup_simulation_environment(eval_cfg)

    # Create algo and load checkpoint
    algo_class = get_class(eval_cfg.algo._target_)
    checkpoint_dir = os.path.dirname(args.checkpoint)
    algo = algo_class(
        device=device,
        env=env,
        config=eval_cfg.algo.config,
        log_dir=checkpoint_dir,
        multi_gpu_cfg=None,
    )
    algo.setup()
    algo.attach_checkpoint_metadata(saved_cfg, saved_wandb_path)
    algo.load(str(args.checkpoint))
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    # Determine output path
    if args.output:
        onnx_path = args.output
    else:
        pt_name = Path(args.checkpoint).stem  # e.g. model_106000
        onnx_path = os.path.join(checkpoint_dir, "exported", f"{pt_name}.onnx")

    # Export
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    algo.export(onnx_file_path=onnx_path)
    logger.info(f"Exported ONNX to: {onnx_path}")

    # Verify the new model has dynamic batch
    try:
        import onnx
        m = onnx.load(onnx_path)
        for inp in m.graph.input:
            shape = []
            for d in inp.type.tensor_type.shape.dim:
                shape.append(d.dim_param if d.dim_param else d.dim_value)
            logger.info(f"  {inp.name}: {shape}")
    except ImportError:
        pass

    if simulation_app:
        close_simulation_app(simulation_app)


if __name__ == "__main__":
    main()
