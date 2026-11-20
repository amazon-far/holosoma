from __future__ import annotations

import dataclasses
import os
from typing import Any

import tyro
from loguru import logger

import holosoma.config_values.logger
from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.config_types.eval_callback import EvalCallbacksConfig
from holosoma.config_types.evaluation import EvalConfig
from holosoma.config_types.experiment import EvaluationExperimentConfig, TrainingExperimentConfig
from holosoma.utils.config_registry import parse_config
from holosoma.utils.config_utils import CONFIG_NAME
from holosoma.utils.eval_utils import (
    CheckpointConfig,
    init_eval_logging,
    load_checkpoint,
    load_saved_experiment_config,
    replace_retired_simulator,
)
from holosoma.utils.experiment_paths import get_experiment_dir, get_timestamp
from holosoma.utils.helpers import get_class


def evaluation_config(saved: TrainingExperimentConfig, eval_config: EvalConfig) -> EvaluationExperimentConfig:
    """``saved``'s environment and MDP with evaluation's defaults.

    One env in the viewer, logger off, a deterministic spawn and no episode time limit; the saved
    seed and run naming are kept. The command line is parsed on top, so any of these can be overridden.
    """
    terrain = saved.env.terrain
    spawn = dataclasses.replace(terrain.terrain_term.spawn, randomize_tiles=False, xy_offset_range=0.0)
    simulator = saved.env.simulator
    sim = dataclasses.replace(simulator.config.sim, max_episode_length_s=100000.0)
    env = dataclasses.replace(
        saved.env,
        terrain=dataclasses.replace(terrain, terrain_term=dataclasses.replace(terrain.terrain_term, spawn=spawn)),
        simulator=dataclasses.replace(simulator, config=dataclasses.replace(simulator.config, sim=sim)),
    )
    run = dataclasses.replace(
        saved.run, device=None, num_envs=1, headless=False, logger=holosoma.config_values.logger.disabled
    )
    task = f"{saved.task.rpartition('.')[0]}.evaluate"
    return EvaluationExperimentConfig(task=task, env=env, mdp=saved.mdp, eval=eval_config, run=run)


def run_eval_with_tyro(
    tyro_config: EvaluationExperimentConfig,
    checkpoint_cfg: CheckpointConfig,
    saved_config: TrainingExperimentConfig,
    saved_wandb_path: str | None,
    eval_cbs_cfg: EvalCallbacksConfig | None = None,
) -> None:
    with tyro_config.create_task() as env:
        _run_eval_session(
            env,
            tyro_config,
            checkpoint_cfg,
            saved_config=saved_config,
            saved_wandb_path=saved_wandb_path,
            eval_cbs_cfg=eval_cbs_cfg,
        )


def _run_eval_session(
    env: Any,
    tyro_config: EvaluationExperimentConfig,
    checkpoint_cfg: CheckpointConfig,
    *,
    saved_config: TrainingExperimentConfig,
    saved_wandb_path: str | None,
    eval_cbs_cfg: EvalCallbacksConfig | None,
) -> None:
    run = tyro_config.run
    eval_log_dir = get_experiment_dir(run.logger, run.project, run.name, get_timestamp(), task_name="eval")
    eval_log_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving eval logs to {eval_log_dir}")
    tyro_config.save_config(str(eval_log_dir / CONFIG_NAME))

    # Inject eval callbacks into algo config
    if eval_cbs_cfg is not None:
        cb_configs = eval_cbs_cfg.collect_active_callbacks()
        if cb_configs:
            object.__setattr__(saved_config.algo.config, "eval_callbacks", cb_configs)

    assert checkpoint_cfg.checkpoint is not None
    checkpoint = load_checkpoint(checkpoint_cfg.checkpoint, str(eval_log_dir))
    checkpoint_path = str(checkpoint)

    algo_class = get_class(saved_config.algo._target_)
    algo: BaseAlgo = algo_class(
        device=env.device,
        env=env,
        config=saved_config.algo.config,
        log_dir=str(eval_log_dir),
        multi_gpu_cfg=None,
    )
    # Size the policy like the saved one (e.g. its teacher slots) before setup() allocates it.
    restore_structure = getattr(algo, "restore_structure_from_checkpoint", None)
    if restore_structure is not None:
        import torch

        checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint_data, dict):
            raise TypeError("Evaluation checkpoint must contain a dictionary.")
        restore_structure(checkpoint_data)
    algo.setup()
    algo.attach_checkpoint_metadata(saved_config, saved_wandb_path)
    algo.load(checkpoint_path)

    checkpoint_dir = os.path.dirname(checkpoint_path)

    exported_policy_dir_path = os.path.join(checkpoint_dir, "exported")
    os.makedirs(exported_policy_dir_path, exist_ok=True)
    exported_policy_name = checkpoint_path.split("/")[-1]  # example: model_5000.pt
    exported_onnx_name = exported_policy_name.replace(".pt", ".onnx")  # example: model_5000.onnx

    if tyro_config.eval.export_onnx:
        exported_onnx_path = os.path.join(exported_policy_dir_path, exported_onnx_name)
        if not hasattr(algo, "export"):
            raise AttributeError(
                f"{algo_class.__name__} is missing an `export` method required for ONNX export during evaluation."
            )

        algo.export(onnx_file_path=exported_onnx_path)
        logger.info(f"Exported policy as onnx to: {exported_onnx_path}")

    algo.evaluate_policy(
        max_eval_steps=tyro_config.eval.max_eval_steps,
    )


def main() -> None:
    init_eval_logging()
    checkpoint_cfg, remaining_args = tyro.cli(CheckpointConfig, return_unknown_args=True, add_help=False)
    eval_cbs_cfg, remaining_args = tyro.cli(
        EvalCallbacksConfig, return_unknown_args=True, add_help=False, args=remaining_args
    )
    saved_cfg, saved_wandb_path = load_saved_experiment_config(checkpoint_cfg)
    eval_part = EvalConfig(checkpoint=checkpoint_cfg.checkpoint, callbacks=eval_cbs_cfg)
    eval_cfg = replace_retired_simulator(evaluation_config(saved_cfg, eval_part), remaining_args)
    overwritten_tyro_config = parse_config(
        EvaluationExperimentConfig,
        default=eval_cfg,
        args=remaining_args,
        description="Overriding config on top of what's loaded.",
    )

    run_eval_with_tyro(overwritten_tyro_config, checkpoint_cfg, saved_cfg, saved_wandb_path, eval_cbs_cfg=eval_cbs_cfg)


if __name__ == "__main__":
    main()
