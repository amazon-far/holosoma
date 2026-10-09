from __future__ import annotations

import os
from typing import Any

import tyro
from loguru import logger

from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.config_types.eval_callback import EvalCallbacksConfig
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.utils.config_registry import parse_config
from holosoma.utils.config_utils import CONFIG_NAME
from holosoma.utils.eval_utils import (
    CheckpointConfig,
    init_eval_logging,
    load_checkpoint,
    load_saved_experiment_config,
)
from holosoma.utils.experiment_paths import get_experiment_dir, get_timestamp
from holosoma.utils.helpers import get_class
from holosoma.utils.sim_utils import (
    graceful_simulation_signals,
    setup_simulation_environment,
    simulation_resource_session,
)


@graceful_simulation_signals
def run_eval_with_tyro(
    tyro_config: ExperimentConfig,
    checkpoint_cfg: CheckpointConfig,
    saved_config: ExperimentConfig,
    saved_wandb_path: str | None,
    eval_cbs_cfg: EvalCallbacksConfig | None = None,
) -> None:
    env, device, simulation_app = setup_simulation_environment(tyro_config)
    with simulation_resource_session(env, simulation_app):
        _run_eval_session(
            env,
            device,
            tyro_config,
            checkpoint_cfg,
            saved_config=saved_config,
            saved_wandb_path=saved_wandb_path,
            eval_cbs_cfg=eval_cbs_cfg,
        )


def _run_eval_session(
    env: Any,
    device: str,
    tyro_config: ExperimentConfig,
    checkpoint_cfg: CheckpointConfig,
    *,
    saved_config: ExperimentConfig,
    saved_wandb_path: str | None,
    eval_cbs_cfg: EvalCallbacksConfig | None,
) -> None:
    eval_log_dir = get_experiment_dir(tyro_config.logger, tyro_config.training, get_timestamp(), task_name="eval")
    eval_log_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving eval logs to {eval_log_dir}")
    tyro_config.save_config(str(eval_log_dir / CONFIG_NAME))

    # Inject eval callbacks into algo config
    if eval_cbs_cfg is not None:
        cb_configs = eval_cbs_cfg.collect_active_callbacks()
        if cb_configs:
            object.__setattr__(tyro_config.algo.config, "eval_callbacks", cb_configs)

    assert checkpoint_cfg.checkpoint is not None
    checkpoint = load_checkpoint(checkpoint_cfg.checkpoint, str(eval_log_dir))
    checkpoint_path = str(checkpoint)

    algo_class = get_class(tyro_config.algo._target_)
    algo: BaseAlgo = algo_class(
        device=device,
        env=env,
        config=tyro_config.algo.config,
        log_dir=str(eval_log_dir),
        multi_gpu_cfg=None,
    )
    algo.setup()
    algo.attach_checkpoint_metadata(saved_config, saved_wandb_path)
    algo.load(checkpoint_path)

    checkpoint_dir = os.path.dirname(checkpoint_path)

    exported_policy_dir_path = os.path.join(checkpoint_dir, "exported")
    os.makedirs(exported_policy_dir_path, exist_ok=True)
    exported_policy_name = checkpoint_path.split("/")[-1]  # example: model_5000.pt
    exported_onnx_name = exported_policy_name.replace(".pt", ".onnx")  # example: model_5000.onnx

    if tyro_config.training.export_onnx:
        exported_onnx_path = os.path.join(exported_policy_dir_path, exported_onnx_name)
        if not hasattr(algo, "export"):
            raise AttributeError(
                f"{algo_class.__name__} is missing an `export` method required for ONNX export during evaluation."
            )

        algo.export(onnx_file_path=exported_onnx_path)
        logger.info(f"Exported policy as onnx to: {exported_onnx_path}")

    algo.evaluate_policy(
        max_eval_steps=tyro_config.training.max_eval_steps,
    )


def main() -> None:
    init_eval_logging()
    checkpoint_cfg, remaining_args = tyro.cli(CheckpointConfig, return_unknown_args=True, add_help=False)
    eval_cbs_cfg, remaining_args = tyro.cli(
        EvalCallbacksConfig, return_unknown_args=True, add_help=False, args=remaining_args
    )
    saved_cfg, saved_wandb_path = load_saved_experiment_config(checkpoint_cfg)
    eval_cfg = saved_cfg.get_eval_config()
    overwritten_tyro_config = parse_config(
        ExperimentConfig,
        default=eval_cfg,
        args=remaining_args,
        description="Overriding config on top of what's loaded.",
    )

    run_eval_with_tyro(overwritten_tyro_config, checkpoint_cfg, saved_cfg, saved_wandb_path, eval_cbs_cfg=eval_cbs_cfg)


if __name__ == "__main__":
    main()
