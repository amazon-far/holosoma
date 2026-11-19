from __future__ import annotations

import dataclasses
import os
from typing import Any, cast

from loguru import logger

import holosoma.config_values.logger
from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.config_types.eval_callback import EvalCallbacksConfig
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_types.task import EvalTaskConfig
from holosoma.tasks.base import Task
from holosoma.utils.config_document import (
    ConfigDocumentError,
    alias_task_flags,
    expand_config_args,
    write_document,
)
from holosoma.utils.config_registry import parse_config
from holosoma.utils.config_utils import CONFIG_NAME, RUN_DOCUMENT_NAME
from holosoma.utils.eval_utils import (
    CheckpointConfig,
    init_eval_logging,
    load_checkpoint,
    load_saved_experiment_config,
    replace_retired_simulator,
)
from holosoma.utils.experiment_paths import get_experiment_dir, get_timestamp
from holosoma.utils.helpers import get_class


def run_eval_with_tyro(
    tyro_config: ExperimentConfig,
    checkpoint_cfg: CheckpointConfig,
    saved_config: ExperimentConfig,
    saved_wandb_path: str | None,
    eval_cbs_cfg: EvalCallbacksConfig | None = None,
    run_args: list[str] | None = None,
) -> None:
    with tyro_config.create_task() as env:
        _run_eval_session(
            env,
            tyro_config,
            checkpoint_cfg,
            saved_config=saved_config,
            saved_wandb_path=saved_wandb_path,
            eval_cbs_cfg=eval_cbs_cfg,
            run_args=run_args,
        )


def _run_eval_session(
    env: Any,
    tyro_config: ExperimentConfig,
    checkpoint_cfg: CheckpointConfig,
    *,
    saved_config: ExperimentConfig,
    saved_wandb_path: str | None,
    eval_cbs_cfg: EvalCallbacksConfig | None,
    run_args: list[str] | None = None,
) -> None:
    eval_log_dir = get_experiment_dir(tyro_config.logger, tyro_config.run, get_timestamp(), task_name="eval")
    eval_log_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving eval logs to {eval_log_dir}")
    tyro_config.save_config(str(eval_log_dir / CONFIG_NAME))
    if run_args:
        write_document(eval_log_dir / RUN_DOCUMENT_NAME, run_args)

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
        device=env.device,
        env=env,
        config=tyro_config.algo.config,
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

    task = cast("EvalTaskConfig", tyro_config.task)
    if task.export_onnx:
        exported_onnx_path = os.path.join(exported_policy_dir_path, exported_onnx_name)
        if not hasattr(algo, "export"):
            raise AttributeError(
                f"{algo_class.__name__} is missing an `export` method required for ONNX export during evaluation."
            )

        algo.export(onnx_file_path=exported_onnx_path)
        logger.info(f"Exported policy as onnx to: {exported_onnx_path}")

    algo.evaluate_policy(
        max_eval_steps=task.max_eval_steps,
    )


def _eval_config(saved: ExperimentConfig) -> ExperimentConfig:
    """``saved`` with evaluation's defaults: one viewed env, logger off, deterministic spawn, no time limit.

    The command line is parsed on top, so any of these can still be overridden.
    """
    spawn = dataclasses.replace(saved.terrain.terrain_term.spawn, randomize_tiles=False, xy_offset_range=0.0)
    return dataclasses.replace(
        saved,
        run=dataclasses.replace(saved.run, device=None, headless=False, num_envs=1),
        terrain=dataclasses.replace(
            saved.terrain, terrain_term=dataclasses.replace(saved.terrain.terrain_term, spawn=spawn)
        ),
        simulator=dataclasses.replace(
            saved.simulator,
            config=dataclasses.replace(
                saved.simulator.config,
                sim=dataclasses.replace(saved.simulator.config.sim, max_episode_length_s=100000.0),
            ),
        ),
        logger=holosoma.config_values.logger.disabled,
    )


_LEGACY_CALLBACK_PREFIXES = ("--recording.", "--push.", "--payload.")


def _alias_legacy_flags(args: list[str]) -> list[str]:
    """Rewrite ``--checkpoint`` and the eval-callback flags to their ``--task.*`` paths."""
    aliased = []
    for token in args:
        name, eq, value = token.partition("=")
        if name == "--checkpoint":
            aliased.append(f"--task.checkpoint{eq}{value}")
        elif token.startswith(_LEGACY_CALLBACK_PREFIXES):
            aliased.append(f"--task.callbacks.{token[2:]}")
        else:
            aliased.append(token)
    return aliased


def _flag_value(tokens: list[str], flag: str) -> str | None:
    """The last value given for ``flag`` in ``tokens``, as ``flag=value`` or ``flag value``."""
    value = None
    for index, token in enumerate(tokens):
        if token.startswith(f"{flag}="):
            value = token.partition("=")[2]
        elif token == flag and index + 1 < len(tokens):
            value = tokens[index + 1]
    return value


class EvalTask(Task):
    """Evaluate a checkpoint on its saved experiment, with the CLI's overrides applied on top."""

    name = "eval"

    def __init__(
        self,
        config: ExperimentConfig,
        saved_config: ExperimentConfig,
        saved_wandb_path: str | None,
        args: list[str] | None = None,
    ) -> None:
        super().__init__(config, args)
        self.saved_config = saved_config
        self.saved_wandb_path = saved_wandb_path

    @classmethod
    def from_args(cls, args: list[str]) -> EvalTask:
        init_eval_logging()
        args = cls.with_selection(_alias_legacy_flags(args))
        if "--print-args" in args or "--to-config" in args:
            parse_config(ExperimentConfig, args=args)  # prints the command line or document and exits
        # The checkpoint supplies the base config, so find it (and any simulator selection) first.
        try:
            tokens = alias_task_flags(expand_config_args(args))
        except ConfigDocumentError as exc:
            raise SystemExit(str(exc)) from exc
        checkpoint = _flag_value(tokens, "--task.checkpoint")
        if checkpoint is None:
            raise SystemExit("task:eval needs --eval.checkpoint=<path or wandb:// URI> (or --checkpoint=...).")
        saved_config, saved_wandb_path = load_saved_experiment_config(CheckpointConfig(checkpoint=checkpoint))
        eval_config = replace_retired_simulator(_eval_config(saved_config), tokens)
        record: list[str] = []
        config = parse_config(
            ExperimentConfig,
            default=eval_config,
            args=args,
            record=record,
            description="Overriding config on top of what's loaded.",
        )
        return cls(config, saved_config, saved_wandb_path, record)

    def run(self) -> None:
        task = cast("EvalTaskConfig", self.config.task)
        run_eval_with_tyro(
            self.config,
            CheckpointConfig(checkpoint=task.checkpoint),
            self.saved_config,
            self.saved_wandb_path,
            eval_cbs_cfg=task.callbacks,
            run_args=self.args,
        )
