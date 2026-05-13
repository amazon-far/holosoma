#!/usr/bin/env python3
"""Train a single-action transition predictor from simulator rollouts.

The current implementation collects transitions by rolling out a loaded policy checkpoint
inside the existing holosoma simulator stack, then trains a supervised MLP to predict the
state observed after the action finishes.
"""

from __future__ import annotations

import dataclasses
import sys
import traceback
from pathlib import Path

import tyro
from loguru import logger
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset, random_split

from holosoma.agents.state_predictor import TransitionNormalizer, TransitionPredictor
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_types.state_predictor import STATE_PREDICTOR_CONFIG_NAME, StatePredictorConfig
from holosoma.utils.eval_utils import CheckpointConfig, init_eval_logging, load_checkpoint, load_saved_experiment_config
from holosoma.utils.experiment_paths import get_experiment_dir, get_timestamp
from holosoma.utils.helpers import get_class
from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment
from holosoma.utils.tyro_utils import TYRO_CONIFG


def _resolve_device(config: ExperimentConfig, requested_device: str | None) -> str:
    if requested_device == "cpu" and hasattr(config.simulator.config, "mujoco_backend"):
        from holosoma.config_types.simulator import MujocoBackend  # noqa: PLC0415 -- deferred

        if config.simulator.config.mujoco_backend == MujocoBackend.WARP:
            logger.info("Auto-detected MuJoCo Warp backend - setting device to cuda:0")
            return "cuda:0"

    if requested_device is not None:
        return requested_device

    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _select_tensor(obs_dict: dict[str, Tensor], key: str) -> Tensor:
    if key not in obs_dict:
        raise KeyError(f"Observation key '{key}' is missing. Available keys: {list(obs_dict.keys())}")
    value = obs_dict[key]
    if not isinstance(value, Tensor):
        raise TypeError(f"Observation '{key}' must be a tensor, got {type(value)!r}")
    return value


def _target_state_from_step(
    obs_dict: dict[str, Tensor],
    extras: dict[str, object],
    target_key: str,
    done_mask: Tensor,
) -> Tensor:
    target_state = _select_tensor(obs_dict, target_key)
    final_obs = extras.get("final_observations")
    if isinstance(final_obs, dict) and target_key in final_obs:
        value = final_obs[target_key]
        if isinstance(value, Tensor):
            done_mask = done_mask.to(dtype=torch.bool).reshape(-1, 1)
            return torch.where(done_mask, value, target_state)
    return target_state


def _maybe_load_cached_transitions(cache_path: Path, expected_metadata: dict[str, object]) -> dict[str, object] | None:
    if not cache_path.exists():
        return None

    cached = torch.load(cache_path, map_location="cpu")
    if not isinstance(cached, dict):
        return None

    for key, expected_value in expected_metadata.items():
        if cached.get(key) != expected_value:
            return None

    if not all(isinstance(cached.get(key), Tensor) for key in ("states", "actions", "next_states")):
        return None

    logger.info(f"Loaded cached transitions from {cache_path}")
    return cached


def _save_transitions(cache_path: Path, payload: dict[str, object]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    logger.info(f"Saved transitions to {cache_path}")


def collect_transitions(
    source_config: ExperimentConfig,
    checkpoint_cfg: CheckpointConfig,
    predictor_config: StatePredictorConfig,
    experiment_dir: Path,
    device: str,
    saved_wandb_path: str | None,
    resolved_checkpoint_path: str | None = None,
) -> dict[str, Tensor]:
    if resolved_checkpoint_path is None:
        checkpoint_path = load_checkpoint(checkpoint_cfg.checkpoint, str(experiment_dir))
        resolved_checkpoint_path = str(checkpoint_path)

    rollout_cfg = dataclasses.replace(
        source_config,
        training=dataclasses.replace(
            source_config.training,
            headless=predictor_config.training.headless,
            num_envs=predictor_config.training.num_envs,
            seed=predictor_config.training.seed,
        ),
        logger=predictor_config.logger,
    )

    env, device, simulation_app = setup_simulation_environment(rollout_cfg, device=device)

    try:
        algo_class = get_class(rollout_cfg.algo._target_)
        algo = algo_class(
            device=device,
            env=env,
            config=rollout_cfg.algo.config,
            log_dir=str(experiment_dir),
            multi_gpu_cfg=None,
        )
        algo.setup()
        algo.attach_checkpoint_metadata(source_config, saved_wandb_path)
        algo.load(resolved_checkpoint_path)

        policy = algo.get_inference_policy(device)

        states: list[Tensor] = []
        actions: list[Tensor] = []
        next_states: list[Tensor] = []
        dones: list[Tensor] = []

        obs_dict = env.reset_all()
        current_state = _select_tensor(obs_dict, predictor_config.data.state_key)

        rollout_steps = predictor_config.data.rollout_steps
        logger.info(
            f"Collecting {rollout_steps} rollout steps across {env.num_envs} envs using state key '"
            f"{predictor_config.data.state_key}' and target key '{predictor_config.data.target_key}'"
        )

        for step_idx in range(rollout_steps):
            try:
                with torch.inference_mode():
                    action = policy({"actor_obs": _select_tensor(obs_dict, "actor_obs")})

                next_obs_dict, _, reset_buf, extras = env.step({"actions": action})
                target_state = _target_state_from_step(
                    next_obs_dict,
                    extras,
                    predictor_config.data.target_key,
                    reset_buf,
                )

                states.append(current_state.detach().cpu())
                actions.append(action.detach().cpu())
                next_states.append(target_state.detach().cpu())
                dones.append(reset_buf.detach().cpu())

                obs_dict = next_obs_dict
                current_state = _select_tensor(obs_dict, predictor_config.data.state_key)

                if (step_idx + 1) % max(1, rollout_steps // 10) == 0:
                    logger.info(f"Collected {step_idx + 1} / {rollout_steps} steps")
            except Exception as e:
                logger.error(f"Error at step {step_idx}: {e}")
                traceback.print_exc()
                raise

        logger.info(f"Collected {len(states)} states, {len(actions)} actions, {len(next_states)} next_states")
        logger.info(f"State shape sample: {states[0].shape if len(states) > 0 else 'empty'}")
        logger.info(f"Action shape sample: {actions[0].shape if len(actions) > 0 else 'empty'}")
        return {
            "states": torch.cat(states, dim=0),
            "actions": torch.cat(actions, dim=0),
            "next_states": torch.cat(next_states, dim=0),
            "dones": torch.cat(dones, dim=0),
            "source_checkpoint": resolved_checkpoint_path,
            "state_key": predictor_config.data.state_key,
            "target_key": predictor_config.data.target_key,
        }
    finally:
        logger.info("Skipping simulator shutdown in state predictor collection to avoid IsaacSim hang")


def _build_dataloaders(
    states: Tensor,
    actions: Tensor,
    next_states: Tensor,
    validation_ratio: float,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[DataLoader[tuple[Tensor, Tensor, Tensor]], DataLoader[tuple[Tensor, Tensor, Tensor]]]:
    dataset = TensorDataset(states, actions, next_states)
    total_size = len(dataset)
    validation_size = max(1, int(total_size * validation_ratio)) if total_size > 1 else 0
    training_size = total_size - validation_size

    generator = torch.Generator().manual_seed(seed)
    if validation_size == 0:
        training_dataset = dataset
        validation_dataset = dataset
    else:
        training_dataset, validation_dataset = random_split(dataset, [training_size, validation_size], generator=generator)

    train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, validation_loader


def _save_predictor_checkpoint(
    checkpoint_path: Path,
    model: TransitionPredictor,
    normalizer: TransitionNormalizer | None,
    optimizer,
    epoch: int,
    train_loss: float,
    validation_loss: float,
    metadata: dict[str, object],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "train_loss": train_loss,
        "validation_loss": validation_loss,
    }
    if normalizer is not None:
        payload["normalizer_state_dict"] = normalizer.state_dict()
    payload.update(metadata)
    torch.save(payload, checkpoint_path)


def _train_one_epoch(
    model: TransitionPredictor,
    normalizer: TransitionNormalizer | None,
    loader: DataLoader[tuple[Tensor, Tensor, Tensor]],
    optimizer,
    device: str,
    grad_clip: float,
) -> float:
    import torch.nn.functional as F

    model.train()
    if normalizer is not None:
        normalizer.eval()

    total_loss = 0.0
    num_batches = 0
    for states, actions, target_states in loader:
        states = states.to(device)
        actions = actions.to(device)
        target_states = target_states.to(device)

        optimizer.zero_grad(set_to_none=True)
        predicted_next_states = model(states, actions, normalizer=normalizer)
        loss = F.mse_loss(predicted_next_states, target_states)
        loss.backward()
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += float(loss.item())
        num_batches += 1

    return total_loss / max(1, num_batches)


def _evaluate(
    model: TransitionPredictor,
    normalizer: TransitionNormalizer | None,
    loader: DataLoader[tuple[Tensor, Tensor, Tensor]],
    device: str,
) -> float:
    import torch.nn.functional as F

    model.eval()
    if normalizer is not None:
        normalizer.eval()

    total_loss = 0.0
    num_batches = 0
    with torch.inference_mode():
        for states, actions, target_states in loader:
            states = states.to(device)
            actions = actions.to(device)
            target_states = target_states.to(device)
            predicted_next_states = model(states, actions, normalizer=normalizer)
            loss = F.mse_loss(predicted_next_states, target_states)
            total_loss += float(loss.item())
            num_batches += 1

    return total_loss / max(1, num_batches)


def train_state_predictor(
    predictor_config: StatePredictorConfig,
    rollout_config: ExperimentConfig,
    checkpoint_cfg: CheckpointConfig,
    saved_config: ExperimentConfig,
    saved_wandb_path: str | None,
) -> None:
    device = _resolve_device(rollout_config, predictor_config.device)
    experiment_dir = get_experiment_dir(
        predictor_config.logger,
        predictor_config.training,
        get_timestamp(),
        task_name="state_predictor",
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)

    logger.add(str(experiment_dir / "train.log"), level="DEBUG")
    predictor_config.save_config(str(experiment_dir / STATE_PREDICTOR_CONFIG_NAME))

    resolved_checkpoint_path = str(load_checkpoint(checkpoint_cfg.checkpoint, str(experiment_dir)))

    cache_path = predictor_config.get_cache_path(experiment_dir)
    expected_metadata = {
        "source_checkpoint": resolved_checkpoint_path,
        "state_key": predictor_config.data.state_key,
        "target_key": predictor_config.data.target_key,
    }

    cached_data: dict[str, object] | None = None
    if not predictor_config.data.force_collect:
        cached_data = _maybe_load_cached_transitions(cache_path, expected_metadata)

    if cached_data is None:
        logger.info("Starting transition collection...")
        cached_data = collect_transitions(
            source_config=rollout_config,
            checkpoint_cfg=checkpoint_cfg,
            predictor_config=predictor_config,
            experiment_dir=experiment_dir,
            device=device,
            saved_wandb_path=saved_wandb_path,
            resolved_checkpoint_path=resolved_checkpoint_path,
        )
        logger.info(f"Transition collection completed. Saving to {cache_path}")
        _save_transitions(cache_path, cached_data)
        logger.info("Transitions saved successfully")

    logger.info("Extracting states, actions, next_states from cached data")
    states = cached_data["states"]
    actions = cached_data["actions"]
    next_states = cached_data["next_states"]

    logger.info(f"Building dataloaders with batch_size={predictor_config.optim.batch_size}")
    train_loader, validation_loader = _build_dataloaders(
        states=states,
        actions=actions,
        next_states=next_states,
        validation_ratio=predictor_config.data.validation_ratio,
        batch_size=predictor_config.optim.batch_size,
        num_workers=predictor_config.optim.num_workers,
        seed=predictor_config.training.seed,
    )

    model = TransitionPredictor(
        state_dim=states.shape[-1],
        action_dim=actions.shape[-1],
        hidden_dims=predictor_config.model.hidden_dims,
        activation=predictor_config.model.activation,
        use_layer_norm=predictor_config.model.use_layer_norm,
        dropout_prob=predictor_config.model.dropout_prob,
        predict_residual=predictor_config.model.predict_residual,
        normalize_inputs=predictor_config.model.normalize_inputs,
    ).to(device)

    normalizer: TransitionNormalizer | None = None
    if predictor_config.model.normalize_inputs:
        normalizer = TransitionNormalizer.from_dataset(states.to(device), actions.to(device)).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=predictor_config.optim.learning_rate,
        weight_decay=predictor_config.optim.weight_decay,
    )

    start_epoch = 1
    best_validation_loss = float("inf")
    resume_checkpoint = predictor_config.training.checkpoint
    if resume_checkpoint is not None:
        resume_path = Path(resume_checkpoint)
        if resume_path.exists():
            resumed = torch.load(resume_path, map_location=device)
            model.load_state_dict(resumed["model_state_dict"])
            if normalizer is not None and resumed.get("normalizer_state_dict") is not None:
                normalizer.load_state_dict(resumed["normalizer_state_dict"])
            optimizer.load_state_dict(resumed["optimizer_state_dict"])
            start_epoch = int(resumed.get("epoch", 0)) + 1
            best_validation_loss = float(resumed.get("validation_loss", best_validation_loss))
            logger.info(f"Resumed state predictor training from {resume_path}")

    logger.info(
        f"Training transition predictor on {states.shape[0]} samples with state_dim={states.shape[-1]} and "
        f"action_dim={actions.shape[-1]}"
    )

    train_loss = 0.0
    validation_loss = 0.0
    checkpoint_metadata = {
        "source_checkpoint": cached_data["source_checkpoint"],
        "state_key": predictor_config.data.state_key,
        "target_key": predictor_config.data.target_key,
        "state_dim": int(states.shape[-1]),
        "action_dim": int(actions.shape[-1]),
        "predictor_config": predictor_config.to_serializable_dict(),
        "source_experiment_config": saved_config.to_serializable_dict(),
        "saved_wandb_path": saved_wandb_path,
    }

    for epoch in range(start_epoch, predictor_config.optim.epochs + 1):
        train_loss = _train_one_epoch(
            model=model,
            normalizer=normalizer,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip=predictor_config.optim.grad_clip,
        )
        validation_loss = _evaluate(
            model=model,
            normalizer=normalizer,
            loader=validation_loader,
            device=device,
        )

        logger.info(
            f"Epoch {epoch:04d} | train_loss={train_loss:.6f} | validation_loss={validation_loss:.6f}"
        )

        if epoch % predictor_config.optim.save_interval == 0:
            _save_predictor_checkpoint(
                checkpoint_path=experiment_dir / f"model_{epoch:07d}.pt",
                model=model,
                normalizer=normalizer,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                validation_loss=validation_loss,
                metadata=checkpoint_metadata,
            )

        if validation_loss <= best_validation_loss:
            best_validation_loss = validation_loss
            _save_predictor_checkpoint(
                checkpoint_path=experiment_dir / "best.pt",
                model=model,
                normalizer=normalizer,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                validation_loss=validation_loss,
                metadata=checkpoint_metadata,
            )

    _save_predictor_checkpoint(
        checkpoint_path=experiment_dir / f"model_{predictor_config.optim.epochs:07d}.pt",
        model=model,
        normalizer=normalizer,
        optimizer=optimizer,
        epoch=predictor_config.optim.epochs,
        train_loss=train_loss,
        validation_loss=validation_loss,
        metadata=checkpoint_metadata,
    )

    model.eval()
    if normalizer is not None:
        normalizer.eval()
    with torch.inference_mode():
        sample_states, sample_actions, sample_targets = next(iter(validation_loader))
        sample_states = sample_states.to(device)
        sample_actions = sample_actions.to(device)
        sample_targets = sample_targets.to(device)
        sample_predictions = model(sample_states, sample_actions, normalizer=normalizer)
        sample_mse = torch.mean((sample_predictions - sample_targets) ** 2).item()

    logger.info(f"Final validation sample MSE: {sample_mse:.6f}")
    logger.info(f"State predictor checkpoints saved to {experiment_dir}")


def main() -> None:
    init_eval_logging()

    checkpoint_cfg, remaining_args = tyro.cli(CheckpointConfig, return_unknown_args=True, add_help=False)
    predictor_cfg, remaining_args = tyro.cli(
        StatePredictorConfig,
        return_unknown_args=True,
        add_help=False,
        args=remaining_args,
    )
    saved_cfg, saved_wandb_path = load_saved_experiment_config(checkpoint_cfg)
    rollout_cfg = saved_cfg.get_eval_config()
    rollout_cfg = dataclasses.replace(
        rollout_cfg,
        training=dataclasses.replace(
            rollout_cfg.training,
            headless=predictor_cfg.training.headless,
            num_envs=predictor_cfg.training.num_envs,
            seed=predictor_cfg.training.seed,
        ),
        logger=predictor_cfg.logger,
    )
    overwritten_tyro_config = tyro.cli(
        ExperimentConfig,
        default=rollout_cfg,
        args=remaining_args,
        description="Overriding config on top of the checkpoint configuration.",
        config=TYRO_CONIFG,
    )

    try:
        train_state_predictor(
            predictor_config=predictor_cfg,
            rollout_config=overwritten_tyro_config,
            checkpoint_cfg=checkpoint_cfg,
            saved_config=saved_cfg,
            saved_wandb_path=saved_wandb_path,
        )
    except Exception as error:
        logger.error(f"Error during state predictor training: {error}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()