#!/usr/bin/env python3
"""Validate multi-step prediction accuracy of the trained state predictor.

This script tests how well the predictor works in closed-loop scenarios
where each predicted state feeds into the next prediction.
"""

from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import yaml
from loguru import logger

from src.holosoma.holosoma.agents.state_predictor import TransitionPredictor, TransitionNormalizer


def load_model_and_config(model_dir: Path, states_shape, actions_shape, device: str = "cuda:0"):
    """Load trained model, normalizer, and config."""
    # Load config
    config_path = model_dir / "state_predictor_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load model state dict
    checkpoint = torch.load(model_dir / "best.pt", map_location=device)

    # Infer state_dim and action_dim from data shape
    state_dim = states_shape[-1]
    action_dim = actions_shape[-1]

    # Reconstruct model
    hidden_dims = config["model"]["hidden_dims"]
    model = TransitionPredictor(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
        activation=config["model"]["activation"],
        use_layer_norm=config["model"]["use_layer_norm"],
        dropout_prob=config["model"]["dropout_prob"],
        predict_residual=config["model"]["predict_residual"],
        normalize_inputs=config["model"]["normalize_inputs"],
    )
    model = model.to(device)

    # Load model state
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            # Try to load directly
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    # Load normalizer if it exists
    normalizer = None
    if isinstance(checkpoint, dict) and "normalizer_state_dict" in checkpoint:
        normalizer_state = checkpoint["normalizer_state_dict"]
        normalizer = TransitionNormalizer(
            state_mean=normalizer_state["state_mean"],
            state_std=normalizer_state["state_std"],
            action_mean=normalizer_state["action_mean"],
            action_std=normalizer_state["action_std"],
        )
        normalizer = normalizer.to(device)

    model.eval()
    if normalizer is not None:
        normalizer.eval()

    return model, normalizer, config


def load_transitions_data(model_dir: Path, device: str = "cuda:0") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load cached transition data."""
    data_path = model_dir / "state_predictor_transitions.pt"
    data = torch.load(data_path, map_location=device)

    states = data["states"]
    actions = data["actions"]
    next_states = data["next_states"]

    logger.info(f"Loaded data: states={states.shape}, actions={actions.shape}, next_states={next_states.shape}")

    return states, actions, next_states


def evaluate_multistep_prediction(
    model: TransitionPredictor,
    normalizer: TransitionNormalizer | None,
    states: torch.Tensor,
    actions: torch.Tensor,
    next_states: torch.Tensor,
    device: str,
    num_steps: int = 5,
    num_samples: int = 100,
) -> dict:
    """Evaluate multi-step prediction accuracy.

    For each starting state, predict forward `num_steps` times using
    consecutive actions from the rollout, comparing predicted vs actual states.
    """
    results = {
        "num_steps": num_steps,
        "single_step_mse": [],
        "cumulative_mse": [],
        "trajectory_errors": [],
    }

    # Randomly select starting indices
    max_start_idx = len(states) - num_steps
    if max_start_idx <= 0:
        logger.warning(f"Not enough data for {num_steps}-step prediction (need {num_steps} transitions)")
        return results

    sample_indices = torch.randperm(max_start_idx)[: min(num_samples, max_start_idx)]

    with torch.inference_mode():
        for start_idx in sample_indices:
            # Collect true trajectory
            true_trajectory = [states[start_idx].unsqueeze(0)]
            for i in range(num_steps):
                true_trajectory.append(next_states[start_idx + i].unsqueeze(0))

            true_trajectory = torch.cat(true_trajectory, dim=0).to(device)  # (num_steps+1, state_dim)
            actions_seq = actions[start_idx : start_idx + num_steps].to(device)  # (num_steps, action_dim)

            # Predict trajectory
            pred_trajectory = [states[start_idx].unsqueeze(0).to(device)]
            for i in range(num_steps):
                current_state = pred_trajectory[-1]
                action = actions_seq[i].unsqueeze(0)
                next_state = model(current_state, action, normalizer=normalizer)
                pred_trajectory.append(next_state)

            pred_trajectory = torch.cat(pred_trajectory, dim=0)  # (num_steps+1, state_dim)

            # Compute errors
            mse_per_step = F.mse_loss(pred_trajectory, true_trajectory, reduction="none").mean(dim=1)
            results["single_step_mse"].append(mse_per_step[1:].cpu())  # Skip first (which is input)
            results["cumulative_mse"].append(
                F.mse_loss(pred_trajectory[1:], true_trajectory[1:], reduction="mean").item()
            )
            results["trajectory_errors"].append(
                torch.norm(pred_trajectory - true_trajectory, p=2, dim=1).cpu()
            )

    # Aggregate results
    results["single_step_mse"] = torch.cat(results["single_step_mse"], dim=0).mean().item()
    results["cumulative_mse"] = torch.tensor(results["cumulative_mse"]).mean().item()
    results["trajectory_errors"] = torch.cat(results["trajectory_errors"], dim=0)

    return results


def main():
    model_dir = Path("/home/ubuntu22/holosoma/logs/StatePrediction/20260430_184651-single_action-state_predictor")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    logger.info("Loading transition data")
    states, actions, next_states = load_transitions_data(model_dir, device=device)

    logger.info(f"Loading model from {model_dir}")
    model, normalizer, config = load_model_and_config(model_dir, states.shape, actions.shape, device=device)

    # Test single-step (baseline)
    logger.info("\n=== Single-Step Prediction (Baseline) ===")
    dataset = TensorDataset(states, actions, next_states)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    total_mse = 0.0
    num_batches = 0
    with torch.inference_mode():
        for batch_states, batch_actions, batch_next in loader:
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)
            batch_next = batch_next.to(device)

            pred = model(batch_states, batch_actions, normalizer=normalizer)
            mse = F.mse_loss(pred, batch_next).item()
            total_mse += mse
            num_batches += 1

    single_step_mse = total_mse / num_batches
    logger.info(f"Single-step Validation MSE: {single_step_mse:.6f}")

    # Test multi-step predictions
    logger.info("\n=== Multi-Step Prediction ===")
    for num_steps in [5, 10]:
        logger.info(f"\nTesting {num_steps}-step predictions...")
        results = evaluate_multistep_prediction(
            model, normalizer, states, actions, next_states, device, num_steps=num_steps, num_samples=200
        )

        logger.info(f"{num_steps}-Step Results:")
        logger.info(f"  Average MSE per step: {results['single_step_mse']:.6f}")
        logger.info(f"  Cumulative trajectory MSE: {results['cumulative_mse']:.6f}")
        logger.info(f"  Mean L2 trajectory error: {results['trajectory_errors'].mean():.6f}")
        logger.info(f"  Max L2 trajectory error:  {results['trajectory_errors'].max():.6f}")
        logger.info(f"  Std L2 trajectory error:  {results['trajectory_errors'].std():.6f}")

        # Error growth analysis
        cumulative_vs_single = results["cumulative_mse"] / single_step_mse
        logger.info(f"  Cumulative error / Single-step error ratio: {cumulative_vs_single:.2f}x")

    logger.info("\n=== Summary ===")
    logger.info(f"Single-step MSE: 0.{int(single_step_mse * 1e6):06d} (baseline)")
    logger.info(
        "If 5-step cumulative MSE is significantly larger, "
        "the model drifts under closed-loop prediction."
    )
    logger.info("If 5-step error < 0.2 and 10-step error < 0.5, predictor is usable.")


if __name__ == "__main__":
    main()
