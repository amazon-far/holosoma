import argparse
import os
import glob
import torch
from torch.utils.data import DataLoader, TensorDataset
from holosoma.agents.cvae.cvae_model import ConditionedVAE, compute_cvae_loss
from tqdm import tqdm

from holosoma.config_types.experiment import ExperimentConfig
from holosoma.utils.eval_utils import CheckpointConfig, load_saved_experiment_config
from holosoma.utils.sim_utils import setup_simulation_environment, close_simulation_app
from holosoma.utils.helpers import get_class


def get_latest_checkpoint(checkpoint_dir: str) -> str:
    """Find the latest .pt file in a directory, or return the path if it's already a .pt file."""
    if os.path.isfile(checkpoint_dir) and checkpoint_dir.endswith('.pt'):
        return checkpoint_dir
        
    pts = glob.glob(os.path.join(checkpoint_dir, "model_*.pt"))
    if not pts:
        pts = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
    if not pts:
        raise ValueError(f"No .pt files found in {checkpoint_dir}")
    
    # Sort files by modification time
    pts.sort(key=os.path.getmtime)
    return pts[-1]


def collect_data(sac_checkpoint_dir: str, num_episodes: int = 100) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Simulates episodes using the pre-trained SAC agent to collect states and actions.
    
    Args:
        sac_checkpoint_dir: Path to the pretrained model logs/checkpoints.
        num_episodes: Number of steps/episodes to rollout (mapped to steps here for simplicity).
        
    Returns:
        states: Tensor of shape (N, state_dim).
        actions: Tensor of shape (N, action_dim).
    """
    checkpoint_path = get_latest_checkpoint(sac_checkpoint_dir)
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint_cfg = CheckpointConfig(checkpoint=checkpoint_path)
    saved_cfg, _ = load_saved_experiment_config(checkpoint_cfg)
    
    # Optional performance optimization: Reduce num_envs if you want exactly num_episodes
    # For now, we take eval config directly
    eval_cfg = saved_cfg.get_eval_config()
    
    # Force headless mode computationally to prevent IsaacSim window from opening
    import dataclasses
    if hasattr(eval_cfg, "simulator"):
        # Simulator config might also be frozen, try recursive replacement if needed
        # Or safely ignore if rendering doesn't strictly depend on this field
        try:
            if hasattr(eval_cfg.simulator, "sim") and hasattr(eval_cfg.simulator.sim, "render_mode"):
                object.__setattr__(eval_cfg.simulator.sim, "render_mode", "headless")
            if hasattr(eval_cfg.simulator, "debug_viz"):
                object.__setattr__(eval_cfg.simulator, "debug_viz", False)
        except Exception:
            pass
            
    if hasattr(eval_cfg, "training"):
        # `training` is a frozen dataclass, so we must mutate it properly
        # We can either replace it using dataclasses.replace or use object set-attribute hack if it's frozen
        # Since it's deeply nested, the safest hack for frozen dataclass patching at runtime is object.__setattr__
        object.__setattr__(eval_cfg.training, "headless", True)
        
    # 1. Setup Simulation Env
    env, device, simulation_app = setup_simulation_environment(eval_cfg)
    print(f"Environment instantiated successfully on {device}.")
    
    try:
        # 2. Setup Algorithm (SAC / Evaluator)
        algo_class = get_class(eval_cfg.algo._target_)
        algo = algo_class(
            device=device,
            env=env,
            config=eval_cfg.algo.config,
            log_dir=None,
            multi_gpu_cfg=None,
        )
        algo.setup()
        algo.load(checkpoint_path)
        
        policy = algo.get_inference_policy(device)
        
        # 3. Rollout loop
        states_list = []
        actions_list = []
        
        # We define arbitrary steps to map `num_episodes`. 
        # By default num_envs environments run in parallel.
        total_steps = num_episodes * 10  # Arbitrary length heuristic for RL rollout
        print(f"Simulating rollouts for approx {total_steps} steps across {env.num_envs} environments...")
        
        print("calling reset"); obs = env.reset_all(); print("reset done")
        
        # Determine how to extract the actor_obs from env observation
        for step in tqdm(range(total_steps), desc="Rollout Progress"):
            # Some environments return tuple of state/info vs dict, safe inference:
            if isinstance(obs, tuple) and len(obs) > 1 and isinstance(obs[0], dict):
                obs = obs[0]
            
            # Apply Policy
            with torch.no_grad():
                actions = policy(obs)
            
            # Fetch the actual obs that we conditioned on natively
            actor_obs = obs["actor_obs"] if isinstance(obs, dict) else obs
            
            states_list.append(actor_obs.clone().cpu())
            actions_list.append(actions.clone().cpu())
            
            # Env step
            obs, _, _, _ = env.step({"actions": actions})
            
        states_tensor = torch.cat(states_list, dim=0)   # shape [total_steps * num_envs, state_dim]
        actions_tensor = torch.cat(actions_list, dim=0) # shape [total_steps * num_envs, action_dim]
        
        print(f"Collected total of {len(states_tensor)} transitions.")
        return states_tensor, actions_tensor, simulation_app

    except BaseException as e:
        import traceback
        print("\n\n=== EXCEPTION CAUGHT BEFORE FINALLY ===")
        traceback.print_exc()
        print("=======================================\n\n")
        raise e


def train(args: argparse.Namespace):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training CVAE on: {device}")
    
    # IMPORTANT FIX: the `isaaclab.app.AppLauncher` implicitly parses `sys.argv`
    # inside `setup_simulation_environment`. We must clear `sys.argv` here 
    # to prevent it from trying to parse script arguments like "--env".
    import sys
    sys.argv = [sys.argv[0]]
    
    # 1. Collect / Load Data
    cache_path = "cvae_train_data.pt"
    simulation_app = None
    if os.path.exists(cache_path) and not args.force_collect:
        print("Loading cached training data...")
        data = torch.load(cache_path)
        states = data['states']
        actions = data['actions']
    else:
        # Will collect data over the simulator instance based on the checkpoint provided
        states, actions, simulation_app = collect_data(
            sac_checkpoint_dir=args.checkpoint_dir,
            num_episodes=args.episodes
        )
        torch.save({'states': states, 'actions': actions}, cache_path)
    
    dataset = TensorDataset(states, actions)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    state_dim = states.shape[-1]
    action_dim = actions.shape[-1]
    
    # 2. Instantiate CVAE
    cvae = ConditionedVAE(
        state_dim=state_dim, 
        action_dim=action_dim, 
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim
    ).to(device)
    
    optimizer = torch.optim.Adam(cvae.parameters(), lr=args.lr)
    
    # 3. Training Loop
    epochs = args.epochs
    # We apply KL annealing over the first 50% of the epochs to prevent posterior collapse
    # and encourage the network to learn a diverse generative latent space.
    anneal_epochs = max(1, int(epochs * 0.5))
    for epoch in range(1, epochs + 1):
        cvae.train()
        total_epoch_loss = 0.0
        recon_epoch_loss = 0.0
        kl_epoch_loss = 0.0
        
        current_beta = args.beta * min(1.0, (epoch - 1) / anneal_epochs)
        
        progress = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs} (beta={current_beta:.3f})")
        for batch_states, batch_actions in progress:
            batch_states, batch_actions = batch_states.to(device), batch_actions.to(device)
            
            optimizer.zero_grad()
            reconstructed_action, mu, logvar = cvae(batch_states, batch_actions)
            loss, metrics = compute_cvae_loss(reconstructed_action, batch_actions, mu, logvar, beta=current_beta)
            loss.backward()
            optimizer.step()
            
            total_epoch_loss += metrics['loss']
            recon_epoch_loss += metrics['recon_loss']
            kl_epoch_loss += metrics['kl_div']
            
            progress.set_postfix(**metrics)
            
        print(f"Epoch {epoch} | Loss: {total_epoch_loss/len(dataloader):.4f} "
              f"| Recon: {recon_epoch_loss/len(dataloader):.4f} "
              f"| KL: {kl_epoch_loss/len(dataloader):.4f}")
              
    # 4. Save Model Separately
    encoder_save_path = os.path.join(args.save_dir, f"cvae_encoder_{epoch}.pt")
    decoder_save_path = os.path.join(args.save_dir, f"cvae_decoder_{epoch}.pt")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    torch.save(cvae.encoder.state_dict(), encoder_save_path)
    torch.save(cvae.decoder.state_dict(), decoder_save_path)
    print(f"Training Complete. \nEncoder saved to: {encoder_save_path}\nDecoder saved to: {decoder_save_path}")

    if simulation_app is not None:
        close_simulation_app(simulation_app)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CVAE from pre-trained SAC rollouts.")
    parser.add_argument("--checkpoint_dir", type=str, 
                        default="/home/ubuntu22/holosoma/logs/WholeBodyTracking/20260329_023617-t1_23dof_wbt_fast_sac_manager-locomotion",
                        help="Path to the SAC pretrained manager checkpoint.")
    parser.add_argument("--env", type=str, default="WholeBodyTracking-v0", help="Environment to trigger data collection. (Deprecated but preserved for compatibility)")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes for data collection.")
    parser.add_argument("--force_collect", action="store_true", help="Force data collection even if local cache exists.")
    parser.add_argument("--save_dir", type=str, default="./logs/cvae_models/", help="Directory to save the trained encoder and decoder weights.")
    
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent_dim", type=int, default=16, help="Dimensionality of the CVAE z space.")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimensions.")
    parser.add_argument("--beta", type=float, default=1.0, help="KL penalty multiplier.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
