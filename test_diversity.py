import torch
import sys
import os

# Add the holosoma package to the path
sys.path.insert(0, os.path.abspath("src/holosoma"))
from holosoma.agents.cvae.cvae_model import CVAEDecoder

def test_cvae_diversity():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data to get a real state and the dimensions
    data_path = "cvae_train_data.pt"
    if not os.path.exists(data_path):
        print(f"Error: Could not find '{data_path}'. Please ensure training data was cached.")
        return

    data = torch.load(data_path, map_location=device)
    states = data['states']
    actions = data['actions']

    state_dim = states.shape[-1]
    action_dim = actions.shape[-1]
    latent_dim = 16

    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    # Initialize decoder
    decoder = CVAEDecoder(state_dim, action_dim, latent_dim).to(device)
    decoder_weights_path = "./logs/cvae_models/cvae_decoder_100.pt"
    
    if not os.path.exists(decoder_weights_path):
        print(f"Error: Could not find weights at '{decoder_weights_path}'")
        return
        
    decoder.load_state_dict(torch.load(decoder_weights_path, map_location=device))
    decoder.eval()

    # Pick the first state as our condition
    test_state = states[0:1]
    real_action = actions[0:1]

    print("\n--- Testing Generation Diversity ---")
    print("We will input the SAME state but sample DIFFERENT latent variables (z).")
    
    num_samples = 5
    generated_actions = []

    with torch.no_grad():
        for i in range(num_samples):
            z = torch.randn(1, latent_dim, device=device)
            action_pred = decoder(z, test_state)
            generated_actions.append(action_pred)
            
    # Combine generated actions to calculate basic stats
    gen_tensor = torch.cat(generated_actions, dim=0) # shape (5, action_dim)
    # std deviation across the 5 samples for each of the 23 action dimensions, then averaged
    action_std = gen_tensor.std(dim=0).mean().item()  
    
    print(f"\nReal Action (first 5 dims):      {real_action[0, :5].cpu().numpy().round(4)}")
    for i in range(num_samples):
        print(f"Generated Action {i+1} (first 5 dims): {gen_tensor[i, :5].cpu().numpy().round(4)}")

    print(f"\nAverage Std Dev across all {action_dim} action dimensions: {action_std:.4f}")
    if action_std > 1e-4:
        print("✅ SUCCESS! The CVAE generates diverse actions for the exact same state thanks to KL annealing.")
    else:
        print("❌ Warning: The generated actions are nearly identical. You might be experiencing posterior collapse.")

if __name__ == "__main__":
    test_cvae_diversity()
