# Conditional Variational Autoencoder (CVAE) in Holosoma

This document describes how to train, use, and modify the Conditional Variational Autoencoder (CVAE) integrated into the `holosoma` library. This is typically used for generative motion modeling, trajectory optimization, or robust reinforcement learning leveraging prior behaviors.

## 1. CVAE Module Introduction & Architecture

The CVAE generates diverse and plausible character actions $a_t$ based on the environment state conditioning $s_t$.

The implemented architecture (`src/holosoma/holosoma/agents/cvae/cvae_model.py`) contains:
- **Encoder**: Takes both $[s_t, a_t]$ as input, processes them through Multi-Layer Perceptrons (MLPs), and outputs the latent distribution parameters $\mu, \sigma$.
- **Latent Space (z)**: Sampled using the Reparameterization Trick $z = \mu + \sigma \odot \epsilon$.
- **Decoder**: Takes the sampled latent variable $[z, s_t]$ and attempts to reconstruct the exact action $\hat{a}_t$.

The model is optimized using the sum of the **Reconstruction Loss (MSE)** and the **Kullback-Leibler (KL) Divergence**, regulated by a hyperparameter $\beta$.

## 2. Environment Dependencies

Ensure that you have set up the `holosoma` repository correctly:
- Valid python environment (`pip install -e .` from within `src/holosoma`).
- Access to the Isaac Gym / MuJoCo dependencies per your chosen backend.

## 3. Data Collection via Provided Checkpoint

To train the CVAE, we must first capture the expert distribution of $a_t$ given $s_t$. We do this by rolling out trajectories using a pre-trained Soft Actor-Critic (SAC) model.

You can use the pre-trained weights located in:
```
/home/ubuntu22/holosoma/logs/WholeBodyTracking/20260329_023617-t1_23dof_wbt_fast_sac_manager-locomotion
```

### Collecting the data
The data is collected implicitly when you launch the training script. By default, it runs the environment policy with the SAC agent to populate an offline dataset.

## 4. Training the CVAE

We provide a script (`scripts/train_cvae.py`) to handle both data rollout and model training.

Execute the training script from the `scripts` directories:

```bash
python src/holosoma/holosoma/train_cvae.py \
    --checkpoint_dir /home/ubuntu22/holosoma/logs/WholeBodyTracking/20260329_023617-t1_23dof_wbt_fast_sac_manager-locomotion/model_0400000.pt \
    --env WholeBodyTracking-v0 \
    --episodes 500 \
    --epochs 100 \
    --latent_dim 16 \
    --batch_size 256 \
    --save_dir ./logs/cvae_models/
```

_Note: If data is already collected and cached in `cvae_train_data.pt`, the script will automatically load it to save time unless `--force_collect` is provided._

## 5. Loading and Inference

Once the CVAE is trained, the Encoder and Decoder are saved as separate weight files (e.g., `cvae_encoder_100.pt` and `cvae_decoder_100.pt`). 
For inference in the environment, you typically only need the Decoder initialized and loaded.

Here is an example snippet of how you would load the decoder and query actions from states:

```python
import torch
from holosoma.agents.cvae.cvae_model import CVAEDecoder

state_dim = ... # match your state dim
action_dim = ... # match your action dim
latent_dim = 16

# 1. Initialize logic
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# We only instantiate the Decoder for inference
decoder = CVAEDecoder(state_dim, action_dim, latent_dim).to(device)

# 2. Load trained Decoder weights 
decoder_weights_path = "./logs/cvae_models/cvae_decoder_100.pt"
decoder.load_state_dict(torch.load(decoder_weights_path, map_location=device))
decoder.eval()

# 3. Inference
# Given an environment state:
state = environment.get_state()  # Shape: (batch_size, state_dim)
state_tensor = torch.tensor(state, dtype=torch.float32).to(device)

with torch.no_grad():
    # Sample z ~ N(0, I)
    batch_size = state_tensor.shape[0] if state_tensor.dim() > 1 else 1
    z = torch.randn(batch_size, latent_dim, device=device)
    if state_tensor.dim() == 1:
        z = z.squeeze(0)
    
    # Generate action
    action_pred_tensor = decoder(z, state_tensor)

action_pred = action_pred_tensor.cpu().numpy()
# environment.step(action_pred)
```
