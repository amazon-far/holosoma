import torch
import torch.nn as torch_nn
import torch.nn.functional as F

class CVAEEncoder(torch_nn.Module):
    """
    Encoder for the Conditional Variational Autoencoder (CVAE).
    
    This network takes the current state (s_t) and action (a_t) to output the parameters
    for the latent distribution (mean and log variance).
    """

    def __init__(self, state_dim: int, action_dim: int, latent_dim: int, hidden_dim: int = 256):
        """
        Args:
            state_dim: Dimension of the state space.
            action_dim: Dimension of the action space.
            latent_dim: Dimension of the latent space (z).
            hidden_dim: Number of units in hidden layers.
        """
        super().__init__()
        # Input to the encoder is [state, action] concatenated
        input_dim = state_dim + action_dim
        
        self.encoder_net = torch_nn.Sequential(
            torch_nn.Linear(input_dim, hidden_dim),
            torch_nn.ReLU(),
            torch_nn.Linear(hidden_dim, hidden_dim),
            torch_nn.ReLU()
        )
        self.fc_mu = torch_nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = torch_nn.Linear(hidden_dim, latent_dim)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the encoder.
        
        Args:
            state: Environment state tensor of shape (batch_size, state_dim).
            action: Action tensor of shape (batch_size, action_dim).
            
        Returns:
            mu: Mean of the latent Gaussian distribution.
            logvar: Log variance of the latent Gaussian distribution.
        """
        x = torch.cat([state, action], dim=-1)
        hidden = self.encoder_net(x)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        # Using softplus or clamping on logvar can add stability, 
        # but standard logvar outputs directly from linear layer.
        return mu, logvar


class CVAEDecoder(torch_nn.Module):
    """
    Decoder for the Conditional Variational Autoencoder (CVAE).
    
    This network takes the latent variable (z) and current state (s_t) to reconstruct
    or predict the action.
    """

    def __init__(self, state_dim: int, action_dim: int, latent_dim: int, hidden_dim: int = 256):
        """
        Args:
            state_dim: Dimension of the state space.
            action_dim: Dimension of the action space.
            latent_dim: Dimension of the latent space (z).
            hidden_dim: Number of units in hidden layers.
        """
        super().__init__()
        # Input to the decoder is [latent, state] concatenated
        input_dim = latent_dim + state_dim
        
        self.decoder_net = torch_nn.Sequential(
            torch_nn.Linear(input_dim, hidden_dim),
            torch_nn.ReLU(),
            torch_nn.Linear(hidden_dim, hidden_dim),
            torch_nn.ReLU(),
            torch_nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, latent: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the decoder.
        
        Args:
            latent: Latent variable tensor (sampled or prior) of shape (batch_size, latent_dim).
            state: Environment state tensor of shape (batch_size, state_dim).
            
        Returns:
            reconstructed_action: Predicted action tensor of shape (batch_size, action_dim).
        """
        x = torch.cat([latent, state], dim=-1)
        reconstructed_action = self.decoder_net(x)
        return reconstructed_action


class ConditionedVAE(torch_nn.Module):
    """
    Conditional Variational Autoencoder (CVAE).
    
    Combines the CVAEEncoder and CVAEDecoder to formulate the full autoencoder.
    """

    def __init__(self, state_dim: int, action_dim: int, latent_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = CVAEEncoder(state_dim, action_dim, latent_dim, hidden_dim)
        self.decoder = CVAEDecoder(state_dim, action_dim, latent_dim, hidden_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + std * epsilon
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass through the CVAE (Encoding -> Reparameterization -> Decoding).
        
        Returns:
            reconstructed_action: The predicted/reconstructed action.
            mu: The latent mean.
            logvar: The latent log variance.
        """
        mu, logvar = self.encoder(state, action)
        z = self.reparameterize(mu, logvar)
        reconstructed_action = self.decoder(z, state)
        return reconstructed_action, mu, logvar

    def inference(self, state: torch.Tensor) -> torch.Tensor:
        """
        Generate actions given only states by sampling z from N(0, I).
        
        Args:
            state: The current environment state.
            
        Returns:
            action: The generated action.
        """
        batch_size = state.shape[0] if state.dim() > 1 else 1
        device = state.device
        z = torch.randn(batch_size, self.latent_dim, device=device)
        if state.dim() == 1:
            z = z.squeeze(0)
        return self.decoder(z, state)

def compute_cvae_loss(reconstructed_action: torch.Tensor, action: torch.Tensor, 
                      mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0) -> tuple[torch.Tensor, dict]:
    """
    Computes the CVAE loss comprising the Reconstruction Loss (MSE) and KL Divergence.
    
    Args:
        reconstructed_action: Action predicted by Decoder.
        action: Ground truth action.
        mu: Mean of the latent representation.
        logvar: Log variance of the latent representation.
        beta: Weighting parameter for KL divergence (Beta-VAE concept).
        
    Returns:
        total_loss: The scalar loss tensor to backprop against.
        info: A dictionary of loss components for logging.
    """
    # Reconstruction Loss (MSE)
    recon_loss = F.mse_loss(reconstructed_action, action, reduction='mean')
    
    # KL Divergence
    # KL(N(mu, sigma^2) || N(0, 1)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
    
    total_loss = recon_loss + beta * kl_div
    
    return total_loss, {
        "loss": total_loss.item(),
        "recon_loss": recon_loss.item(),
        "kl_div": kl_div.item()
    }
