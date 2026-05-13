from __future__ import annotations

from collections.abc import Sequence

from holosoma.utils.safe_torch_import import nn, torch


def _make_activation(name: str) -> nn.Module:
    activation_name = name.lower()
    if activation_name == "relu":
        return nn.ReLU()
    if activation_name == "gelu":
        return nn.GELU()
    if activation_name == "elu":
        return nn.ELU()
    if activation_name == "tanh":
        return nn.Tanh()
    if activation_name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation '{name}'.")


class TransitionNormalizer(nn.Module):
    def __init__(
        self,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
    ):
        super().__init__()
        self.register_buffer("state_mean", state_mean.reshape(1, -1))
        self.register_buffer("state_std", state_std.reshape(1, -1))
        self.register_buffer("action_mean", action_mean.reshape(1, -1))
        self.register_buffer("action_std", action_std.reshape(1, -1))

    @classmethod
    def from_dataset(cls, states: torch.Tensor, actions: torch.Tensor, eps: float = 1e-6) -> TransitionNormalizer:
        state_mean = states.mean(dim=0)
        state_std = states.std(dim=0, unbiased=False).clamp_min(eps)
        action_mean = actions.mean(dim=0)
        action_std = actions.std(dim=0, unbiased=False).clamp_min(eps)
        return cls(state_mean=state_mean, state_std=state_std, action_mean=action_mean, action_std=action_std)

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        return (state - self.state_mean) / self.state_std

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return (action - self.action_mean) / self.action_std

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.normalize_state(state), self.normalize_action(action)], dim=-1)


class TransitionPredictor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (512, 512, 256),
        activation: str = "SiLU",
        use_layer_norm: bool = True,
        dropout_prob: float = 0.0,
        predict_residual: bool = True,
        normalize_inputs: bool = True,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.predict_residual = predict_residual
        self.normalize_inputs = normalize_inputs

        input_dim = state_dim + action_dim
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(_make_activation(activation))
            if dropout_prob > 0.0:
                layers.append(nn.Dropout(dropout_prob))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, state_dim))
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        normalizer: TransitionNormalizer | None = None,
    ) -> torch.Tensor:
        if self.normalize_inputs and normalizer is not None:
            inputs = normalizer(state, action)
        else:
            inputs = torch.cat([state, action], dim=-1)

        delta_state = self.network(inputs)
        if self.predict_residual:
            return state + delta_state
        return delta_state
