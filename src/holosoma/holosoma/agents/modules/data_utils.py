from __future__ import annotations

import torch
from torch import Tensor


class RolloutStorage:
    """Simple buffer for storing rollout data during training.

    This is a lightweight storage for PPO rollout data. It stores transitions in tensors
    and provides methods for adding data and generating mini-batches.
    """

    def __init__(self, num_envs: int, num_transitions_per_env: int, device: str = "cpu"):
        """Initialize the rollout storage.

        Args:
            num_envs: Number of parallel environments
            num_transitions_per_env: Number of transitions to store per environment
            device: Device to store tensors on
        """
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.step = 0

        # Dictionary to store all data buffers
        self._buffers: dict[str, Tensor] = {}

    def register(self, key: str, shape: tuple[int, ...] | list[int] = (), dtype: torch.dtype = torch.float):
        """Register a new data key to store in the buffer.

        Args:
            key: Name of the data field (e.g., "obs", "actions", "rewards")
            shape: Shape of each data element (excluding batch dimensions)
            dtype: Data type of the tensor
        """
        if key in self._buffers:
            raise ValueError(f"Key '{key}' already registered")

        if not isinstance(shape, (list, tuple)):
            raise ValueError("shape must be a list or tuple")

        # Create buffer with shape: [num_transitions_per_env, num_envs, *shape]
        buffer = torch.zeros((self.num_transitions_per_env, self.num_envs, *shape), dtype=dtype, device=self.device)
        self._buffers[key] = buffer

    def add(self, **data: Tensor):
        """Add a transition to the buffer.

        Args:
            **data: Keyword arguments where keys are buffer names and values are tensors
                   of shape [num_envs, ...] to store at the current step

        Example:
            storage.add(obs=obs, actions=actions, rewards=rewards, dones=dones)
        """
        if self.step >= self.num_transitions_per_env:
            raise RuntimeError(f"Buffer overflow: step {self.step} >= {self.num_transitions_per_env}")

        for key, value in data.items():
            if key not in self._buffers:
                continue  # Skip keys that aren't registered

            if value.requires_grad:
                raise ValueError(f"Cannot store tensor with requires_grad=True for key '{key}'")

            # Store the data at current step
            self._buffers[key][self.step].copy_(value)

        self.step += 1

    def __getitem__(self, key: str) -> Tensor:
        """Get the buffer for a specific key.

        Args:
            key: Name of the buffer

        Returns:
            Tensor of shape [num_transitions_per_env, num_envs, ...]
        """
        if key not in self._buffers:
            raise KeyError(f"Key '{key}' not registered")
        return self._buffers[key]

    def __setitem__(self, key: str, value: Tensor):
        """Set the entire buffer for a specific key.

        Args:
            key: Name of the buffer
            value: Tensor of shape [num_transitions_per_env, num_envs, ...]
        """
        if key not in self._buffers:
            raise KeyError(f"Key '{key}' not registered")

        if value.requires_grad:
            raise ValueError("Cannot store tensor with requires_grad=True")

        self._buffers[key].copy_(value)

    def clear(self):
        """Clear the buffer and reset the step counter."""
        self.step = 0

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8):
        """Generate randomized mini-batches for training.

        This flattens the time and environment dimensions and creates random mini-batches.

        Args:
            num_mini_batches: Number of mini-batches to create per epoch
            num_epochs: Number of times to iterate over the data

        Yields:
            Dictionary mapping buffer keys to mini-batch tensors
        """
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches

        # Flatten all buffers: [num_transitions_per_env, num_envs, ...] -> [batch_size, ...]
        flattened = {key: buf.flatten(0, 1) for key, buf in self._buffers.items()}

        for _ in range(num_epochs):
            # Reshuffle each epoch so the same minibatch isn't trained twice
            # in a row — matches the reference mini_batch_generator_distillation,
            # which is re-invoked per epoch.
            indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_indices = indices[start:end]

                # Extract mini-batch for each buffer
                mini_batch = {key: flattened[key][batch_indices] for key in self._buffers}
                yield mini_batch
