"""Action manager for processing and applying actions."""

from __future__ import annotations

import importlib
from typing import Any

import torch

from holosoma.config_types.action import ActionManagerCfg

from .base import ActionTermBase


def _raise_if_non_finite(values: torch.Tensor, description: str) -> None:
    invalid = ~torch.isfinite(values)
    if values.device.type == "cuda":
        torch._assert_async(~torch.any(invalid), f"{description} contain non-finite values")
        return
    if not torch.any(invalid):
        return

    env_ids = torch.where(invalid.reshape(invalid.shape[0], -1).any(dim=1))[0]
    shown_ids = env_ids[:10].detach().cpu().tolist()
    suffix = "" if env_ids.numel() <= len(shown_ids) else f" (showing {len(shown_ids)} of {env_ids.numel()})"
    raise FloatingPointError(f"{description} contain non-finite values; env_ids={shown_ids}{suffix}")


class ActionManager:
    """Manages action processing and application.

    The action manager handles splitting the raw action vector into different
    action terms and orchestrates their processing and application. It operates
    in two phases:

    1. **process_actions**: Called once per environment step. Splits the raw
       action vector and passes each portion to the corresponding action term
       for processing (e.g., scaling, clipping, computing targets).

    2. **apply_actions**: Called at every simulation step. Applies the processed
       actions to the environment (e.g., setting joint targets).

    Parameters
    ----------
    cfg : ActionManagerCfg
        Configuration specifying action terms
    env : Any
        Environment instance (typically BaseTask or subclass)
    device : str
        Device to place tensors on
    """

    def __init__(self, cfg: ActionManagerCfg, env: Any, device: str):
        self.cfg = cfg
        self.env = env
        self.device = device
        self.logger = getattr(env, "logger", None)

        # Storage for action term instances
        self._term_instances: dict[str, ActionTermBase] = {}
        self._term_names: list[str] = []
        self._term_dims: list[int] = []

        # Initialize action terms
        self._initialize_terms()

        # Total action dimension
        self._total_action_dim = sum(self._term_dims)

        # Buffers for actions
        self._action = torch.zeros((self.env.num_envs, self._total_action_dim), device=self.device)
        self._prev_action = torch.zeros_like(self._action)

    def _initialize_terms(self) -> None:
        """Initialize action terms and resolve their classes."""
        for term_name, term_cfg in self.cfg.terms.items():
            # Resolve the action term class
            term_class = self._resolve_class(term_cfg.func)

            # Check if it's a valid action term class
            if not issubclass(term_class, ActionTermBase):
                raise TypeError(f"Action term '{term_name}' must inherit from ActionTermBase. Got: {term_class}")

            # Instantiate the action term
            instance = term_class(term_cfg, self.env)
            self._term_instances[term_name] = instance
            self._term_names.append(term_name)
            self._term_dims.append(instance.action_dim)

    def _resolve_class(self, class_path: str) -> type:
        """Resolve class from string path.

        Parameters
        ----------
        class_path : str
            String path like "module.path:ClassName"

        Returns
        -------
        type
            Resolved class type

        Raises
        ------
        ValueError
            If string path is malformed or class not found
        """
        if ":" not in class_path:
            raise ValueError(f"Action term class path must be in format 'module:ClassName', got: {class_path}")

        module_path, class_name = class_path.split(":", 1)
        try:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Failed to import action term class '{class_path}': {e}") from e

    def setup(self) -> None:
        """Setup action manager after all managers are initialized.

        This method calls setup() on all action terms after all managers
        (including randomization manager) have been created and initialized.
        """
        for term in self._term_instances.values():
            term.setup()

    def iter_terms(self):
        """Iterate over registered action terms."""
        return self._term_instances.items()

    @property
    def total_action_dim(self) -> int:
        """Total dimension of the action space."""
        return self._total_action_dim

    @property
    def action(self) -> torch.Tensor:
        """The processed actions sent to the environment.

        Returns
        -------
        torch.Tensor
            Clipped/processed action tensor [num_envs, total_action_dim]
        """
        return self._action

    @property
    def prev_action(self) -> torch.Tensor:
        """The previous processed actions sent to the environment.

        Returns
        -------
        torch.Tensor
            Previous clipped/processed action tensor [num_envs, total_action_dim]
        """
        return self._prev_action

    @property
    def active_terms(self) -> list[str]:
        """Names of active action terms."""
        return self._term_names

    @property
    def action_term_dims(self) -> list[int]:
        """Dimensions of each action term."""
        return self._term_dims

    def process_actions(self, actions: torch.Tensor) -> None:
        """Process raw actions by distributing them to action terms.

        This should be called once per environment step.

        Parameters
        ----------
        actions : torch.Tensor
            Raw action tensor [num_envs, total_action_dim]

        Raises
        ------
        ValueError
            If action dimension doesn't match expected
        FloatingPointError
            If raw or processed actions contain non-finite values
        """
        expected_shape = (self.env.num_envs, self._total_action_dim)
        if actions.shape != expected_shape:
            raise ValueError(f"Invalid action shape. Expected: {expected_shape}, received: {tuple(actions.shape)}")

        actions = actions.to(self.device)
        _raise_if_non_finite(actions, "Raw actions")

        # Split actions and process each term
        idx = 0
        processed_actions = []
        for term_name in self._term_names:
            term = self._term_instances[term_name]
            term_actions = actions[:, idx : idx + term.action_dim]
            term.process_actions(term_actions)
            processed = term.processed_actions
            if processed.shape != term_actions.shape:
                raise ValueError(
                    f"Action term '{term_name}' returned wrong processed-action shape. "
                    f"Expected {tuple(term_actions.shape)}, got {tuple(processed.shape)}"
                )
            _raise_if_non_finite(processed, f"Processed actions from term '{term_name}'")
            processed_actions.append(processed)
            idx += term.action_dim

        # Advance history only after every term has processed successfully. Rewards
        # and observations therefore see the same clipped actions that control uses.
        self._prev_action[:] = self._action
        self._action[:] = torch.cat(processed_actions, dim=1) if processed_actions else actions

    def apply_actions(self) -> None:
        """Apply processed actions to the environment.

        This should be called at every simulation step.
        """
        for term in self._term_instances.values():
            term.apply_actions()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset action manager state.

        Parameters
        ----------
        env_ids : torch.Tensor or None, optional
            Environment IDs to reset. If None, reset all.
        """
        # Reset action history
        if env_ids is None:
            self._prev_action[:] = 0.0
            self._action[:] = 0.0
        else:
            self._prev_action[env_ids] = 0.0
            self._action[env_ids] = 0.0

        # Reset all action terms
        for term in self._term_instances.values():
            term.reset(env_ids=env_ids)

    def get_term(self, name: str) -> ActionTermBase:
        """Get action term by name.

        Parameters
        ----------
        name : str
            Name of the action term

        Returns
        -------
        ActionTermBase
            The action term instance

        Raises
        ------
        KeyError
            If term name not found
        """
        return self._term_instances[name]

    def __str__(self) -> str:
        """String representation of action manager."""
        msg = f"<ActionManager> contains {len(self._term_names)} active terms.\n"
        msg += f"Total action dimension: {self._total_action_dim}\n"
        msg += "Terms:\n"
        for name, dim in zip(self._term_names, self._term_dims):
            msg += f"  - {name}: {dim}\n"
        return msg
