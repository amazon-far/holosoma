"""Reward manager for computing reward signals."""

from __future__ import annotations

import importlib
from typing import Any

import torch

from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg

from .base import RewardTermBase


def _raise_if_non_finite(values: torch.Tensor, description: str) -> None:
    invalid = ~torch.isfinite(values)
    if values.device.type == "cuda":
        torch._assert_async(~torch.any(invalid), f"{description} contains non-finite values")
        return
    if not torch.any(invalid):
        return

    env_ids = torch.where(invalid.reshape(invalid.shape[0], -1).any(dim=1))[0]
    shown_ids = env_ids[:10].detach().cpu().tolist()
    suffix = "" if env_ids.numel() <= len(shown_ids) else f" (showing {len(shown_ids)} of {env_ids.numel()})"
    raise FloatingPointError(f"{description} contains non-finite values; env_ids={shown_ids}{suffix}")


class RewardManager:
    """Manages reward computation as a weighted sum of individual terms.

    The reward manager computes the total reward by evaluating each configured
    reward term, multiplying by its weight and the environment's time step (dt),
    and summing the results. It tracks episodic sums for logging and supports
    both stateless (function) and stateful (class) reward terms.

    Parameters
    ----------
    cfg : RewardManagerCfg
        Configuration specifying reward terms and settings.
    env : Any
        Environment instance (typically a ``BaseTask`` subclass).
    device : str
        Device where tensors should be allocated.
    """

    def __init__(self, cfg: RewardManagerCfg, env: Any, device: str):
        self.cfg = cfg
        self.env = env
        self.device = device
        self.logger = getattr(env, "logger", None)

        # Storage for resolved functions and stateful terms
        self._term_funcs: dict[str, Any] = {}
        self._term_instances: dict[str, RewardTermBase] = {}
        self._term_names: list[str] = []
        self._term_cfgs: list[RewardTermCfg] = []

        # Initialize terms
        self._initialize_terms()

        # Buffers for reward tracking
        self._reward_buf = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Episode sums for each term (for logging)
        self._episode_sums: dict[str, torch.Tensor] = {}
        self._episode_sums_raw: dict[str, torch.Tensor] = {}
        for term_name in self._term_names:
            self._episode_sums[term_name] = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            self._episode_sums_raw[term_name] = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

    def _initialize_terms(self) -> None:
        """Initialize reward terms and resolve their functions/classes."""
        for term_name, term_cfg in self.cfg.terms.items():
            # Skip terms with zero weight
            if term_cfg.weight == 0.0:
                continue

            # Resolve function or class
            func = self._resolve_function(term_cfg.func)

            # Check if it's a class (stateful) or function (stateless)
            if isinstance(func, type) and issubclass(func, RewardTermBase):
                # Stateful term - instantiate
                instance = func(term_cfg, self.env)
                self._term_instances[term_name] = instance
            else:
                # Stateless function
                self._term_funcs[term_name] = func

            self._term_names.append(term_name)
            self._term_cfgs.append(term_cfg)

    def _resolve_function(self, func: Any | str) -> Any:
        """Resolve a reward callable or class from a string specification.

        Parameters
        ----------
        func : Any or str
            Function or class reference, or a string like ``"module:object_name"``.

        Returns
        -------
        Any
            Resolved callable or class.

        Raises
        ------
        ValueError
            If the string path is malformed or the target cannot be imported.
        """
        if isinstance(func, str):
            # Parse string like "module.path:function_name"
            if ":" not in func:
                raise ValueError(f"Function string must be in format 'module:function', got: {func}")

            module_path, func_name = func.split(":", 1)
            try:
                module = importlib.import_module(module_path)
                return getattr(module, func_name)
            except (ImportError, AttributeError) as e:
                raise ValueError(f"Failed to import function '{func}': {e}") from e
        return func

    @property
    def active_terms(self) -> list[str]:
        """Names of active reward terms."""
        return self._term_names

    @property
    def episode_sums(self) -> dict[str, torch.Tensor]:
        """Episodic sums for each reward term (scaled)."""
        return self._episode_sums

    @property
    def episode_sums_raw(self) -> dict[str, torch.Tensor]:
        """Episodic sums for each reward term (raw, unscaled)."""
        return self._episode_sums_raw

    def compute(self, dt: float) -> torch.Tensor:
        """Compute the total reward as a weighted sum of individual terms.

        Each reward term is evaluated, scaled by its configured weight and the
        environment time step, and accumulated into the total reward. Episodic
        sums are updated for logging purposes.

        Notes
        -----
        Curriculum scaling is handled by directly modifying term weights via
        :meth:`set_term_cfg`, rather than through extra scaling parameters.

        Parameters
        ----------
        dt : float
            Environment time-step interval.

        Returns
        -------
        torch.Tensor
            Net reward tensor with shape ``[num_envs]``.
        """
        # Reset computation
        self._reward_buf[:] = 0.0

        # Iterate over all reward terms
        for term_name, term_cfg in zip(self._term_names, self._term_cfgs):
            # Compute raw reward value
            if term_name in self._term_instances:
                # Stateful term
                instance = self._term_instances[term_name]
                rew_raw = instance(self.env, **term_cfg.params)
            else:
                # Stateless function
                func = self._term_funcs[term_name]
                rew_raw = func(self.env, **term_cfg.params)

            # Validate shape
            if rew_raw.shape != self._reward_buf.shape:
                raise ValueError(
                    f"Reward term '{term_name}' returned wrong shape. "
                    f"Expected [{self.env.num_envs}], got {rew_raw.shape}"
                )
            _raise_if_non_finite(rew_raw, f"Reward term '{term_name}' raw output")

            # Scale by weight and dt
            rew_scaled = rew_raw * term_cfg.weight * dt
            _raise_if_non_finite(rew_scaled, f"Reward term '{term_name}' scaled output")

            # Accumulate
            accumulated = self._reward_buf + rew_scaled
            _raise_if_non_finite(accumulated, f"Total reward after term '{term_name}'")
            self._reward_buf[:] = accumulated

            # Track episodic sums
            self._episode_sums[term_name] += rew_scaled
            self._episode_sums_raw[term_name] += rew_raw

        # Optionally clip to positive
        if self.cfg.only_positive_rewards:
            self._reward_buf[:] = torch.clip(self._reward_buf, min=0.0)

        return self._reward_buf

    def reset(self, env_ids: torch.Tensor | None = None) -> dict[str, dict[str, torch.Tensor]]:
        """Reset reward tracking and return episodic sums for logging.

        Parameters
        ----------
        env_ids : torch.Tensor or None, optional
            Environment IDs to reset. If ``None``, reset all environments.

        Returns
        -------
        dict[str, dict[str, torch.Tensor]]
            Dictionary mirroring the direct reward path structure::

                {
                    "episode": {term_name: tensor_per_reset_env},
                    "episode_all": {term_name: tensor_per_all_envs},
                    "raw_episode": {...},
                    "raw_episode_all": {...},
                }
        """
        extras: dict[str, dict[str, torch.Tensor]] = {
            "episode": {},
            "episode_all": {},
            "raw_episode": {},
            "raw_episode_all": {},
        }

        # Resolve environment ids to operate on
        if env_ids is None:
            env_ids_tensor: torch.Tensor | None = None
            env_ids_slice: slice | torch.Tensor = slice(None)
        else:
            if isinstance(env_ids, torch.Tensor):
                env_ids_tensor = env_ids.to(device=self.device, dtype=torch.long)
            else:
                env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

            env_ids_slice = env_ids_tensor

        # Helper to detach values before zeroing internal buffers
        def _clone(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.detach().clone()

        # Populate scaled reward statistics
        for term_name in self._term_names:
            rew_all = self._episode_sums[term_name] / self.env.max_episode_length_s
            extras["episode_all"][f"rew_{term_name}"] = _clone(rew_all)
            if env_ids_tensor is None:
                extras["episode"][f"rew_{term_name}"] = _clone(rew_all)
            else:
                extras["episode"][f"rew_{term_name}"] = _clone(rew_all[env_ids_slice])

            # Reset episodic sums for the completed environments
            self._episode_sums[term_name][env_ids_slice] = 0.0

        # Populate raw (unscaled) reward statistics
        for term_name in self._term_names:
            rew_raw_all = self._episode_sums_raw[term_name] / self.env.max_episode_length_s
            extras["raw_episode_all"][f"raw_rew_{term_name}"] = _clone(rew_raw_all)
            if env_ids_tensor is None:
                extras["raw_episode"][f"raw_rew_{term_name}"] = _clone(rew_raw_all)
            else:
                extras["raw_episode"][f"raw_rew_{term_name}"] = _clone(rew_raw_all[env_ids_slice])

            self._episode_sums_raw[term_name][env_ids_slice] = 0.0

        # Reset stateful reward terms
        for instance in self._term_instances.values():
            instance.reset(env_ids=env_ids_tensor)

        return extras

    def get_term(self, name: str) -> Any:
        """Get reward term function or instance by name.

        Parameters
        ----------
        name : str
            Name of the reward term.

        Returns
        -------
        Any
            Reward term function or instance.

        Raises
        ------
        KeyError
            If the term name is not found.
        """
        if name in self._term_instances:
            return self._term_instances[name]
        if name in self._term_funcs:
            return self._term_funcs[name]
        raise KeyError(f"Reward term '{name}' not found")

    def get_term_cfg(self, name: str) -> RewardTermCfg:
        """Get reward term configuration by name.

        Parameters
        ----------
        name : str
            Name of the reward term.

        Returns
        -------
        RewardTermCfg
            Configuration for the specified reward term.

        Raises
        ------
        KeyError
            If the term name is not found.
        """
        try:
            idx = self._term_names.index(name)
            return self._term_cfgs[idx]
        except ValueError:
            raise KeyError(f"Reward term '{name}' not found")

    def set_term_cfg(self, name: str, cfg: RewardTermCfg) -> None:
        """Set reward term configuration by name.

        Parameters
        ----------
        name : str
            Name of the reward term.
        cfg : RewardTermCfg
            New configuration for the term.

        Raises
        ------
        KeyError
            If the term name is not found.
        """
        try:
            idx = self._term_names.index(name)
            self._term_cfgs[idx] = cfg
        except ValueError:
            raise KeyError(f"Reward term '{name}' not found")

    def __str__(self) -> str:
        """String representation of reward manager."""
        msg = f"<RewardManager> contains {len(self._term_names)} active terms.\n"
        msg += "Terms:\n"
        for name, cfg in zip(self._term_names, self._term_cfgs):
            msg += f"  - {name}: weight={cfg.weight}\n"
        return msg
