"""Observation manager for computing and processing observations."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import Any

import torch

from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg
from holosoma.managers.utils import resolve_callable

from .base import FunctionTerm, ObservationTermBase


class ObservationManager:
    """Manages observation computation with groups, terms, and processing.

    Each term reads, adds noise, scales and clips its own observation
    (:meth:`ObservationTermBase.observe`); the manager runs the per-frame clock, supplies
    the noise generator and global clip limit, and handles history and concatenation.

    Parameters
    ----------
    cfg : ObservationManagerCfg
        Configuration specifying observation groups and terms.
    env : Any
        Environment instance (typically a ``BaseTask`` subclass).
    device : str
        Device to place tensors on.
    """

    def __init__(self, cfg: ObservationManagerCfg, env: Any, device: str):
        self.cfg = cfg
        self.env = env
        self.device = device
        self.logger = getattr(env, "logger", None)

        # Term objects: group_name -> term_name -> term (plain functions wrapped in FunctionTerm)
        self._terms: dict[str, dict[str, ObservationTermBase]] = {}

        # History buffers: group_name -> term_name -> deque
        self._history_buffers: dict[str, dict[str, deque[torch.Tensor]]] = {}

        # Dedicated noise stream, so observation noise never shifts the global RNG. Seeded from the global
        # seed (which carries the per-rank offset), shifted so it does not replay the global generator's.
        self._noise_generator = torch.Generator(device=device)
        self._noise_generator.manual_seed(torch.initial_seed() + 0x0B5E)

        # Per-frame clock (see new_frame): at most one step_terminal then one step per frame, and
        # every row advanced by step_terminal must be reset before step.
        self._frame_ready = False
        self._terminal_ran = False
        self._terminal_rows = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
        # Envs reset since the last step, re-rendered before it reads (see reset).
        self._rerender_rows = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
        self._has_rerenders = False

        # Initialize groups
        self._initialize_groups()

    def _initialize_groups(self) -> None:
        """Initialize observation groups and resolve term functions."""
        for group_name, group_cfg in self.cfg.groups.items():
            self._terms[group_name] = {}
            self._history_buffers[group_name] = {}

            for term_name, term_cfg in group_cfg.terms.items():
                func = resolve_callable(term_cfg.func, context="observation term")
                if isinstance(func, type) and issubclass(func, ObservationTermBase):
                    self._terms[group_name][term_name] = func(term_cfg, self.env)
                else:
                    self._terms[group_name][term_name] = FunctionTerm(term_cfg, self.env, func)

                # Initialize history buffer if needed (using group-level history_length)
                if group_cfg.history_length > 1:
                    self._history_buffers[group_name][term_name] = deque(maxlen=group_cfg.history_length)

    def new_frame(self) -> None:
        """Mark that the simulation advanced, allowing one :meth:`step_terminal` and one :meth:`step`.

        The env calls it once per control step, after physics; call it yourself when driving the
        manager without an env.
        """
        # step() clears _terminal_ran once it has checked the rows, so this only syncs when it didn't.
        if self._terminal_ran and bool(self._terminal_rows.any()):
            raise RuntimeError(
                "ObservationManager.new_frame() called before resetting the envs advanced by step_terminal()."
            )
        self._frame_ready = True
        self._terminal_ran = False

    def step(self) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Compute this frame's observations for all envs, advancing history and stateful terms.

        The only call that advances observation state for every env. Allowed once per
        :meth:`new_frame`.

        Returns
        -------
        dict[str, torch.Tensor | dict[str, torch.Tensor]]
            Mapping from group names to observation tensors or dictionaries of tensors.
        """
        if not self._frame_ready:
            raise RuntimeError(
                "ObservationManager.step() called without an open frame: it runs once per new_frame(), "
                "so either it already ran this frame or new_frame() was never called."
            )
        if self._terminal_ran and bool(self._terminal_rows.any()):
            raise RuntimeError(
                "ObservationManager.step() called before resetting the envs advanced by step_terminal()."
            )
        self._frame_ready = False
        self._terminal_ran = False
        if self._has_rerenders:
            self._rerender_reset_rows()
        return {group_name: self._compute_group(group_name, None) for group_name in self.cfg.groups}

    def step_terminal(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Step ``env_ids`` in their current (terminal) state and return their observations.

        Stateful terms advance only ``env_ids``, and history buffers are left untouched, so the
        caller must :meth:`reset` those envs before the next :meth:`step`. Allowed once per frame,
        before :meth:`step`.

        Parameters
        ----------
        env_ids : torch.Tensor
            Environment IDs to observe.

        Returns
        -------
        dict[str, torch.Tensor | dict[str, torch.Tensor]]
            As :meth:`step`, with ``len(env_ids)`` rows per tensor.
        """
        if not self._frame_ready or self._terminal_ran:
            raise RuntimeError("ObservationManager.step_terminal() must run once per frame, before step().")
        self._terminal_ran = True
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        self._terminal_rows[env_ids] = True
        return {group_name: self._compute_group(group_name, env_ids) for group_name in self.cfg.groups}

    def _compute_group(self, group_name: str, env_ids: torch.Tensor | None) -> torch.Tensor | dict[str, torch.Tensor]:
        """Compute one group for all envs (``env_ids=None``, advancing history) or for ``env_ids``.

        Parameters
        ----------
        group_name : str
            Name of the observation group to compute.
        env_ids : torch.Tensor or None
            Rows to compute. ``None`` computes all envs and appends to history buffers; otherwise
            history is read for those rows without being modified.

        Returns
        -------
        torch.Tensor | dict[str, torch.Tensor]
            Concatenated tensor when ``group.concatenate`` is ``True``; otherwise
            a dictionary of tensors keyed by term name.
        """
        group_cfg = self.cfg.groups[group_name]
        obs_tensors = {}

        for term_name, term in self._terms[group_name].items():
            # The term reads, adds noise, scales and clips; history is kept per group here.
            try:
                obs = term.observe(
                    self.env,
                    env_ids,
                    with_noise=group_cfg.enable_noise,
                    generator=self._noise_generator,
                    clip_limit=self.cfg.clip_observations,
                )
            except ValueError as err:
                # Terms don't know their config names; add them so the message says which entry failed.
                raise ValueError(f"Observation term '{term_name}' in group '{group_name}': {err}") from err
            if group_cfg.history_length > 1:
                obs = self._apply_history(group_name, term_name, obs, group_cfg, env_ids=env_ids)

            obs_tensors[term_name] = obs

        # Concatenate or return dict
        if group_cfg.concatenate:
            # Concatenate in alphabetically sorted order (to match direct system behavior)
            # Direct system does: sorted(obs_config) before concatenation
            sorted_keys = sorted(obs_tensors.keys())
            for key in sorted_keys:
                if obs_tensors[key].ndim != 2:
                    raise ValueError(
                        f"Observation term '{key}' in concatenate=True group '{group_name}' must be "
                        f"2-D [num_envs, feature], got shape {tuple(obs_tensors[key].shape)}. Image terms "
                        f"(e.g. camera_rgb/camera_depth) belong only in a concatenate=False (dict) group."
                    )
            return torch.cat([obs_tensors[key] for key in sorted_keys], dim=-1)
        return obs_tensors

    def _apply_history(
        self,
        group_name: str,
        term_name: str,
        obs: torch.Tensor,
        group_cfg: ObsGroupCfg,
        *,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply history buffering to an observation term.

        Maintains a circular buffer of past observations and returns them
        concatenated along the feature dimension.

        Parameters
        ----------
        group_name : str
            Name of the observation group.
        term_name : str
            Name of the observation term.
        obs : torch.Tensor
            Current observation tensor with shape ``[num_rows, obs_dim]``.
        group_cfg : ObsGroupCfg
            Configuration for the observation group (contains ``history_length``).
        env_ids : torch.Tensor or None, optional
            ``None`` appends ``obs`` (all envs) to the buffer. Otherwise ``obs`` holds these rows,
            and their history is read with ``obs`` as the newest frame without modifying the buffer.

        Returns
        -------
        torch.Tensor
            Historical observations with shape ``[num_rows, obs_dim * history_length]``.
        """
        buffer = self._history_buffers[group_name][term_name]

        if env_ids is None:
            # Add current observation to buffer
            buffer.append(obs)
            history = list(buffer)
        else:
            # Don't modify buffer - create temporary history with current obs
            history = [frame[env_ids] for frame in buffer] + [obs]
            # Trim to history_length if needed
            if len(history) > group_cfg.history_length:
                history = history[-group_cfg.history_length :]

        # If buffer not full yet, pad with zeros (same as direct behavior)
        num_rows, obs_dim = obs.shape[0], obs.shape[1]
        if len(history) < group_cfg.history_length:
            num_missing = group_cfg.history_length - len(history)
            padding = [torch.zeros(num_rows, obs_dim, device=self.device) for _ in range(num_missing)]
            history = padding + history

        # Stack along time dimension: [num_rows, history_length, obs_dim]
        stacked = torch.stack(history, dim=1)

        # Flatten to [num_rows, history_length * obs_dim]; explicit, since -1 is ambiguous for zero rows
        return stacked.reshape(num_rows, group_cfg.history_length * obs_dim)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset observation history and stateful terms.

        Call on episode resets and teleports. The next :meth:`step` re-renders ``env_ids`` for
        sensors that support partial rendering, so it observes the state written after this call.

        Parameters
        ----------
        env_ids : torch.Tensor or None, optional
            Environment IDs to reset. If ``None``, reset all environments.
        """
        # Normalize environment ids
        env_ids_tensor: torch.Tensor | None
        if env_ids is None:
            env_ids_tensor = None
        elif isinstance(env_ids, torch.Tensor):
            env_ids_tensor = env_ids.to(device=self.device, dtype=torch.long)
        else:
            env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        if env_ids_tensor is None:
            self._terminal_rows[:] = False
            self._rerender_rows[:] = True
            self._has_rerenders = True
        else:
            self._terminal_rows[env_ids_tensor] = False
            if env_ids_tensor.numel() > 0:
                self._rerender_rows[env_ids_tensor] = True
                self._has_rerenders = True

        # Reset or clear history buffers
        for group_buffers in self._history_buffers.values():
            for buffer in group_buffers.values():
                if env_ids_tensor is None:
                    buffer.clear()
                else:
                    if env_ids_tensor.numel() == 0:
                        continue
                    for history in buffer:
                        history[env_ids_tensor] = 0.0

        for group_terms in self._terms.values():
            for term in group_terms.values():
                term.reset(env_ids_tensor)

    def term_shape(self, group_name: str, term_name: str) -> tuple[int, ...]:
        """Per-env output shape of one term, measured without advancing it.

        Parameters
        ----------
        group_name : str
            Name of the observation group.
        term_name : str
            Name of the term within the group.

        Returns
        -------
        tuple[int, ...]
            The term's output shape without the env axis.
        """
        try:
            return self._terms[group_name][term_name].shape(self.env)
        except ValueError as err:
            raise ValueError(f"Observation term '{term_name}' in group '{group_name}': {err}") from err

    def _rerender_reset_rows(self) -> None:
        """Re-render the envs marked by :meth:`reset`, after all state writes for this frame."""
        env_ids = self._rerender_rows.nonzero(as_tuple=False).flatten()
        self._rerender_rows[:] = False
        self._has_rerenders = False
        self.env.simulator.rerender_sensors(env_ids)

    def get_obs_dims(self) -> dict[str, int | dict[str, int | tuple[int, ...]]]:
        """Get observation dimensions for each group.

        Concatenate groups require 2-D terms and report a single summed int. Dict groups
        return per-term feature size: an int for 2-D terms, or the trailing-dim shape tuple
        (``obs.shape[1:]``) for image terms.

        Returns
        -------
        dict[str, int or dict[str, int or tuple]]
            Mapping from group names to observation dimensions. Values are integers
            when the group concatenates terms, otherwise dictionaries of per-term
            dimensions.
        """
        dims: dict[str, int | dict[str, int | tuple[int, ...]]] = {}
        for group_name, group_cfg in self.cfg.groups.items():
            if group_cfg.concatenate:
                # Sum up all term dimensions
                total_dim = 0
                for term_name in group_cfg.terms:
                    shape = self.term_shape(group_name, term_name)
                    if len(shape) != 1:
                        raise ValueError(
                            f"Observation term '{term_name}' in concatenate=True group '{group_name}' must be "
                            f"2-D [num_envs, feature], got per-env shape {shape}. Image terms "
                            f"(e.g. camera_rgb/camera_depth) belong only in a concatenate=False (dict) group."
                        )
                    term_dim = shape[0]

                    # Account for history at group level
                    if group_cfg.history_length > 1:
                        term_dim *= group_cfg.history_length

                    total_dim += term_dim
                dims[group_name] = total_dim
            else:
                # Return dict of per-term feature sizes: an int for 2-D terms, or the
                # trailing-dim shape tuple for image terms ([N, H, W, C] -> (H, W, C)).
                term_dims: dict[str, int | tuple[int, ...]] = {}
                for term_name in group_cfg.terms:
                    shape = self.term_shape(group_name, term_name)
                    term_dims[term_name] = shape[0] if len(shape) == 1 else shape
                dims[group_name] = term_dims
        return dims

    def get_observation_layout(self, group_names: Sequence[str]) -> list[dict[str, Any]]:
        """Describe the flat vector formed by concatenating ``group_names`` in order.

        One entry per term, in concatenation order: groups as given, terms sorted by name
        within a group (as in :meth:`step`). A term occupies ``dim * history_length``
        consecutive values, oldest frame first, with zeros for frames not yet observed. Like
        :meth:`get_obs_dims`, ``dim`` comes from :meth:`term_shape`.

        Parameters
        ----------
        group_names : Sequence[str]
            Concatenated observation groups, in the order the policy consumes them.

        Returns
        -------
        list[dict[str, Any]]
            Per-term ``group``, ``name``, ``func``, ``dim``, ``history_length``, ``scale``,
            ``clip``, ``global_clip`` and ``params``.
        """
        layout: list[dict[str, Any]] = []
        for group_name in group_names:
            group_cfg = self.cfg.groups[group_name]
            if not group_cfg.concatenate:
                raise ValueError(f"Observation group '{group_name}' is not concatenated, so it has no flat layout.")
            for term_name in sorted(group_cfg.terms):
                term_cfg = group_cfg.terms[term_name]
                dim = self.term_shape(group_name, term_name)[0]
                layout.append(
                    {
                        "group": group_name,
                        "name": term_name,
                        "func": term_cfg.func,
                        "dim": dim,
                        "history_length": group_cfg.history_length,
                        "scale": term_cfg.scale,
                        "clip": term_cfg.clip,
                        "global_clip": term_cfg.global_clip,
                        "params": term_cfg.params,
                    }
                )
        return layout
