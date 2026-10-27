"""Base classes and protocols for observation terms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, final

from holosoma.utils.safe_torch_import import torch

if TYPE_CHECKING:
    from holosoma.config_types.observation import ObsTermCfg


class ObservationTermBase(ABC):
    """Base class for observation terms.

    Subclass it for terms that keep internal state (e.g. latency buffers, filters). A plain function
    in a term config is wrapped in :class:`FunctionTerm`, so the manager handles every term the same way.
    Subclasses implement :meth:`read` and may override :meth:`reset` and :meth:`generate_noise`.

    The manager reads a term once per frame for all envs, and may additionally read it for the
    envs about to reset to observe their terminal state.
    """

    def __init__(self, cfg: ObsTermCfg, env: Any):
        """Initialize observation term.

        Args:
            cfg: Configuration for this observation term
            env: Environment instance (typically a ``BaseTask`` subclass)
        """
        self.cfg = cfg
        self.env = env

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset internal state for specified environments. Does nothing by default.

        Args:
            env_ids: Environment IDs to reset. If None, reset all environments.
        """
        del env_ids

    @abstractmethod
    def read(self, env: Any, env_ids: torch.Tensor | None = None, **kwargs: Any) -> torch.Tensor:
        """Compute the observation of ``env_ids`` and advance only their internal state.

        Envs not in ``env_ids`` must keep their state untouched. An empty ``env_ids`` is how
        :meth:`shape` measures the output: return ``[0, *obs_shape]`` and change nothing.

        Args:
            env: Environment instance
            env_ids: Environment IDs to compute and advance. If None, all environments.
            **kwargs: Additional parameters from config

        Returns:
            Observation tensor of shape [num_envs, *obs_shape], or [len(env_ids), *obs_shape]
        """

    def generate_noise(self, obs: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
        """Return ``obs`` with sensor noise added. Override to model a specific sensor.

        The default adds uniform noise in ``[-cfg.noise, cfg.noise]`` and does nothing when ``cfg.noise`` is 0.

        Args:
            obs: Raw observation from :meth:`read`
            generator: Random generator to draw from, so noise does not shift the global RNG

        Returns:
            Noisy observation, same shape as ``obs``: a new tensor, or ``obs`` itself, never a view of it
        """
        if self.cfg.noise <= 0:
            return obs
        rand = torch.rand(obs.shape, generator=generator, device=obs.device, dtype=obs.dtype)
        return obs + (rand * 2.0 - 1.0) * self.cfg.noise

    @final
    def postprocess(self, obs: torch.Tensor, clip_limit: float) -> torch.Tensor:
        """Apply the config's scale and clip, then ``[-clip_limit, clip_limit]`` if ``cfg.global_clip``.

        Final, because the exported observation layout describes this transform by its config alone.

        Args:
            obs: Observation after noise
            clip_limit: The manager's ``clip_observations``

        Returns:
            The observation the policy sees for this term

        Raises:
            ValueError: If ``cfg.global_clip`` is set on a term that is not 2-D float, since clipping
                would convert it to float and saturate it (e.g. uint8 RGB, the depth ``+inf`` sentinel).
        """
        # Skip the multiply at the default 1.0 so a non-float term (e.g. uint8 RGB) keeps its dtype.
        if self.cfg.scale != 1.0:
            obs = obs * self.cfg.scale
        if self.cfg.clip is not None:
            obs = obs.clip(self.cfg.clip[0], self.cfg.clip[1])
        if self.cfg.global_clip:
            if not (obs.is_floating_point() and obs.ndim == 2):
                raise ValueError(
                    f"Observation term {type(self).__name__} ({self.cfg.func}) has global_clip set but returns "
                    f"{obs.dtype} with per-env shape {tuple(obs.shape[1:])}; the global clip only applies to 2-D "
                    f"float observations. Set global_clip=False for image and non-float terms."
                )
            obs = obs.clip(-clip_limit, clip_limit)
        return obs

    @final
    def observe(
        self,
        env: Any,
        env_ids: torch.Tensor | None = None,
        *,
        with_noise: bool,
        generator: torch.Generator | None,
        clip_limit: float,
    ) -> torch.Tensor:
        """Read, add noise if ``with_noise``, and postprocess the observation of ``env_ids``.

        Args:
            env: Environment instance
            env_ids: Environment IDs to compute and advance. If None, all environments.
            with_noise: Whether to apply :meth:`generate_noise` (the group's ``enable_noise``)
            generator: Random generator for the noise
            clip_limit: The manager's ``clip_observations``

        Returns:
            A copy, since the term may return a view of state that ``reset`` overwrites
        """
        raw = self._checked_read(env, env_ids)
        obs = self.generate_noise(raw, generator) if with_noise else raw
        out = self.postprocess(obs, clip_limit)
        # Scale, clip and noise already produce a new tensor; copy only when none of them ran.
        return out.clone() if out is raw else out

    @final
    def shape(self, env: Any) -> tuple[int, ...]:
        """Per-env output shape, measured by reading zero envs.

        Args:
            env: Environment instance
        """
        no_envs = torch.zeros(0, dtype=torch.long, device=env.device)
        return tuple(self._checked_read(env, no_envs).shape[1:])

    @final
    def _checked_read(self, env: Any, env_ids: torch.Tensor | None) -> torch.Tensor:
        """:meth:`read` with the config's params, raising if a row subset returns the wrong row count."""
        obs = self.read(env, env_ids=env_ids, **self.cfg.params)
        if not isinstance(obs, torch.Tensor):
            # Simulator state views (e.g. MuJoCo's) are tensor-like but not tensors; copy into one.
            obs = obs.clone()
        if env_ids is not None and obs.shape[0] != len(env_ids):
            raise ValueError(
                f"Observation term {type(self).__name__} ({self.cfg.func}) returned {obs.shape[0]} rows for "
                f"{len(env_ids)} env_ids; ObservationTermBase.read must compute and advance only env_ids."
            )
        return obs


class FunctionTerm(ObservationTermBase):
    """A stateless observation term defined by a plain function ``func(env, **params)``."""

    def __init__(self, cfg: ObsTermCfg, env: Any, func: Callable[..., torch.Tensor]):
        super().__init__(cfg, env)
        self.func = func

    def read(self, env: Any, env_ids: torch.Tensor | None = None, **kwargs: Any) -> torch.Tensor:
        obs = self.func(env, **kwargs)
        return obs if env_ids is None else obs[env_ids]
