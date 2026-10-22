from __future__ import annotations

from collections.abc import Callable

import torch

from holosoma.simulator.isaacsim.state_utils import fullstate_wxyz_to_xyzw, fullstate_xyzw_to_wxyz


class RootStatesProxy:
    """Wrapper for root states tensor with quaternion format conversion.

    This proxy handles the conversion between xyzw and wxyz quaternion formats
    for consistency between BaseTask/LeggedRobotBase (which uses xyzw) and
    IsaacSim (which uses wxyz).

    The __getitem__ and __setitem__ methods provide access in xyzw format for
    BaseTask/LeggedRobotBase, while tensor_wxyz is used for IsaacSim interfacing.

    Parameters
    ----------
    tensor_wxyz : torch.Tensor
        Root states tensor with quaternions in wxyz format.
    version : Callable[[], int], optional
        Returns the backend's state version. When it differs from the version the cached
        tensors were taken at, the next access re-reads ``latest()``. Writes through the
        proxy stay authoritative until the version advances.
    latest : Callable[[], torch.Tensor], optional
        Returns the backend's current wxyz root states. Required with ``version``.

    Attributes
    ----------
    tensor_wxyz : torch.Tensor
        Original tensor with quaternions in wxyz format.
    tensor_xyzw : torch.Tensor
        Converted tensor with quaternions in xyzw format.
    """

    def __init__(
        self,
        tensor_wxyz: torch.Tensor,
        *,
        version: Callable[[], int] | None = None,
        latest: Callable[[], torch.Tensor] | None = None,
    ) -> None:
        if (version is None) != (latest is None):
            raise ValueError("RootStatesProxy needs both version and latest, or neither.")
        self._version = version
        self._latest = latest
        self.reset(tensor_wxyz)

    def reset(self, tensor_wxyz: torch.Tensor) -> None:
        self.tensor_wxyz = tensor_wxyz
        self.tensor_xyzw = fullstate_wxyz_to_xyzw(tensor_wxyz)
        self._cached_version = None if self._version is None else self._version()

    def invalidate(self) -> None:
        """Re-read on next access, e.g. after a root write that bypassed this proxy."""
        self._cached_version = None

    def _ensure_current(self) -> None:
        if self._version is not None and self._cached_version != self._version():
            assert self._latest is not None
            self.reset(self._latest())

    def __getitem__(self, index: int | slice | torch.Tensor) -> torch.Tensor:
        """Get tensor values in xyzw quaternion format.

        Parameters
        ----------
        index : int, slice, or tensor
            Index for tensor access.

        Returns
        -------
        torch.Tensor
            Tensor values with quaternions in xyzw format.
        """
        self._ensure_current()
        return self.tensor_xyzw[index]

    def __setitem__(self, index: int | slice | torch.Tensor, value_xyzw: torch.Tensor) -> None:
        """Set tensor values from xyzw quaternion format.

        Parameters
        ----------
        index : int, slice, or tensor
            Index for tensor access.
        value_xyzw : torch.Tensor
            Values to set with quaternions in xyzw format.
        """
        self._ensure_current()
        self.tensor_xyzw[index] = value_xyzw
        self.tensor_wxyz = fullstate_xyzw_to_wxyz(self.tensor_xyzw)

    def _get_wxyz(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Get tensor in wxyz quaternion format for IsaacSim interfacing.

        Parameters
        ----------
        env_ids : torch.Tensor, optional
            Environment IDs to select, by default None (returns all).

        Returns
        -------
        torch.Tensor
            Tensor with quaternions in wxyz format.
        """
        self._ensure_current()
        if env_ids is None:
            return self.tensor_wxyz
        return self.tensor_wxyz[env_ids]
