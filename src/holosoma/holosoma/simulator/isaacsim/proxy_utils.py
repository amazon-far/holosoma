from __future__ import annotations

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

    Attributes
    ----------
    tensor_wxyz : torch.Tensor
        Original tensor with quaternions in wxyz format.
    tensor_xyzw : torch.Tensor
        Converted tensor with quaternions in xyzw format.
    """

    def __init__(self, tensor_wxyz: torch.Tensor):
        self.reset(tensor_wxyz)

    def reset(self, tensor_wxyz: torch.Tensor):
        self.tensor_wxyz = tensor_wxyz
        self.tensor_xyzw = fullstate_wxyz_to_xyzw(tensor_wxyz)

    def __getitem__(self, index):
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
        return self.tensor_xyzw[index]

    def __setitem__(self, index, value_xyzw):
        """Set tensor values from xyzw quaternion format.

        Parameters
        ----------
        index : int, slice, or tensor
            Index for tensor access.
        value_xyzw : torch.Tensor
            Values to set with quaternions in xyzw format.
        """
        self.tensor_xyzw[index] = value_xyzw
        self.tensor_wxyz = fullstate_xyzw_to_wxyz(self.tensor_xyzw)

    def _get_wxyz(self, env_ids=None):
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
        if env_ids is None:
            return self.tensor_wxyz
        return self.tensor_wxyz[env_ids]
