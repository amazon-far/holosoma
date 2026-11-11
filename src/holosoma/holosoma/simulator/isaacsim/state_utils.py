"""State utilities for IsaacSim"""

import torch


def fullstate_wxyz_to_xyzw(tensor_wxyz: torch.Tensor) -> torch.Tensor:
    """Convert quaternion from wxyz to xyzw format.

    Parameters
    ----------
    tensor_wxyz : torch.Tensor
        Tensor with quaternions in wxyz format at indices 3:7.
        Shape: [..., 13] where quaternion is at [..., 3:7]

    Returns
    -------
    torch.Tensor
        Tensor with quaternions converted to xyzw format.

    Examples
    --------
    >>> wxyz_state = torch.tensor([[1.0, 2.0, 3.0, 0.707, 0.0, 0.0, 0.707, 0.1, 0.2, 0.3, 0.01, 0.02, 0.03]])
    >>> xyzw_state = fullstate_wxyz_to_xyzw(wxyz_state)
    >>> # Quaternion at indices 3:7 is now [0.0, 0.0, 0.707, 0.707] (xyzw format)
    """
    converted = tensor_wxyz.clone()
    converted[:, 3:6] = tensor_wxyz[:, 4:7]  # [w,x,y,z] -> [x,y,z,w]
    converted[:, 6] = tensor_wxyz[:, 3]
    return converted


def fullstate_xyzw_to_wxyz(tensor_xyzw: torch.Tensor) -> torch.Tensor:
    """Convert quaternion from xyzw to wxyz format.

    Parameters
    ----------
    tensor_xyzw : torch.Tensor
        Tensor with quaternions in xyzw format at indices 3:7.
        Shape: [..., 13] where quaternion is at [..., 3:7]

    Returns
    -------
    torch.Tensor
        Tensor with quaternions converted to wxyz format.

    Examples
    --------
    >>> xyzw_state = torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.707, 0.707, 0.1, 0.2, 0.3, 0.01, 0.02, 0.03]])
    >>> wxyz_state = fullstate_xyzw_to_wxyz(xyzw_state)
    >>> # Quaternion at indices 3:7 is now [0.707, 0.0, 0.0, 0.707] (wxyz format)
    """
    converted = tensor_xyzw.clone()
    converted[:, 3] = tensor_xyzw[:, 6]  # [x,y,z,w] -> [w,x,y,z]
    converted[:, 4:7] = tensor_xyzw[:, 3:6]
    return converted
