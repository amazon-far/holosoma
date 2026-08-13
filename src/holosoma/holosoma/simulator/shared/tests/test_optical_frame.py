"""Unit test for the IsaacGym camera optical-axis change-of-basis (pure torch, no simulator).

The invariant: composing the optical mount orientation with the optical->IsaacGym basis and
applying it to IsaacGym's native axes (+X fwd / +Z up) yields the SAME world forward AND up as
applying the mount directly to the optical axes (-Z fwd / +Y up), which is what MuJoCo/IsaacSim
do natively. The basis is defined in ``simulator/isaacgym/isaacgym.py``; it is duplicated here as a
literal so the convention can be checked without importing the IsaacGym SDK.

The cases include roll-bearing mounts (upright forward, pitched-down): a forward-only
basis check passes a pure-pitch-about-Y mount but misses an optical-axis roll (up -> -Y instead of
+Y), so up is asserted explicitly.
"""

from __future__ import annotations

import pytest
import torch

from holosoma.utils.rotations import quat_apply, quat_mul

pytestmark = pytest.mark.no_sim

# Optical->IsaacGym change-of-basis (xyzw); mirrors _CANONICAL_TO_ISAACGYM_XYZW in the backend.
_CANONICAL_TO_ISAACGYM_XYZW = (0.5, -0.5, -0.5, -0.5)

# Optical-frame axes (-Z fwd / +Y up) and IsaacGym's native camera axes (+X fwd / +Z up).
_CANON_FWD = torch.tensor([0.0, 0.0, -1.0])  # -Z forward
_CANON_UP = torch.tensor([0.0, 1.0, 0.0])  # +Y up
_IG_FWD = torch.tensor([1.0, 0.0, 0.0])  # +X forward
_IG_UP = torch.tensor([0.0, 0.0, 1.0])  # +Z up

# Mount orientations (w, x, y, z). Includes the production g1 presets (which carry real roll) plus a
# pure-pitch case (forward-only checks cannot detect roll on it).
_MOUNTS_WXYZ = {
    "identity": [1.0, 0.0, 0.0, 0.0],
    "look_forward_upright": [0.5, 0.5, -0.5, -0.5],  # g1 head_cam: fwd +X, up +Z
    "look_forward_down": [0.61237244, 0.35355339, -0.35355339, -0.61237244],  # g1 wrist grasp view
    "pure_pitch_-90_about_y": [0.70710678, 0.0, -0.70710678, 0.0],
}


@pytest.mark.parametrize("name", list(_MOUNTS_WXYZ))
def test_isaacgym_basis_matches_canonical_forward_and_up(name: str) -> None:
    """IsaacGym's converted mount must yield the SAME world forward AND up as the optical frame."""
    wxyz = torch.tensor([_MOUNTS_WXYZ[name]])
    xyzw = wxyz[:, [1, 2, 3, 0]]

    # MuJoCo/IsaacSim: mount applied directly to the optical axes.
    canon_fwd = quat_apply(xyzw, _CANON_FWD, w_last=True)
    canon_up = quat_apply(xyzw, _CANON_UP, w_last=True)
    # IsaacGym: composed mount applied to IsaacGym's native axes.
    basis = torch.tensor([_CANONICAL_TO_ISAACGYM_XYZW])
    native = quat_mul(xyzw, basis, w_last=True)
    ig_fwd = quat_apply(native, _IG_FWD, w_last=True)
    ig_up = quat_apply(native, _IG_UP, w_last=True)

    torch.testing.assert_close(ig_fwd, canon_fwd, atol=1e-5, rtol=0, msg=f"{name}: forward axis differs")
    # Up must also match: a forward-only basis check passes but leaves an optical-axis roll.
    torch.testing.assert_close(ig_up, canon_up, atol=1e-5, rtol=0, msg=f"{name}: up axis differs (optical-axis roll)")
