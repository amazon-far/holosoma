"""Root-state proxy for the Isaac Lab 3.x lane, where no quaternion reordering is needed.

Isaac Lab 2.3.2 reported and accepted root states with the quaternion **w-first**, while
holosoma's cross-backend contract for ``robot_root_states`` is **w-last** (``[pos(3),
quat_xyzw(4), lin_vel(3), ang_vel(3)]``). The 5.1 lane therefore reorders on every read and
write, which is what :class:`holosoma.simulator.isaacsim.proxy_utils.RootStatesProxy` does.

Isaac Lab 3.0 flipped its own convention to w-last throughout -- ``root_state_w`` and
``root_link_quat_w`` document ``(x, y, z, w)``, and ``write_root_pose_to_sim_index`` expects the
same -- so in this lane the simulator's order already *is* holosoma's order. Reusing the shared
proxy here would reorder data that is already correct, silently rotating every robot root
orientation, so this lane needs its own pass-through instead.

The external interface is deliberately identical: ``__getitem__`` / ``__setitem__`` speak w-last
to callers like ``managers/command/terms/wbt.py`` and ``managers/observation/terms/wbt.py``,
which are backend-agnostic and must not be able to tell the two lanes apart.

One subtlety that is load-bearing: the buffer is **cloned**. The 5.1 proxy clones as a side
effect of reordering, which makes a write to the proxy a write to holosoma's own copy that only
reaches the simulator on an explicit flush (``set_actor_root_state_tensor_robots``). Callers rely
on that -- ``wbt.py`` mutates several slices and flushes once afterwards. Holding the simulator's
tensor directly would make each slice assignment take effect immediately, so the clone is kept
even though no reordering requires it.
"""

from __future__ import annotations

import torch


class RootStatesProxy:
    """Holosoma-order (w-last) view over an Isaac Lab 3.x root-state tensor.

    Parameters
    ----------
    tensor_xyzw : torch.Tensor
        Root states as Isaac Lab 3.x reports them, ``(num_instances, 13)`` with the quaternion
        at indices ``3:7`` in ``(x, y, z, w)`` order -- already holosoma's convention.

    Attributes
    ----------
    tensor_xyzw : torch.Tensor
        The cloned, w-last buffer that reads and writes go through.
    """

    def __init__(self, tensor_xyzw: torch.Tensor) -> None:
        self.reset(tensor_xyzw)

    def reset(self, tensor_xyzw: torch.Tensor) -> None:
        """Rebind to a fresh simulator tensor, taking a private copy of it."""
        self.tensor_xyzw = tensor_xyzw.clone()

    def __getitem__(self, index: int | slice | torch.Tensor) -> torch.Tensor:
        """Read root states in holosoma's w-last order."""
        return self.tensor_xyzw[index]

    def __setitem__(self, index: int | slice | torch.Tensor, value_xyzw: torch.Tensor) -> None:
        """Stage a w-last write. Not visible to the simulator until the caller flushes."""
        self.tensor_xyzw[index] = value_xyzw

    def _get_sim_order(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """The buffer in the order Isaac Lab 3.x's write APIs expect, optionally sliced.

        Named for the role rather than the layout, unlike the 5.1 lane's ``_get_wxyz``: here the
        simulator's order and holosoma's coincide, so a layout-based name would suggest a
        conversion that does not happen.
        """
        if env_ids is None:
            return self.tensor_xyzw
        return self.tensor_xyzw[env_ids]
