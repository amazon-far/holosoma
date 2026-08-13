"""Backend-agnostic unified root-state view.

``UnifiedRootStatesView`` is a duck-typed proxy (NOT a ``torch.Tensor``) backing
``sim.all_root_states`` on every backend. It spans all actors — robot, scene
bodies, and free objects — indexable with the flat indices ``get_actor_indices``
returns, and routes ``proxy[indices]`` reads and ``proxy[indices] = states``
writes through the simulator's :meth:`get_actor_states_by_index` /
:meth:`set_actor_states_by_index`.

The instance is created once at ``prepare_sim``, so the identity check in
``set_actor_root_state_tensor`` (``root_states is self.all_root_states``) holds
for callers that pass ``sim.all_root_states`` wholesale; that call writes the
robot only and does not index this proxy.

Each row is the 13-vector ``[x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]``
in the WORLD frame, quaternion xyzw, both velocities world-frame.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator


class UnifiedRootStatesView:
    """Duck-typed ``[indices]`` view over all actors (robot + objects).

    Routes reads/writes to the backend's ``get_actor_states_by_index`` /
    ``set_actor_states_by_index``. Supports ``proxy[indices]`` and
    ``proxy[indices, column_slice]`` for both reads and writes, plus the
    tensor-like ``.shape`` / ``.device`` / ``.dtype`` / ``.clone()`` surface.

    Parameters
    ----------
    simulator : BaseSimulator
        The owning simulator. Must expose ``object_registry``,
        ``get_actor_states_by_index``, ``set_actor_states_by_index``,
        ``sim_device`` and ``num_envs``.
    """

    def __init__(self, simulator: BaseSimulator):
        self._sim = simulator

    def _split_key(self, key):
        """Split a ``[]`` key into (row_indices, column_slice).

        Non-tensor row indices (list / tuple / ndarray of ints) are coerced to a tensor on the
        simulator device, so ``proxy[[0, 1]]`` works on every backend.
        """
        indices, column_slice = key if isinstance(key, tuple) else (key, slice(None))
        if not isinstance(indices, torch.Tensor):
            indices = torch.tensor(indices, device=self._sim.sim_device)
        return indices, column_slice

    def __getitem__(self, key):
        """Read actor states for ``indices`` (optionally a column slice).

        ``proxy[indices]`` -> ``[len(indices), 13]``; ``proxy[indices, cols]``
        applies the column slice to the gathered states.
        """
        indices, column_slice = self._split_key(key)
        states = self._sim.get_actor_states_by_index(indices)  # [N, 13] world-frame, xyzw
        if column_slice == slice(None):
            return states
        return states[:, column_slice]

    def __setitem__(self, key, values):
        """Write actor states for ``indices`` (optionally a column slice).

        ``proxy[indices] = states`` writes the full 13-vector per actor. A
        partial-column write (``proxy[indices, cols] = vals``) reads the current
        state, overwrites the given columns, and writes it back, so untouched
        columns are preserved. Writes pass ``write_updates=False``; callers sync
        via :meth:`write_state_updates` (a no-op on backends with immediate writes).
        """
        indices, column_slice = self._split_key(key)
        if column_slice == slice(None):
            self._sim.set_actor_states_by_index(indices, values, write_updates=False)
            return
        current = self._sim.get_actor_states_by_index(indices)
        current[:, column_slice] = values
        self._sim.set_actor_states_by_index(indices, current, write_updates=False)

    @property
    def shape(self) -> torch.Size:
        """``[total_actors, 13]`` = ``len(objects) * num_envs`` rows."""
        registry = self._sim.object_registry
        return torch.Size([len(registry.objects) * self._sim.num_envs, 13])

    @property
    def device(self):
        return self._sim.sim_device

    @property
    def dtype(self):
        return torch.float32

    def clone(self) -> torch.Tensor:
        """Materialize all actor states (every actor, every env) into a tensor.

        Row order matches ``torch.arange(total_actors)`` through
        ``resolve_indices`` (the registry's interleaved layout), so this is
        identical to ``proxy[torch.arange(total_actors)]``.
        """
        registry = self._sim.object_registry
        total = len(registry.objects) * self._sim.num_envs
        if total == 0:
            return torch.empty(0, 13, device=self.device, dtype=self.dtype)
        indices = torch.arange(total, device=self.device, dtype=torch.long)
        return self._sim.get_actor_states_by_index(indices)
