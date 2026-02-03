from __future__ import annotations

from typing import Tuple

import numpy as np

Pair = Tuple[str, str]


def _mesh_local_vf(model, geom_id):
    """Return local vertices and faces for a MuJoCo mesh geom."""
    mesh_id = int(model.geom_dataid[geom_id])  # Note: sometime geom does not have mesh, mesh_id will be -1

    v0, nv = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
    f0, nf = int(model.mesh_faceadr[mesh_id]), int(model.mesh_facenum[mesh_id])

    V = model.mesh_vert[v0 : v0 + nv].astype(np.float64, copy=True)

    F = model.mesh_face[f0 : f0 + nf].astype(np.int32, copy=True)

    return V, F


def _to_world(v_local, data, geom_id):
    """Transform local vertices to world using geom pose."""
    R = data.geom_xmat[geom_id].reshape(3, 3)
    t = data.geom_xpos[geom_id]

    return v_local @ R.T + t


def _world_mesh_from_geom(model, data, geom_id, geom_name):
    V_local, F = _mesh_local_vf(model, geom_id)

    V_world = _to_world(V_local, data, geom_id)

    return V_world, F
