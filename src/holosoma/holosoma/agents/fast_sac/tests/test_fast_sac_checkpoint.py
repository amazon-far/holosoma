"""FastSAC checkpoints hold only primitives and tensors, so a ``weights_only`` ``torch.load`` reads them.

``eval_utils._load_config_from_checkpoint`` uses torch's default loader, which on torch >= 2.6 refuses pickled
Python objects; the ``FastSACConfig`` written by ``save_params`` nests ``JointSymmetryConfig``, which a shallow
``vars()`` used to leave as an object (CI IsaacSim e2e eval failed on exactly that).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from holosoma.agents.fast_sac.fast_sac_utils import save_params
from holosoma.config_types.algo import FastSACConfig

pytestmark = pytest.mark.no_sim


def test_saved_checkpoint_loads_with_weights_only(tmp_path: Path) -> None:
    actor, qnet, qnet_target = nn.Linear(2, 2), nn.Linear(2, 1), nn.Linear(2, 1)
    log_alpha = torch.zeros(1)
    path = tmp_path / "model.pt"

    save_params(
        3,
        actor,
        qnet,
        qnet_target,
        log_alpha,
        nn.Identity(),
        nn.Identity(),
        torch.optim.Adam(actor.parameters()),
        torch.optim.Adam(qnet.parameters()),
        torch.optim.Adam([log_alpha.requires_grad_()]),
        torch.amp.GradScaler("cpu", enabled=False),
        FastSACConfig(),
        str(path),
        metadata={"experiment_config": {"algo": "fast_sac"}},
    )

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    assert checkpoint["global_step"] == 3 and checkpoint["experiment_config"] == {"algo": "fast_sac"}
    assert isinstance(checkpoint["args"]["symmetry"], dict)  # nested config stored as primitives
    assert checkpoint["args"]["symmetry"]["left"] == ["#mirror_left"]
