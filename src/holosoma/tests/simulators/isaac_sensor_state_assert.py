"""Verify contact-force histories and height-scanner hits in each environment."""

from __future__ import annotations

import argparse
import sys

if sys.path and sys.path[0].endswith("tests/simulators"):
    sys.path.pop(0)

from tests.simulators._sim_harness import run_and_hard_exit, step, steps_for_seconds
from tests.simulators.static_move_assert import _build


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulator", required=True)
    args = parser.parse_args()
    sim, _ = _build(args.simulator, 4)

    import torch

    from holosoma.simulator.isaacsim.proxy_array import as_tensor

    peak = torch.zeros(4, device=sim.sim_device)
    for _ in range(steps_for_seconds(sim, 1.5)):
        step(sim, 1, render=False)
        forces = sim.contact_forces
        assert forces.shape == (4, sim.num_bodies, 3)
        assert torch.isfinite(forces).all()
        peak = torch.maximum(peak, forces.norm(dim=-1).max(dim=-1).values)
    assert (peak > 10).all(), f"No contact response in some environments: {peak.tolist()}"
    assert torch.isfinite(sim.contact_forces_history).all()

    hits = as_tensor(sim.scene.sensors["height_scanner"].data.ray_hits_w)
    assert hits.shape[0] == 4 and hits.shape[-1] == 3, hits.shape
    assert torch.isfinite(hits).all(), hits
    assert hits[..., 2].abs().max() < 0.002, hits
    print(f"PASS: contact forces/history and height hits in four environments; peak forces={peak.tolist()}")
    return 0


if __name__ == "__main__":
    run_and_hard_exit(main)
