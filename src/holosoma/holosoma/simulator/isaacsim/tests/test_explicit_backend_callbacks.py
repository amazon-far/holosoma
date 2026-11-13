"""SDK event callbacks retain the frontend's explicit API-family dependency."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch
import warp as wp

from holosoma.simulator.isaacsim.backends.isaaclab2 import IsaacLab2Backend
from holosoma.simulator.isaacsim.backends.isaaclab3 import IsaacLab3Backend
from holosoma.utils.sampler import STAGE_SETUP, TermSampler

pytestmark = pytest.mark.no_sim


@pytest.mark.parametrize("family", [2, 3])
def test_mass_event_uses_the_supplied_frontend_backend_for_sparse_writes(
    monkeypatch: pytest.MonkeyPatch, family: int
) -> None:
    """Run the real event and sampler, mocking only SDK imports and asset buffers."""
    wp.init()
    assets = ModuleType("isaaclab.assets")
    assets.Articulation = object  # type: ignore[attr-defined]
    assets.RigidObject = object  # type: ignore[attr-defined]
    managers = ModuleType("isaaclab.managers")
    managers.SceneEntityCfg = SimpleNamespace  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "isaaclab.assets", assets)
    monkeypatch.setitem(sys.modules, "isaaclab.managers", managers)
    # Load the unchanged production module under an isolated name so its external
    # SDK import substitutes cannot leak into another test's normal module cache.
    path = Path(__file__).resolve().parents[1] / "events.py"
    spec = importlib.util.spec_from_file_location("_explicit_backend_events", path)
    assert spec is not None and spec.loader is not None
    events = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(events)

    backend = IsaacLab3Backend() if family == 3 else IsaacLab2Backend()
    monkeypatch.setenv("HOLOSOMA_ISAACSIM_BACKEND", "isaaclab2" if family == 3 else "isaaclab3")
    physical_masses = torch.tensor([[2.0, 3.0], [5.0, 7.0], [11.0, 13.0], [17.0, 19.0]])
    expected = torch.tensor([[2.0, 3.0], [5.0, 9.5], [11.0, 13.0], [17.0, 21.5]])
    received_selectors: list[list[int]] = []

    def legacy_setter(values: torch.Tensor, env_ids: torch.Tensor) -> None:
        assert isinstance(env_ids, torch.Tensor)
        assert env_ids.device.type == "cpu"
        received_selectors.append(env_ids.tolist())
        physical_masses[env_ids] = values[env_ids]

    def native_setter(*, masses: torch.Tensor, env_ids: wp.array, full_data: bool) -> None:
        assert isinstance(env_ids, wp.array)
        assert env_ids.dtype == wp.int64
        assert str(env_ids.device) == "cpu"
        assert full_data is True
        indices = wp.to_torch(env_ids)
        received_selectors.append(indices.tolist())
        physical_masses[indices] = masses[indices]

    asset = SimpleNamespace(
        device="cpu",
        num_bodies=2,
        data=SimpleNamespace(body_mass=SimpleNamespace(torch=physical_masses)),
        root_physx_view=SimpleNamespace(get_masses=lambda: physical_masses.clone(), set_masses=legacy_setter),
        set_masses_index=native_setter,
    )

    class Scene(dict[str, Any]):
        num_envs = 4

    env = SimpleNamespace(backend=backend, scene=Scene(robot=asset))
    sampler = TermSampler.bind(base_seed=23, term_name="explicit-backend-mass", stage=STAGE_SETUP)
    events.randomize_rigid_body_mass(
        env,
        torch.tensor([3, 1]),
        SimpleNamespace(name="robot", body_ids=[1]),
        mass=[2.5, 2.5],
        operation="add",
        sampler=sampler,
        recompute_inertia=False,
    )
    assert received_selectors == [[3, 1]]
    torch.testing.assert_close(physical_masses, expected, rtol=0, atol=0)
