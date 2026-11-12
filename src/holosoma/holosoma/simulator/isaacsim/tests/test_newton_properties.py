"""Newton adapters obey public setter shapes and preserve actual shape ownership."""

from __future__ import annotations

import math
import sys
from enum import IntFlag
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch
import warp as wp

from holosoma.simulator.isaacsim.backends import newton_properties
from holosoma.simulator.isaacsim.backends.isaacsim6 import IsaacSim6Backend
from holosoma.config_types.simulator import IsaacSim6Config

pytestmark = pytest.mark.no_sim


@pytest.fixture(scope="module", autouse=True)
def initialize_warp() -> None:
    wp.init()


@pytest.mark.parametrize(
    ("name", "shape", "expected"),
    [
        ("masses", (4, 2), [[6.0, 7.0], [2.0, 3.0]]),
        ("coms", (4, 2, 7), [[[42.0, 43.0, 44.0], [49.0, 50.0, 51.0]], [[14.0, 15.0, 16.0], [21.0, 22.0, 23.0]]]),
        (
            "inertias",
            (4, 1, 9),
            [
                [[27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0]],
                [[9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]],
            ],
        ),
        (
            "inertias",
            (4, 9),
            [
                [[27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0]],
                [[9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]],
            ],
        ),
    ],
)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_body_writes_pass_partial_rows_without_physx_keywords(
    name: str, shape: tuple[int, ...], expected: list[Any], index_dtype: torch.dtype
) -> None:
    values = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)
    original = values.clone()
    selector = torch.tensor([3, 99, 1, 99], dtype=index_dtype)[::2]
    calls: list[torch.Tensor] = []

    def setter(*, env_ids: wp.array, **properties: torch.Tensor) -> None:
        assert isinstance(env_ids, wp.array)
        assert env_ids.dtype == wp.int64
        assert str(env_ids.device) == "cpu"
        assert wp.to_torch(env_ids).tolist() == [3, 1]
        assert set(properties) == {name}, "Newton public setters do not accept full_data"
        selected = properties[name]
        assert selected.device.type == "cpu"
        assert selected.dtype == torch.float32
        torch.testing.assert_close(selected, torch.tensor(expected), rtol=0, atol=0)
        calls.append(selected)

    asset = SimpleNamespace(device="cpu", **{f"set_{name}_index": setter})
    backend = IsaacSim6Backend(IsaacSim6Config(physics_backend="newton"))
    backend.write_physics_property(asset, name, values, selector)
    assert len(calls) == 1
    torch.testing.assert_close(values, original, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["articulation", "rigid_object"])
def test_materials_expose_two_real_channels_and_write_only_selected_rows(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """Interleaved scene shapes stay unchanged for articulation and rigid-object views."""
    friction = wp.array(
        [1.0, 101.0, 2.0, 3.0, 4.0, 102.0, 5.0, 6.0, 7.0, 103.0, 8.0, 9.0, 10.0, 104.0, 11.0, 12.0],
        dtype=wp.float32,
        device="cpu",
    )
    restitution = wp.array(
        [0.01, 0.91, 0.02, 0.03, 0.04, 0.92, 0.05, 0.06, 0.07, 0.93, 0.08, 0.09, 0.10, 0.94, 0.11, 0.12],
        dtype=wp.float32,
        device="cpu",
    )
    notifications: list[object] = []
    body_count = 2 if kind == "articulation" else 1
    body_names = ["pelvis", "hand"] if kind == "articulation" else ["freebox"]
    body_shapes = {}
    for row in range(4):
        body = row * body_count
        body_shapes[body] = [4 * row] if kind == "articulation" else [4 * row, 4 * row + 2, 4 * row + 3]
        if kind == "articulation":
            body_shapes[body + 1] = [4 * row + 2, 4 * row + 3]
    model = SimpleNamespace(
        shape_material_mu=friction,
        shape_material_restitution=restitution,
        articulation_start=wp.array([row * body_count for row in range(4)], dtype=wp.int32, device="cpu"),
        articulation_end=wp.array([(row + 1) * body_count for row in range(4)], dtype=wp.int32, device="cpu"),
        joint_child=wp.array(list(range(4 * body_count)), dtype=wp.int32, device="cpu"),
        body_label=[f"actor/{name}" for _ in range(4) for name in body_names],
        body_shapes=body_shapes,
    )
    manager = SimpleNamespace(get_model=lambda: model, add_model_change=notifications.append)
    native_sim = ModuleType("isaaclab.sim")
    native_sim.SimulationContext = SimpleNamespace(  # type: ignore[attr-defined]
        instance=lambda: SimpleNamespace(physics_manager=manager)
    )

    class ModelFlags(IntFlag):
        SHAPE_PROPERTIES = 8

    native_newton = ModuleType("newton")
    native_newton.ModelFlags = ModelFlags  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "isaaclab.sim", native_sim)
    monkeypatch.setitem(sys.modules, "newton", native_newton)

    view = SimpleNamespace(
        articulation_ids=wp.array([[0], [1], [2], [3]], dtype=wp.int32, device="cpu"),
        link_names=body_names,
        shape_count=3,
    )
    asset = SimpleNamespace(root_view=view, num_instances=4)
    full_values = newton_properties.read_materials(asset)
    expected_read = torch.tensor(
        [
            [[1.0, 0.01], [2.0, 0.02], [3.0, 0.03]],
            [[4.0, 0.04], [5.0, 0.05], [6.0, 0.06]],
            [[7.0, 0.07], [8.0, 0.08], [9.0, 0.09]],
            [[10.0, 0.10], [11.0, 0.11], [12.0, 0.12]],
        ]
    )
    assert tuple(full_values.shape) == (4, 3, 2)
    assert full_values.device.type == "cpu"
    torch.testing.assert_close(full_values, expected_read, rtol=0, atol=0)
    full_values[3] = torch.tensor([[30.0, 0.30], [31.0, 0.31], [32.0, 0.32]])
    full_values[1] = torch.tensor([[20.0, 0.20], [21.0, 0.21], [22.0, 0.22]])
    # Reading creates an owned buffer; only an explicit write changes the model.
    torch.testing.assert_close(
        wp.to_torch(friction),
        torch.tensor([1.0, 101.0, 2.0, 3.0, 4.0, 102.0, 5.0, 6.0, 7.0, 103.0, 8.0, 9.0, 10.0, 104.0, 11.0, 12.0]),
        rtol=0,
        atol=0,
    )

    selector = torch.tensor([3, 99, 1, 99], dtype=torch.int32)[::2]
    newton_properties.write_materials(asset, full_values, selector)
    torch.testing.assert_close(
        wp.to_torch(friction),
        torch.tensor([1.0, 101.0, 2.0, 3.0, 20.0, 102.0, 21.0, 22.0, 7.0, 103.0, 8.0, 9.0, 30.0, 104.0, 31.0, 32.0]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        wp.to_torch(restitution),
        torch.tensor([0.01, 0.91, 0.02, 0.03, 0.20, 0.92, 0.21, 0.22, 0.07, 0.93, 0.08, 0.09, 0.30, 0.94, 0.31, 0.32]),
        rtol=0,
        atol=0,
    )
    assert notifications == [ModelFlags.SHAPE_PROPERTIES]
    with pytest.raises(ValueError, match="must have shape"):
        newton_properties.write_materials(asset, torch.zeros(4, 3, 3), selector)
    assert notifications == [ModelFlags.SHAPE_PROPERTIES]
