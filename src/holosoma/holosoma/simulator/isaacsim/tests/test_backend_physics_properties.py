"""Property writer contracts at the external Isaac Lab and PhysX boundaries."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
import warp as wp

from holosoma.simulator.isaacsim.backends.isaacsim5 import IsaacSim5Backend
from holosoma.simulator.isaacsim.backends.isaacsim6 import IsaacSim6Backend

pytestmark = pytest.mark.no_sim


@pytest.fixture(scope="module", autouse=True)
def initialize_warp() -> None:
    wp.init()


@pytest.mark.parametrize(
    ("name", "input_shape", "native_shape"),
    [
        ("masses", (4, 2), (4, 2)),
        ("coms", (4, 2, 7), (4, 2, 7)),
        ("inertias", (4, 2, 9), (4, 2, 9)),
        ("inertias", (4, 9), (4, 1, 9)),
    ],
)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_native_property_setters_receive_warp_selectors_and_full_values(
    name: str, input_shape: tuple[int, ...], native_shape: tuple[int, ...], index_dtype: torch.dtype
) -> None:
    """The native public setter gets typed indices, preserving sparse row order and payload."""
    values = torch.arange(math.prod(input_shape), dtype=torch.float32).reshape(input_shape)
    # A non-contiguous selector exercises conversion without hiding a row-order mistake.
    env_ids = torch.tensor([3, 99, 1, 99], dtype=index_dtype)[::2]
    calls: list[dict[str, torch.Tensor]] = []

    def setter(*, env_ids: wp.array, full_data: bool, **properties: torch.Tensor) -> None:
        assert isinstance(env_ids, wp.array), "native indexed setters require an explicit Warp selector"
        assert env_ids.dtype == wp.int64
        assert str(env_ids.device) == "cpu"
        assert wp.to_torch(env_ids).tolist() == [3, 1]
        assert full_data is True
        assert set(properties) == {name}
        supplied = properties[name]
        assert isinstance(supplied, torch.Tensor)
        assert supplied.device == torch.device("cpu")
        assert tuple(supplied.shape) == native_shape
        torch.testing.assert_close(supplied.reshape(input_shape), values, rtol=0, atol=0)
        calls.append(properties)

    asset = SimpleNamespace(device="cpu", **{f"set_{name}_index": setter})
    IsaacSim6Backend().write_physics_property(asset, name, values, env_ids)
    assert len(calls) == 1


def test_native_raw_material_writer_keeps_cpu_int32_selectors() -> None:
    """The raw PhysX material setter keeps its separate CPU index contract."""
    values = torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3)
    env_ids = torch.tensor([3, 99, 1, 99], dtype=torch.int64)[::2]
    calls: list[tuple[wp.array, wp.array]] = []

    def setter(materials: wp.array, indices: wp.array) -> None:
        assert isinstance(materials, wp.array)
        assert isinstance(indices, wp.array)
        assert indices.dtype == wp.int32
        assert str(indices.device) == "cpu"
        assert wp.to_torch(indices).tolist() == [3, 1]
        torch.testing.assert_close(wp.to_torch(materials), values, rtol=0, atol=0)
        calls.append((materials, indices))

    asset = SimpleNamespace(root_view=SimpleNamespace(set_material_properties=setter))
    IsaacSim6Backend().write_physics_property(asset, "material_properties", values, env_ids)
    assert len(calls) == 1


@pytest.mark.parametrize("name", ["masses", "coms", "inertias", "material_properties"])
def test_legacy_property_writer_preserves_raw_torch_arguments(name: str) -> None:
    """Lab 2 still receives the original Torch values and CPU selectors."""
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    env_ids = torch.tensor([3, 1], dtype=torch.int64)
    calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def setter(supplied: torch.Tensor, indices: torch.Tensor) -> None:
        assert supplied is values
        assert indices is env_ids
        calls.append((supplied, indices))

    asset = SimpleNamespace(root_physx_view=SimpleNamespace(**{f"set_{name}": setter}))
    IsaacSim5Backend().write_physics_property(asset, name, values, env_ids)
    assert len(calls) == 1


def test_native_startup_restore_writes_world_states_without_mutating_defaults() -> None:
    """Startup integration is discarded using configured poses, velocities and joint state."""
    defaults = {
        "root_pose": torch.tensor([[1.0, 2.0, 0.6, 0.0, 0.0, 1.0, 0.0], [-2.0, 3.0, 0.8, 0.0, 1.0, 0.0, 0.0]]),
        "root_velocity": torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6]]),
        "position": torch.tensor([[0.2, -0.4], [0.3, -0.5]]),
        "velocity": torch.tensor([[0.7, -0.8], [0.9, -1.0]]),
    }
    original_defaults = {name: value.clone() for name, value in defaults.items()}
    origins = torch.tensor([[10.0, -20.0, 0.5], [-30.0, 40.0, -0.25]])
    original_origins = origins.clone()
    received: dict[str, torch.Tensor] = {}
    calls: list[str] = []

    def write(**state: torch.Tensor) -> None:
        assert len(state) == 1, "all environments use the SDK's default index selection"
        name, value = next(iter(state.items()))
        received[name] = value
        calls.append(name)

    def reset() -> None:
        assert calls == ["root_pose", "root_velocity", "position", "velocity"]
        calls.append("reset")

    asset = SimpleNamespace(
        data=SimpleNamespace(
            default_root_pose=SimpleNamespace(torch=defaults["root_pose"]),
            default_root_vel=SimpleNamespace(torch=defaults["root_velocity"]),
            default_joint_pos=SimpleNamespace(torch=defaults["position"]),
            default_joint_vel=SimpleNamespace(torch=defaults["velocity"]),
        ),
        write_root_pose_to_sim_index=write,
        write_root_velocity_to_sim_index=write,
        write_joint_position_to_sim_index=write,
        write_joint_velocity_to_sim_index=write,
        reset=reset,
    )
    IsaacSim6Backend().restore_initial_articulation_state(asset, origins)

    assert calls == ["root_pose", "root_velocity", "position", "velocity", "reset"]
    expected_pose = torch.tensor([[11.0, -18.0, 1.1, 0.0, 0.0, 1.0, 0.0], [-32.0, 43.0, 0.55, 0.0, 1.0, 0.0, 0.0]])
    torch.testing.assert_close(received["root_pose"], expected_pose, rtol=0, atol=0)
    for name in ("root_velocity", "position", "velocity"):
        torch.testing.assert_close(received[name], original_defaults[name], rtol=0, atol=0)
    # The external writer can retain its input; later changes must not corrupt configured defaults.
    for name, value in received.items():
        value.fill_(-99.0)
        torch.testing.assert_close(defaults[name], original_defaults[name], rtol=0, atol=0)
    torch.testing.assert_close(origins, original_origins, rtol=0, atol=0)
