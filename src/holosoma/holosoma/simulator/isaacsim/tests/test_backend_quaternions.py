"""Quaternion boundary conversion preserves literal values and runtime independence."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from types import ModuleType

import numpy as np
import numpy.typing as npt
import pytest
import torch

from holosoma.simulator.isaacsim.backends.isaacsim5 import IsaacSim5Backend
from holosoma.simulator.isaacsim.backends.isaacsim6 import IsaacSim6Backend
from holosoma.simulator.isaacsim.state_utils import fullstate_wxyz_to_xyzw, fullstate_xyzw_to_wxyz

pytestmark = pytest.mark.no_sim


@pytest.fixture
def sdk_math(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Stand in for the optional SDK's documented array conversion boundary."""
    calls: list[str] = []
    module = ModuleType("isaaclab.utils.math")

    def convert_quat(quat: npt.NDArray[np.float64], to: str) -> npt.NDArray[np.float64]:
        assert isinstance(quat, np.ndarray)
        assert quat.shape == (4,)
        calls.append(to)
        return np.roll(quat, -1 if to == "xyzw" else 1)

    module.convert_quat = convert_quat  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "isaaclab.utils.math", module)
    return calls


@pytest.mark.parametrize("family", [2, 3])
@pytest.mark.parametrize("container", [list, tuple])
def test_config_quaternions_reuse_the_sdk_boundary_in_the_declared_direction(
    sdk_math: list[str], family: int, container: type
) -> None:
    backend = IsaacSim6Backend() if family == 3 else IsaacSim5Backend()
    config = container([0.4, 0.1, -0.2, 0.3])
    expected_native = (0.1, -0.2, 0.3, 0.4) if family == 3 else (0.4, 0.1, -0.2, 0.3)
    native = backend.config_quat_to_native(config)
    restored = backend.config_quat_from_native(container(expected_native))
    assert native == expected_native
    assert restored == (0.4, 0.1, -0.2, 0.3)
    # Native ray patterns use Torch's default float dtype. NumPy float64
    # scalars here would make the SDK's quaternion rotation reject those rays.
    assert torch.tensor(native).dtype == torch.get_default_dtype()
    assert torch.tensor(restored).dtype == torch.get_default_dtype()
    assert sdk_math == (["xyzw", "wxyz"] if family == 3 else [])
    assert config == container([0.4, 0.1, -0.2, 0.3])


@pytest.mark.parametrize("family", [2, 3])
def test_runtime_quaternions_keep_batched_values_dtype_and_native_identity(family: int) -> None:
    backend = IsaacSim6Backend() if family == 3 else IsaacSim5Backend()
    # A non-contiguous view exercises the last dimension across env/body axes.
    storage = torch.tensor(
        [[[0.1, 99.0, -0.2, 99.0, 0.3, 99.0, 0.4, 99.0], [0.5, 99.0, 0.6, 99.0, -0.7, 99.0, 0.8, 99.0]]],
        dtype=torch.float64,
    )
    xyzw = storage[..., ::2]
    original = storage.clone()
    expected_native = (
        torch.tensor([[[0.4, 0.1, -0.2, 0.3], [0.8, 0.5, 0.6, -0.7]]], dtype=torch.float64)
        if family == 2
        else torch.tensor([[[0.1, -0.2, 0.3, 0.4], [0.5, 0.6, -0.7, 0.8]]], dtype=torch.float64)
    )
    native = backend.quat_to_native(xyzw)
    torch.testing.assert_close(native, expected_native, rtol=0, atol=0)
    assert native.dtype == torch.float64
    torch.testing.assert_close(backend.quat_from_native(expected_native), xyzw, rtol=0, atol=0)
    torch.testing.assert_close(storage, original, rtol=0, atol=0)
    if family == 3:
        assert native is xyzw
        assert backend.quat_from_native(xyzw) is xyzw


@pytest.mark.parametrize(
    ("convert", "state", "expected"),
    [
        (fullstate_wxyz_to_xyzw, [[10, 20, 30, 4, 1, 2, 3, 40, 50]], [[10, 20, 30, 1, 2, 3, 4, 40, 50]]),
        (fullstate_xyzw_to_wxyz, [[10, 20, 30, 1, 2, 3, 4, 40, 50]], [[10, 20, 30, 4, 1, 2, 3, 40, 50]]),
    ],
)
def test_full_state_helpers_reuse_quaternion_conversion_without_changing_other_columns(
    convert: Callable[[torch.Tensor], torch.Tensor], state: list[list[int]], expected: list[list[int]]
) -> None:
    values = torch.tensor(state, dtype=torch.float32)
    original = values.clone()
    result = convert(values)
    torch.testing.assert_close(result, torch.tensor(expected, dtype=torch.float32), rtol=0, atol=0)
    torch.testing.assert_close(values, original, rtol=0, atol=0)
    assert result.data_ptr() != values.data_ptr()


def test_backend_import_and_runtime_quaternion_helpers_do_not_import_the_sdk() -> None:
    code = """
import sys
from holosoma.simulator.isaacsim.backends.isaacsim5 import IsaacSim5Backend
from holosoma.simulator.isaacsim.backends.isaacsim6 import IsaacSim6Backend
import torch
value = torch.tensor([[1., 2., 3., 4.]])
assert IsaacSim5Backend().quat_to_native(value).tolist() == [[4., 1., 2., 3.]]
assert IsaacSim6Backend().quat_to_native(value) is value
assert not any(name == 'isaaclab' or name.startswith('isaaclab.') for name in sys.modules)
"""
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
