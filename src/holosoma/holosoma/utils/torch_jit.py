"""PyTorch JIT compilation with caching: each function is scripted once, however often its module is imported."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar, cast

from holosoma.utils.safe_torch_import import torch

F = TypeVar("F", bound=Callable[..., Any])

# Keyed by "{module}.{qualname}", so repeated imports of a module reuse the compiled function.
_COMPILED_FUNCTION_CACHE: dict[str, Callable[..., Any]] = {}


def torch_jit_script(func: F) -> Callable[..., Any]:
    """Compile ``func`` with ``torch.jit.script``, once per module-qualified name.

    Raises:
        AssertionError: If compilation produced no graph.
    """
    func_key = f"{func.__module__}.{func.__qualname__}"
    if func_key in _COMPILED_FUNCTION_CACHE:
        return _COMPILED_FUNCTION_CACHE[func_key]
    scripted_func = torch.jit.script(func)
    assert hasattr(scripted_func, "graph"), f"JIT compilation failed for {func_key}"
    _COMPILED_FUNCTION_CACHE[func_key] = scripted_func
    return cast("Callable[..., Any]", scripted_func)
