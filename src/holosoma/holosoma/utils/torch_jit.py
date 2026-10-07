"""
PyTorch JIT Compilation Utilities with Proxy Object Support

This module provides utilities for PyTorch JIT compilation for handling simulator proxy objects.

Example Usage:
    ```python

    # Minimal proxy class
    class MyTensorProxy:
        def __init__(self, data):
            self.data = data
            self._is_tensor_proxy = True  # Required for detection

        def __getitem__(self, key):
            return torch.tensor(self.data)


    # Usage with the JIT decorator:
    @torch_jit_script
    def my_function(x: torch.Tensor) -> torch.Tensor:
        return x * 2 + 1


    proxy = MyTensorProxy([1.0, 2.0, 3.0])
    result = my_function(proxy)  # Automatically converts proxy to tensor
    ```
"""

from __future__ import annotations

import functools
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Callable, TypeVar, cast

from holosoma.utils.safe_torch_import import torch

# Define a TypeVar for preserving function signatures in type hints
# This ensures that decorated functions maintain their original type signatures
F = TypeVar("F", bound=Callable[..., Any])


# Global dictionary is the key to making the decorator robust against multiple module imports
# and nested JIT calls. It stores the final, fully-decorated function, preventing any function
# from being processed more than once. The key format is "{module}.{qualname}" to ensure
# uniqueness across the entire codebase.
_COMPILED_FUNCTION_CACHE: dict[str, Callable[..., Any]] = {}


# Duck-typing attribute name that tensor proxy objects use to self-identify.
# Proxy objects from libraries like distributed tensor frameworks or lazy
# evaluation systems set this to True, allowing automatic detection and
# conversion to real tensors before JIT compilation.
_PROXY_ATTRIBUTE_NAME = "_is_tensor_proxy"


# =============================================================================
# Proxy Object Compatibility
# =============================================================================


def _is_tensor_proxy(obj: Any) -> bool:
    """
    Detect if an object is a tensor proxy based on an attribute (if it exists)
    """
    return getattr(obj, _PROXY_ATTRIBUTE_NAME, False)


def proxy_compatible(func: F) -> F:
    """
    Decorator to make JIT functions compatible with proxy objects and mixed dtypes.

    The conversion process:
    1. Check if an argument has the `_is_tensor_proxy` attribute (proxy detection)
    2. If it's already a torch.Tensor, pass it through unchanged
    3. If it's a proxy, attempt conversion using slicing (`arg[:]`)
    4. Validate that the conversion produced a torch.Tensor
    5. Raise descriptive errors if conversion fails

    Args:
        func: The function to wrap with proxy compatibility

    Returns:
        A wrapped function that automatically converts proxy objects to tensors

    Raises:
        TypeError: If a proxy object cannot be converted to a torch.Tensor
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        def _convert_if_proxy(arg: Any) -> Any:
            """
            Convert a single argument from proxy to tensor if needed.

            Uses the `_is_tensor_proxy` attribute as a duck-typing mechanism
            to identify proxy objects. This is a common convention in tensor
            libraries that implement lazy evaluation or distributed tensors.
            """
            # Fast path: if it's not a proxy or already a tensor, return as-is
            if not _is_tensor_proxy(arg) or isinstance(arg, torch.Tensor):
                return arg  # do nothing

            # Proxy conversion path: attempt to materialize the proxy
            try:
                # Use slicing to trigger proxy materialization - this is a common
                # pattern that forces lazy tensors to compute their actual values
                converted_arg = arg[:]  # else convert the proxy

                # Validate that conversion actually produced a tensor
                if isinstance(converted_arg, torch.Tensor):
                    return converted_arg

                # Conversion succeeded but didn't produce a tensor - this is unexpected
                raise TypeError(
                    f"Object of type {type(arg).__name__} converted to "
                    f"{type(converted_arg).__name__}, not torch.Tensor."
                )
            except Exception as e:
                # Conversion failed entirely - provide helpful error message
                raise TypeError(
                    f"Object of type {type(arg).__name__} appeared to be a proxy "
                    f"but failed conversion to a tensor via slicing."
                ) from e

        # Apply proxy conversion to all positional and keyword arguments
        # This ensures that the wrapped function receives only real tensors
        converted_args = tuple(_convert_if_proxy(arg) for arg in args)
        converted_kwargs = {k: _convert_if_proxy(v) for k, v in kwargs.items()}

        # Call the original function with converted arguments
        return func(*converted_args, **converted_kwargs)

    return wrapper  # type: ignore[return-value]


# =============================================================================
# Nested JIT Dependency Resolution
# =============================================================================


@contextmanager
def _unwrap_jit_dependencies(func: F) -> Iterator[None]:
    """
    Context manager to solve the nested JIT compilation problem.

    When PyTorch JIT compiles a function that calls other JIT-compiled functions,
    it needs to see the actual compiled code (ScriptFunction) rather than the
    wrapped decorator functions. This context manager temporarily "unwraps" any
    JIT-compiled dependencies in the function's global scope during compilation.

    The Problem:
    - Function A calls Function B
    - Both are decorated with @torch_jit_script
    - When compiling A, PyTorch sees the wrapped version of B (with proxy handling)
    - PyTorch can't compile the wrapper, causing compilation to fail

    The Solution:
    - Before compiling A, replace B in A's globals with B's raw ScriptFunction
    - Compile A successfully (it can now see B's compiled code)
    - Restore B's wrapped version after compilation

    Args:
        func: The function being compiled that may have JIT-compiled dependencies

    Yields:
        None (context manager for use with 'with' statement)

    Example:
        ```python
        @torch_jit_script
        def helper_func(x: torch.Tensor) -> torch.Tensor:
            return x * 2

        @torch_jit_script  # This will use _unwrap_jit_dependencies internally
        def main_func(x: torch.Tensor) -> torch.Tensor:
            return helper_func(x) + 1  # Can call other JIT functions
        ```
    """
    # Get the function's global namespace where dependencies might be defined
    func_globals = func.__globals__
    original_globals = {}

    # Scan through all cached JIT functions to find dependencies
    for key, cached_func in _COMPILED_FUNCTION_CACHE.items():
        # Extract just the function name from the full key (module.qualname)
        func_name = key.split(".")[-1]

        # Check if this cached function exists in the current function's globals
        # and verify it's the same object (using id() for identity comparison)
        if func_name in func_globals and id(func_globals[func_name]) == id(cached_func):
            # Store the original wrapped function for later restoration
            original_globals[func_name] = cached_func

            # Replace with the raw ScriptFunction that PyTorch JIT can understand
            # cached_func.__wrapped__ contains the original torch.jit.script result
            func_globals[func_name] = cached_func.__wrapped__  # type: ignore[attr-defined]

    try:
        # Yield control back to the caller with dependencies unwrapped
        # This is where the actual JIT compilation happens
        yield
    finally:
        # Critical cleanup: restore the original wrapped functions
        # This ensures that future calls to these functions still get proxy handling
        func_globals.update(original_globals)


# =============================================================================
# Main JIT Compilation Interface
# =============================================================================


def torch_jit_script(func: F) -> Callable[..., Any]:
    """
    Main decorator that applies torch.jit.script with proxy object support and caching.

    This is the primary interface for JIT compilation in this module. It combines
    PyTorch's JIT compilation with automatic proxy object handling and intelligent
    caching to create a robust, production-ready JIT decorator.

    The compilation process:
    1. Generate unique key from module and function name
    2. Return cached version if already compiled
    3. Unwrap any JIT dependencies in the function's scope
    4. Compile with torch.jit.script
    5. Wrap result with proxy compatibility
    6. Cache and return the final wrapped function

    Args:
        func: The function to compile with PyTorch JIT

    Returns:
        A JIT-compiled function with proxy object support

    Raises:
        AssertionError: If JIT compilation fails (no graph attribute)

    Example:
        ```python
        @torch_jit_script
        def compute_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.mse_loss(predictions, targets)

        # Function is compiled once, cached, and works with proxy objects
        loss = compute_loss(model_output, ground_truth)
        ```
    """
    # Generate a unique key for this function using module and qualified name
    # This ensures uniqueness even for functions with the same name in different modules
    func_key = f"{func.__module__}.{func.__qualname__}"

    # Fast path: return cached version if already compiled
    # This makes the decorator idempotent and prevents expensive recompilation
    if func_key in _COMPILED_FUNCTION_CACHE:
        return _COMPILED_FUNCTION_CACHE[func_key]

    # Compilation path: handle nested dependencies and compile
    # Use the context manager to temporarily unwrap any JIT-compiled dependencies
    # This allows PyTorch to see the actual compiled code of functions this function calls
    with _unwrap_jit_dependencies(func):
        scripted_func = torch.jit.script(func)

    # Validate that compilation succeeded by checking for the graph attribute
    # All successfully compiled ScriptFunctions have a 'graph' attribute
    assert hasattr(scripted_func, "graph"), f"JIT compilation failed for {func_key}"

    # Wrap the compiled function with proxy compatibility
    # This ensures that the final function can handle proxy objects automatically
    wrapped_func = proxy_compatible(scripted_func)

    # Cache the final wrapped function for future use
    # Store the complete chain: JIT compilation + proxy handling
    _COMPILED_FUNCTION_CACHE[func_key] = wrapped_func

    return cast("Callable[..., Any]", wrapped_func)
