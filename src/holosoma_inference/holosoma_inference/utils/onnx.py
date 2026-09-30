"""ONNX Runtime session construction for policy models."""

from __future__ import annotations

import onnxruntime

from holosoma_inference.config.config_types.task import OnnxRuntimeConfig

_EXECUTION_MODES = {
    "sequential": onnxruntime.ExecutionMode.ORT_SEQUENTIAL,
    "parallel": onnxruntime.ExecutionMode.ORT_PARALLEL,
}


def make_session_options(config: OnnxRuntimeConfig) -> onnxruntime.SessionOptions:
    """Build SessionOptions from config, leaving unset knobs at ORT's defaults."""
    options = onnxruntime.SessionOptions()
    if config.intra_op_num_threads:
        options.intra_op_num_threads = config.intra_op_num_threads
    if config.inter_op_num_threads:
        options.inter_op_num_threads = config.inter_op_num_threads
    options.enable_cpu_mem_arena = config.enable_cpu_mem_arena
    options.enable_mem_pattern = config.enable_mem_pattern
    options.execution_mode = _EXECUTION_MODES[config.execution_mode]
    # session.intra_op.allow_spinning stays at its default (on): turning it off
    # measured a 44 ms forward-pass max. Do not "optimise" it away.
    return options


def create_policy_session(model_path: str, config: OnnxRuntimeConfig) -> onnxruntime.InferenceSession:
    """Load an ONNX policy with bounded thread pools and allocator switches."""
    session_options = make_session_options(config)
    return onnxruntime.InferenceSession(model_path, session_options)
