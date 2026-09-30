"""Unit tests for ONNX Runtime session option construction."""

import dataclasses as dc

import onnxruntime

from holosoma_inference.config.config_types.task import OnnxRuntimeConfig
from holosoma_inference.utils.onnx import make_session_options


def test_defaults_match_bare_session_options():
    """A default config must leave every knob where ORT put it."""
    options = make_session_options(OnnxRuntimeConfig())
    bare = onnxruntime.SessionOptions()

    assert options.intra_op_num_threads == bare.intra_op_num_threads
    assert options.inter_op_num_threads == bare.inter_op_num_threads
    assert options.enable_cpu_mem_arena == bare.enable_cpu_mem_arena
    assert options.enable_mem_pattern == bare.enable_mem_pattern
    assert options.execution_mode == bare.execution_mode


def test_config_fields_reach_session_options():
    """Non-default fields are applied, including the parallel execution mode."""
    config = dc.replace(
        OnnxRuntimeConfig(),
        intra_op_num_threads=1,
        inter_op_num_threads=2,
        enable_cpu_mem_arena=False,
        enable_mem_pattern=False,
        execution_mode="parallel",
    )
    options = make_session_options(config)

    assert options.intra_op_num_threads == 1
    assert options.inter_op_num_threads == 2
    assert options.enable_cpu_mem_arena is False
    assert options.enable_mem_pattern is False
    assert options.execution_mode == onnxruntime.ExecutionMode.ORT_PARALLEL
