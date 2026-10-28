"""Small redistributable contextual graphs for public API regression tests."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def write_model(path, *, head=True, width=30, actions=3, nonfinite=False, raw=False):
    weights = np.zeros((width, actions), dtype=np.float32)
    weights[12, -1] = 1  # Previous clipped action, not scaled joint targets.
    weights[-1, -1] = 1  # Climb command column 14 affects the second inference.
    bias = np.array([np.inf if nonfinite else 50, -50, 5][:actions], dtype=np.float32)
    nodes = [
        helper.make_node("MatMul", ["obs", "weights"], ["feedback"]),
        helper.make_node("Add", ["feedback", "bias"], ["actions"]),
    ]
    outputs = [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, actions])]
    # A graph that preprocesses depth itself takes the camera's frame on "depth_raw" instead.
    depth = "depth_raw" if raw else "depth_image"
    if head:
        nodes.append(helper.make_node("ReduceMean", [depth], ["climb_pred"], keepdims=0))
        outputs.insert(0, helper.make_tensor_value_info("climb_pred", TensorProto.FLOAT, []))
    graph = helper.make_graph(
        nodes,
        "contextual-api",
        [
            helper.make_tensor_value_info("obs", TensorProto.FLOAT, [-1, width]),
            helper.make_tensor_value_info(depth, TensorProto.FLOAT, [-1, 2, 2, 3]),
            helper.make_tensor_value_info("time_step", TensorProto.FLOAT, [-1, 1]),
        ],
        outputs,
        [numpy_helper.from_array(weights, "weights"), numpy_helper.from_array(bias, "bias")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    for name, value in {
        "joint_names": ["ankle", "knee", "hip"],
        "action_scale": [2, 0.5, 3],
        "kp": [30, 20, 10],
        "kd": [3, 2, 1],
    }.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = name, json.dumps(value)
    onnx.checker.check_model(model)
    onnx.save(model, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    root = parser.parse_args().output_dir
    root.mkdir(parents=True, exist_ok=True)
    write_model(root / "contextual.onnx")
    write_model(root / "no-head.onnx", head=False)
    write_model(root / "wrong-width.onnx", width=31)
    write_model(root / "wrong-actions.onnx", actions=2)
    write_model(root / "nonfinite.onnx", nonfinite=True)


if __name__ == "__main__":
    main()
