#!/usr/bin/env python3
"""Emit one small policy as .onnx and as ExecuTorch .pte programs, with the same weights, for C++ tests.

The .pte files carry the `signature` method that holosoma.utils.inference_helpers.export_policy_as_pte
writes. That exporter lives in the training package, which these build-time fixtures do not depend on.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower

OBS_DIM = 47
ACTION_DIM = 12
METADATA = {
    "dof_names": [f"joint_{i}" for i in range(ACTION_DIM)],
    "kp": [40.0] * ACTION_DIM,
    "kd": [1.0] * ACTION_DIM,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorrt", action="store_true", help="also export a TensorRT program when a GPU is present")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    policy = torch.nn.Sequential(torch.nn.Linear(OBS_DIM, 64), torch.nn.ELU(), torch.nn.Linear(64, ACTION_DIM)).eval()
    obs = torch.zeros(1, OBS_DIM)

    onnx_path = args.output_dir / "policy.onnx"
    torch.onnx.export(
        policy, obs, onnx_path, input_names=["actor_obs"], output_names=["action"], opset_version=13, dynamo=False
    )
    model = onnx.load(onnx_path)
    for key, value in METADATA.items():
        model.metadata_props.append(onnx.StringStringEntryProto(key=key, value=json.dumps(value)))
    onnx.save(model, onnx_path)

    signature = {
        "inputs": ["actor_obs"],
        "outputs": ["action"],
        "metadata": {key: json.dumps(value) for key, value in METADATA.items()},
    }
    exported = torch.export.export(policy, (obs,))
    for name, constant_methods in (
        ("policy_xnnpack.pte", {"signature": json.dumps(signature)}),
        ("no_signature.pte", None),
    ):
        program = to_edge_transform_and_lower(
            exported, partitioner=[XnnpackPartitioner()], constant_methods=constant_methods
        ).to_executorch()
        (args.output_dir / name).write_bytes(program.buffer)

    tensorrt_path = args.output_dir / "policy_tensorrt.pte"
    tensorrt_path.unlink(missing_ok=True)
    if args.tensorrt and torch.cuda.is_available():
        import torch_tensorrt
        from executorch.backends.cuda.cuda_partitioner import CudaPartitioner

        compiled = torch_tensorrt.dynamo.compile(exported, inputs=[obs.cuda()], min_block_size=1)
        torch_tensorrt.save(
            compiled,
            str(tensorrt_path),
            output_format="executorch",
            partitioners=[CudaPartitioner([])],
            constant_methods={"signature": json.dumps(signature)},
        )


if __name__ == "__main__":
    main()
