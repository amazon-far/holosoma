"""Fuse a depth-distillation ``depth_backbone.onnx`` + ``student.onnx`` pair into one contextual ONNX.

holosoma_inference's native contextual policy (``ContextualPolicy`` / ``NativeContextualRunner``)
runs one model with inputs ``obs [1, 6 + 3n + command_dim]`` and ``depth_image [1, T, H, W]`` and
output ``actions [1, n]``. Its ``obs`` is ``[gravity | omega | q - default | dq | last action |
command]``. A depth-distillation student pair instead runs the backbone on one depth frame and feeds
the student ``[actions | ang_vel | dof_pos | dof_vel | projected_gravity | command | depth_latent]``
(the training group's sorted term order). The fused graph reorders ``obs`` into that order, runs
the backbone on the single depth frame, concatenates the latent and runs the student. Every term
has the same meaning on both sides; only the order differs.

Usage:
    python src/holosoma_inference/holosoma_cpp/scripts/fuse_depth_student_onnx.py \
        --backbone depth_backbone.onnx --student student.onnx --out contextual.onnx [--check]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, compose, helper, numpy_helper

COMMAND_DIM = 15


def contextual_to_training_order(num_joints: int, command_dim: int = COMMAND_DIM) -> np.ndarray:
    """Indices that turn a contextual ``obs`` into the student's training order."""
    n = num_joints
    gravity, omega = np.arange(0, 3), np.arange(3, 6)
    q, dq = np.arange(6, 6 + n), np.arange(6 + n, 6 + 2 * n)
    action = np.arange(6 + 2 * n, 6 + 3 * n)
    command = np.arange(6 + 3 * n, 6 + 3 * n + command_dim)
    return np.concatenate([action, omega, q, dq, gravity, command]).astype(np.int64)


def _dims(value_info: onnx.ValueInfoProto) -> list[int]:
    return [d.dim_value for d in value_info.type.tensor_type.shape.dim]


def fuse(backbone: onnx.ModelProto, student: onnx.ModelProto, command_dim: int = COMMAND_DIM) -> onnx.ModelProto:
    (bb_in,) = backbone.graph.input
    (bb_out,) = backbone.graph.output
    frame_shape = _dims(bb_in)  # e.g. [1, H, W]: the backbone reads one frame
    if len(frame_shape) != 3 or frame_shape[0] != 1:
        raise ValueError(f"Expected a single-frame backbone input [1, H, W], got {frame_shape}.")
    latent_dim = _dims(bb_out)[1]
    student_inputs = {i.name: _dims(i) for i in student.graph.input}
    if "obs" not in student_inputs:
        raise ValueError(f"Student has no 'obs' input: {sorted(student_inputs)}.")
    student_obs_dim = student_inputs["obs"][1]
    actions = next(o for o in student.graph.output if o.name == "actions")
    num_joints = _dims(actions)[1]
    proprio_dim = 6 + 3 * num_joints + command_dim
    if student_obs_dim != proprio_dim + latent_dim:
        raise ValueError(f"Student obs is {student_obs_dim} wide, expected {proprio_dim} + latent {latent_dim}.")

    bb = compose.add_prefix(backbone, "backbone/")
    st = compose.add_prefix(student, "student/")
    _, height, width = frame_shape

    initializers = [
        numpy_helper.from_array(contextual_to_training_order(num_joints, command_dim), "fuse/obs_order"),
        numpy_helper.from_array(np.asarray(frame_shape, dtype=np.int64), "fuse/frame_shape"),
        *bb.graph.initializer,
        *st.graph.initializer,
    ]
    if "time_step" in student_inputs:
        # The contextual kernel feeds time_step = 0; the depth students ignore it.
        initializers.append(numpy_helper.from_array(np.zeros((1, 1), dtype=np.float32), "student/time_step"))
    nodes = [
        helper.make_node("Gather", ["obs", "fuse/obs_order"], ["fuse/obs_training_order"], axis=1),
        helper.make_node("Reshape", ["depth_image", "fuse/frame_shape"], ["backbone/" + bb_in.name]),
        *bb.graph.node,
        helper.make_node("Concat", ["fuse/obs_training_order", "backbone/" + bb_out.name], ["student/obs"], axis=1),
        *st.graph.node,
        helper.make_node("Identity", ["student/actions"], ["actions"]),
    ]
    graph = helper.make_graph(
        nodes,
        "contextual_depth_student",
        inputs=[
            helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, proprio_dim]),
            helper.make_tensor_value_info("depth_image", TensorProto.FLOAT, [1, 1, height, width]),
        ],
        outputs=[helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, num_joints])],
        initializer=initializers,
    )
    opset = max(o.version for o in [*backbone.opset_import, *student.opset_import] if o.domain in ("", "ai.onnx"))
    fused = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    fused.ir_version = student.ir_version
    metadata = {p.key: p.value for p in student.metadata_props}
    # The student's observation_names lists the training config's declaration order, not the order
    # it consumed; record the fused input explicitly.
    metadata["observation_names"] = "projected_gravity,base_ang_vel,dof_pos,dof_vel,actions,velocity_command"
    metadata["fused_from"] = json.dumps(
        {
            "backbone_input": bb_in.name,
            "student_obs_order": "actions,base_ang_vel,dof_pos,dof_vel,projected_gravity,velocity_command,depth_latent",
            "latent_dim": latent_dim,
        }
    )
    for key, value in metadata.items():
        fused.metadata_props.add(key=key, value=value)
    onnx.checker.check_model(fused, full_check=True)
    return fused


def check(backbone_path: Path, student_path: Path, fused_path: Path, trials: int = 32) -> float:
    """Max abs action difference between the fused model and the pair run separately."""
    import onnxruntime as ort

    bb = ort.InferenceSession(str(backbone_path))
    st = ort.InferenceSession(str(student_path))
    fused = ort.InferenceSession(str(fused_path))
    student_inputs = {i.name for i in st.get_inputs()}
    num_joints = st.get_outputs()[0].shape[1]
    order = contextual_to_training_order(num_joints)
    height, width = bb.get_inputs()[0].shape[1:]
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(trials):
        obs = rng.standard_normal((1, len(order))).astype(np.float32)
        depth = rng.uniform(-0.5, 0.5, (1, 1, height, width)).astype(np.float32)
        (latent,) = bb.run(None, {bb.get_inputs()[0].name: depth[:, 0]})
        feed = {"obs": np.concatenate([obs[:, order], latent], axis=1)}
        if "time_step" in student_inputs:
            feed["time_step"] = np.zeros((1, 1), dtype=np.float32)
        expected = st.run(["actions"], feed)[0]
        (actual,) = fused.run(["actions"], {"obs": obs, "depth_image": depth})
        worst = max(worst, float(np.abs(actual - expected).max()))
    return worst


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True, type=Path)
    parser.add_argument("--student", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--check", action="store_true", help="Compare against the pair run separately")
    args = parser.parse_args()

    fused = fuse(onnx.load(args.backbone), onnx.load(args.student))
    onnx.save(fused, args.out)
    print(f"wrote {args.out}")
    if args.check:
        print(f"max abs action difference vs the pair: {check(args.backbone, args.student, args.out):.3g}")


if __name__ == "__main__":
    main()
