"""Fuse a depth-distillation ``depth_backbone.onnx`` + ``student.onnx`` pair into one contextual ONNX.

holosoma_inference's native contextual policy (``ContextualPolicy`` / ``NativeContextualRunner``)
runs one model with inputs ``obs [1, 6 + 3n + command_dim]`` and ``depth_image [1, T, H, W]`` and
output ``actions [1, n]``. Its ``obs`` is ``[gravity | omega | q - default | dq | last action |
command]``. A depth-distillation student pair instead runs the backbone on one depth frame and feeds
the student its actor terms in the order the observation manager concatenates them (sorted by term
name), then the depth latent. The fused graph reorders ``obs`` into that order, runs the backbone
on the single depth frame, concatenates the latent and runs the student. Every term has the same
meaning on both sides; only the order differs.

The term names come from the student's ``observation_names`` metadata. Exporters wrote it either
already sorted (``sorted(actor_terms) + ["placeholder"]``) or in declaration order, so the names are
sorted here either way. Without that metadata the usual PHP student terms are assumed; ``fused_from``
records which applied.

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
LATENT_TERM = "placeholder"  # the exporters' label for the appended depth latent
ASSUMED_TERMS = ("actions", "base_ang_vel", "dof_pos", "dof_vel", "projected_gravity", "velocity_command")


def _contextual_block(term: str) -> str:
    """The contextual ``obs`` block a student actor term reads."""
    if term.endswith("projected_gravity"):  # e.g. robot_anchor_projected_gravity
        return "gravity"
    blocks = {
        "base_ang_vel": "omega",
        "dof_pos": "q",
        "dof_vel": "dq",
        "actions": "action",
        "velocity_command": "command",
    }
    if term not in blocks:
        raise ValueError(f"Student observation term {term!r} has no counterpart in the contextual obs.")
    return blocks[term]


def student_terms(student: onnx.ModelProto) -> tuple[list[str], str]:
    """The student's actor terms in the order it consumes them, and where that order came from."""
    names = {p.key: p.value for p in student.metadata_props}.get("observation_names")
    if not names:
        return list(ASSUMED_TERMS), "assumed (no observation_names metadata)"
    return sorted(t for t in names.split(",") if t and t != LATENT_TERM), "observation_names, sorted"


def contextual_to_training_order(
    num_joints: int, command_dim: int = COMMAND_DIM, terms: list[str] | tuple[str, ...] = ASSUMED_TERMS
) -> np.ndarray:
    """Indices that turn a contextual ``obs`` into the student's ``terms`` order."""
    n = num_joints
    blocks = {
        "gravity": np.arange(0, 3),
        "omega": np.arange(3, 6),
        "q": np.arange(6, 6 + n),
        "dq": np.arange(6 + n, 6 + 2 * n),
        "action": np.arange(6 + 2 * n, 6 + 3 * n),
        "command": np.arange(6 + 3 * n, 6 + 3 * n + command_dim),
    }
    order = [_contextual_block(t) for t in terms]
    if sorted(order) != sorted(blocks):
        raise ValueError(f"Student terms {list(terms)} must read each contextual block exactly once.")
    return np.concatenate([blocks[b] for b in order]).astype(np.int64)


def _json_number_list(value: str) -> str:
    """A comma-separated number list as a JSON list; other values unchanged.

    The exported students write ``action_scale``, ``kp`` and ``kd`` as ``"1.0,1.0,..."``; the contextual
    kernel reads those keys as JSON lists.
    """
    try:
        return json.dumps([float(v) for v in value.split(",") if v.strip()]) if "," in value else value
    except ValueError:
        return value


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
    terms, order_source = student_terms(student)
    if student_obs_dim != proprio_dim + latent_dim:
        raise ValueError(f"Student obs is {student_obs_dim} wide, expected {proprio_dim} + latent {latent_dim}.")

    bb = compose.add_prefix(backbone, "backbone/")
    st = compose.add_prefix(student, "student/")
    _, height, width = frame_shape

    initializers = [
        numpy_helper.from_array(contextual_to_training_order(num_joints, command_dim, terms), "fuse/obs_order"),
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
    metadata = {p.key: _json_number_list(p.value) for p in student.metadata_props}
    # The fused model reads the contextual obs; the student's own order is kept in fused_from.
    metadata["observation_names"] = "projected_gravity,base_ang_vel,dof_pos,dof_vel,actions,velocity_command"
    metadata["fused_from"] = json.dumps(
        {
            "backbone_input": bb_in.name,
            "student_obs_order": ",".join([*terms, "depth_latent"]),
            "student_obs_order_source": order_source,
            "latent_dim": latent_dim,
        }
    )
    for key, value in metadata.items():
        fused.metadata_props.add(key=key, value=value)
    onnx.checker.check_model(fused, full_check=True)
    return fused


def check(backbone_path: Path, student_path: Path, fused_path: Path, trials: int = 32) -> float:
    """Max abs action difference between the fused model and the pair run separately.

    The pair is fed in the same term order the fusion used, so this checks the fused graph, not the
    order itself; that rests on ``observation_names`` (see the module docstring).
    """
    import onnxruntime as ort

    bb = ort.InferenceSession(str(backbone_path))
    st = ort.InferenceSession(str(student_path))
    fused = ort.InferenceSession(str(fused_path))
    student_inputs = {i.name for i in st.get_inputs()}
    num_joints = st.get_outputs()[0].shape[1]
    order = contextual_to_training_order(num_joints, terms=student_terms(onnx.load(str(student_path)))[0])
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
