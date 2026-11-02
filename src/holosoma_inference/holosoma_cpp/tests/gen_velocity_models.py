"""Continuous-velocity contextual graphs for public API regression tests.

Two sets of fixtures:

* The reference traces. `velocity_reference.json` (checked in) was produced by the
  reviewed native kernel's own golden generator (its independent numpy statement of
  the deployment contract, seed 20261031). This script rebuilds the two graphs that
  trace describes, from the same seed, so the C++ family is replayed against
  numbers no line of it produced.
* The export contract. `velocity_export_metadata.json` (checked in) is the
  verbatim metadata of a real continuous-velocity export, without weights. Each
  stub graph here carries it, either unchanged or with one key broken, so every
  check the family makes on that contract is exercised by name.
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

GOLDEN = Path(__file__).parent / "golden"
EXPORT = json.loads((GOLDEN / "velocity_export_metadata.json").read_text())
G1_DOF_NAMES = json.loads(EXPORT["metadata_props"]["dof_names"])
G1_DEFAULT_DOF_POS = json.loads(EXPORT["metadata_props"]["default_dof_angles"])

# Reference trace contract (must match the generator that wrote velocity_reference.json).
HEIGHT, WIDTH = 2, 3
TERMS = ["actions", "base_ang_vel", "dof_pos", "dof_vel", "fixed_arm_flag", "projected_gravity", "velocity_command"]
DIMS = {
    "actions": 29,
    "base_ang_vel": 3,
    "dof_pos": 29,
    "dof_vel": 29,
    "fixed_arm_flag": 2,
    "projected_gravity": 3,
    "velocity_command": 3,
}
SCALES = {"base_ang_vel": 0.25, "dof_vel": 0.05}
RANGES = {"lin_vel_x": [-1.0, 1.5], "lin_vel_y": [-0.5, 0.5], "ang_vel_yaw": [-1.5, 1.5]}
INPUT = sum(DIMS.values()) + HEIGHT * WIDTH


def contract(rng):
    order = rng.permutation(29)
    names = [G1_DOF_NAMES[i] for i in order]
    defaults = np.asarray(G1_DEFAULT_DOF_POS, dtype=np.float64)[order] + rng.uniform(-0.05, 0.05, 29)
    return {
        "names": names,
        "defaults": defaults,
        "scale": rng.uniform(0.15, 0.9, 29),
        "kp": rng.uniform(20.0, 150.0, 29),
        "kd": rng.uniform(1.0, 5.0, 29),
    }


def save(path, graph, metadata):
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    for key, value in metadata.items():
        prop = model.metadata_props.add()
        prop.key, prop.value = key, value
    onnx.checker.check_model(model)
    onnx.save(model, path)


def reference_model(path, c, nodes, initializers):
    graph = helper.make_graph(
        nodes,
        "velocity_reference",
        [helper.make_tensor_value_info("actor_obs", TensorProto.FLOAT, [1, INPUT])],
        [helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, 29])],
        initializer=initializers,
    )
    meta = {
        "policy_kind": "hybrid_velocity",
        "actor_observation_terms": TERMS,
        "actor_observation_term_dims": DIMS,
        "actor_observation_scales": {t: SCALES.get(t, 1.0) for t in TERMS},
        "actor_observation_history_length": 1,
        "actor_policy_input_dim": INPUT,
        "dof_names": c["names"],
        "default_dof_angles": c["defaults"].tolist(),
        "action_scale": c["scale"].tolist(),
        "kp": c["kp"].tolist(),
        "kd": c["kd"].tolist(),
        "command_ranges": RANGES,
        "depth_policy_shape_hw": [HEIGHT, WIDTH],
    }
    save(path, graph, {key: json.dumps(value) for key, value in meta.items()})


def reference_models(out):
    """Replays the reference generator's RNG draws in its order."""
    rng = np.random.default_rng(20261031)
    c = contract(rng)
    weight = rng.normal(0, 0.08, (INPUT, 29)).astype(np.float32)
    bias = rng.normal(0, 0.3, (1, 29)).astype(np.float32)
    reference_model(
        out / "velocity_linear.onnx",
        c,
        [helper.make_node("MatMul", ["actor_obs", "W"], ["xw"]), helper.make_node("Add", ["xw", "b"], ["action"])],
        [numpy_helper.from_array(weight, "W"), numpy_helper.from_array(bias, "b")],
    )
    # The reference trace consumes draws for its 30 linear steps before the
    # second contract; replay them so the second graph matches.
    for i in range(30):
        if i % 7 == 3:
            rng.integers(0, 3)
        if i % 5:
            rng.uniform(-1.5, 2.0), rng.uniform(-0.8, 0.8), rng.uniform(-2.0, 2.0)
        rng.normal(size=4)
        if not i % 3:
            rng.normal(size=3)
        rng.normal(0, 0.2, 29)
        rng.normal(0, 1.5, 29)
        rng.normal(0, 0.6, 3)
        if i % 4 != 1:
            rng.uniform(0, 1, HEIGHT * WIDTH)
    c2 = contract(rng)
    reference_model(
        out / "velocity_feedback.onnx",
        c2,
        [
            helper.make_node("Slice", ["actor_obs", "s", "e", "a"], ["prev"]),
            helper.make_node("Add", ["prev", "bias"], ["action"]),
        ],
        [
            numpy_helper.from_array(np.array([0], np.int64), "s"),
            numpy_helper.from_array(np.array([29], np.int64), "e"),
            numpy_helper.from_array(np.array([1], np.int64), "a"),
            numpy_helper.from_array(np.full((1, 29), 15.0, np.float32), "bias"),
        ],
    )


def export_stub(path, metadata, *, width=None, actions=None):
    """A graph with the real export's signature: action = actor_obs[:, :29] + bias.

    The bias makes the first action differ from the stance, and feeding back the
    previous action makes the second tick read it.
    """
    width = width or EXPORT["inputs"][0]["shape"][1]
    actions = actions or EXPORT["outputs"][0]["shape"][1]
    graph = helper.make_graph(
        [
            helper.make_node("Slice", ["actor_obs", "s", "e", "a"], ["prev"]),
            helper.make_node("Add", ["prev", "bias"], ["action"]),
        ],
        "velocity_export_stub",
        [helper.make_tensor_value_info("actor_obs", TensorProto.FLOAT, [1, width])],
        [helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, actions])],
        initializer=[
            numpy_helper.from_array(np.array([0], np.int64), "s"),
            numpy_helper.from_array(np.array([actions], np.int64), "e"),
            numpy_helper.from_array(np.array([1], np.int64), "a"),
            numpy_helper.from_array(np.linspace(-1, 1, actions, dtype=np.float32)[None, :], "bias"),
        ],
    )
    save(path, graph, metadata)


def edit(key, change):
    """Metadata with `key` replaced by change(parsed value); a change returning
    None removes the key, and a string result is stored verbatim."""
    metadata = copy.deepcopy(EXPORT["metadata_props"])
    value = change(json.loads(metadata[key]))
    if value is None:
        del metadata[key]
    else:
        metadata[key] = value if isinstance(value, str) else json.dumps(value)
    return metadata


def with_item(index, value):
    def change(items):
        items[index] = value
        return items

    return change


def with_field(name, value):
    def change(fields):
        if value is None:
            del fields[name]
        else:
            fields[name] = value
        return fields

    return change


# Each broken export and the text its refusal must contain.
BROKEN = {
    "policy_kind_missing": (edit("policy_kind", lambda _: None), "policy_kind is missing"),
    "policy_kind_primitive": (edit("policy_kind", lambda _: "contextual"), "policy_kind must be"),
    "dof_names_missing": (edit("dof_names", lambda _: None), "dof_names is missing"),
    "dof_names_repeated": (edit("dof_names", with_item(1, G1_DOF_NAMES[0])), "dof_names must name every"),
    "dof_names_foreign": (edit("dof_names", with_item(0, "tail_joint")), "dof_names must name every"),
    "dof_names_not_names": (edit("dof_names", lambda _: [1, 2]), "dof_names must be a list"),
    "default_dof_angles_short": (edit("default_dof_angles", lambda v: v[:-1]), "default_dof_angles has 28 values"),
    "default_dof_angles_nonfinite": (edit("default_dof_angles", lambda _: "[NaN]"), "default_dof_angles must"),
    "default_dof_angles_too_large": (edit("default_dof_angles", with_item(0, 1e39)), "default_dof_angles must"),
    "action_scale_short": (edit("action_scale", lambda v: v[:2]), "action_scale has 2 values"),
    "action_scale_null": (edit("action_scale", lambda _: "null"), "action_scale is missing"),
    "kp_negative": (edit("kp", with_item(3, -1.0)), "kp must be nonnegative"),
    "kp_short": (edit("kp", lambda v: v[:-1]), "kp has 28 values"),
    "kd_missing": (edit("kd", lambda _: None), "kd is missing"),
    "kd_object": (edit("kd", lambda _: {"a": 1}), "kd must be a number"),
    "command_ranges_inverted": (
        edit("command_ranges", with_field("lin_vel_y", [1.0, -1.0])),
        "command_ranges.lin_vel_y must be [low, high]",
    ),
    "command_ranges_unknown_axis": (edit("command_ranges", with_field("lin_vel_z", [0, 1])), "unknown axis lin_vel_z"),
    "command_ranges_null": (edit("command_ranges", lambda _: "null"), "command_ranges is null"),
    "command_ranges_list": (edit("command_ranges", lambda _: [1, 2]), "command_ranges must be a JSON object"),
    "terms_missing": (edit("actor_observation_terms", lambda _: None), "actor_observation_terms is missing"),
    "terms_unknown": (edit("actor_observation_terms", with_item(0, "base_lin_vel")), "unsupported term base_lin_vel"),
    "terms_repeated": (edit("actor_observation_terms", with_item(1, "actions")), "repeats term actions"),
    "terms_no_command": (
        edit("actor_observation_terms", lambda v: [t for t in v if t != "velocity_command"]),
        "declares velocity_command, which is not a term",
    ),
    "term_dims_wrong": (
        edit("actor_observation_term_dims", with_field("dof_pos", 28)),
        "actor_observation_term_dims.dof_pos is 28 but the term has 29",
    ),
    "term_dims_missing": (
        edit("actor_observation_term_dims", with_field("dof_vel", None)),
        "actor_observation_term_dims.dof_vel is missing",
    ),
    "term_dims_extra": (
        edit("actor_observation_term_dims", with_field("base_lin_vel", 3)),
        "declares base_lin_vel, which is not a term",
    ),
    "term_dims_fractional": (
        edit("actor_observation_term_dims", with_field("dof_pos", 29.5)),
        "actor_observation_term_dims.dof_pos must be an integer",
    ),
    "scales_not_number": (
        edit("actor_observation_scales", with_field("dof_vel", "fast")),
        "actor_observation_scales.dof_vel must be a number",
    ),
    "scales_extra": (
        edit("actor_observation_scales", with_field("base_lin_vel", 1.0)),
        "scales base_lin_vel, which is not a term",
    ),
    "scales_list": (
        edit("actor_observation_scales", lambda _: [1.0]),
        "actor_observation_scales must be a JSON object",
    ),
    "history_two": (edit("actor_observation_history_length", lambda _: 2), "history_length must be 1"),
    "actor_observation_dim_wrong": (edit("actor_observation_dim", lambda _: 97), "actor_observation_dim does not"),
    "input_dim_wrong": (edit("actor_policy_input_dim", lambda _: 5143), "actor_policy_input_dim does not"),
    "depth_dim_wrong": (edit("depth_observation_dim", lambda _: 5045), "depth_observation_dim does not"),
    "depth_shape_wrong": (edit("depth_policy_shape_hw", lambda _: [87, 58]), "depth_policy_shape_hw does not"),
    "input_groups_reversed": (edit("actor_policy_input_groups", lambda v: v[::-1]), "actor_policy_input_groups must"),
    "command_frame_world": (edit("velocity_command_frame", lambda _: "world"), "velocity_command_frame must be"),
    "projection_not_embedded": (
        edit("velocity_command_projection", with_field("embedded_in_graph", False)),
        "embedded_in_graph must be true",
    ),
    "projection_span_moved": (
        edit("velocity_command_projection", with_field("observation_start", 94)),
        "observation span does not match velocity_command at [95, 98)",
    ),
    "projection_deadband": (
        edit("velocity_command_projection", with_field("linear_deadband", 0.05)),
        "linear_deadband must be 0",
    ),
    "projection_yaw_range": (
        edit("velocity_command_projection", with_field("yaw_rate_range", [-1.0, 1.0])),
        "yaw_rate_range does not match",
    ),
    "projection_scale": (
        edit("velocity_command_projection", with_field("observation_scale", 2.0)),
        "observation_scale does not match",
    ),
    "projection_history": (edit("velocity_command_projection", with_field("history_length", 2)), "must be 1"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    out = parser.parse_args().output_dir
    out.mkdir(parents=True, exist_ok=True)
    reference_models(out)
    export_stub(out / "velocity_export.onnx", EXPORT["metadata_props"])
    expected = {}
    for name, (metadata, message) in BROKEN.items():
        export_stub(out / f"velocity_broken_{name}.onnx", metadata)
        expected[f"velocity_broken_{name}.onnx"] = message
    width = EXPORT["inputs"][0]["shape"][1]
    export_stub(out / "velocity_broken_input_width.onnx", EXPORT["metadata_props"], width=width + 1)
    expected["velocity_broken_input_width.onnx"] = "observation terms + depth are 5144"
    export_stub(out / "velocity_broken_action_width.onnx", EXPORT["metadata_props"], actions=28)
    expected["velocity_broken_action_width.onnx"] = "must be [1, 29], one action per joint"
    (out / "velocity_broken.json").write_text(json.dumps(expected, indent=1))


if __name__ == "__main__":
    main()
