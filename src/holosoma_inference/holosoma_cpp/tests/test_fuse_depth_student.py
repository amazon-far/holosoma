"""fuse_depth_student_onnx: a backbone + student pair becomes one contextual ONNX with the same actions."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

pytestmark = pytest.mark.no_sim

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fuse_depth_student_onnx.py"
JOINTS, LATENT, H, W = 3, 2, 2, 3
PROPRIO = 6 + 3 * JOINTS + 15


def _fuser():
    spec = importlib.util.spec_from_file_location("fuse_depth_student_onnx", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _model(nodes, inputs, outputs, initializers, metadata=None) -> onnx.ModelProto:
    model = helper.make_model(
        helper.make_graph(nodes, "g", inputs, outputs, initializers), opset_imports=[helper.make_opsetid("", 11)]
    )
    model.ir_version = 8  # as gen_contextual_models.py: loadable by older ONNX Runtime releases
    for key, value in (metadata or {}).items():
        model.metadata_props.add(key=key, value=value)
    return model


def _write_pair(
    tmp_path: Path, student_obs_dim: int = PROPRIO + LATENT, observation_names: str | None = None
) -> tuple[Path, Path]:
    rng = np.random.default_rng(0)
    backbone = _model(
        [
            helper.make_node("Reshape", ["depth_image", "flat_shape"], ["flat"]),
            helper.make_node("MatMul", ["flat", "w"], ["depth_latent"]),
        ],
        [helper.make_tensor_value_info("depth_image", TensorProto.FLOAT, [1, H, W])],
        [helper.make_tensor_value_info("depth_latent", TensorProto.FLOAT, [1, LATENT])],
        [
            numpy_helper.from_array(np.array([1, H * W], dtype=np.int64), "flat_shape"),
            numpy_helper.from_array(rng.standard_normal((H * W, LATENT)).astype(np.float32), "w"),
        ],
    )
    # A student that also takes time_step and has an extra output, like the exported students.
    student = _model(
        [
            helper.make_node("MatMul", ["obs", "w"], ["mm"]),
            helper.make_node("Add", ["mm", "time_step"], ["actions"]),
            helper.make_node("Identity", ["obs"], ["joint_pos"]),
        ],
        [
            helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, student_obs_dim]),
            helper.make_tensor_value_info("time_step", TensorProto.FLOAT, [1, 1]),
        ],
        [
            helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, JOINTS]),
            helper.make_tensor_value_info("joint_pos", TensorProto.FLOAT, [1, student_obs_dim]),
        ],
        [numpy_helper.from_array(rng.standard_normal((student_obs_dim, JOINTS)).astype(np.float32), "w")],
        metadata={
            "joint_names": "a,b,c",
            "action_scale": "1.0,1.0,1.0",
            **({"observation_names": observation_names} if observation_names else {}),
        },
    )
    paths = tmp_path / "depth_backbone.onnx", tmp_path / "student.onnx"
    onnx.save(backbone, paths[0])
    onnx.save(student, paths[1])
    return paths


def test_fused_model_reproduces_the_pair_on_the_contextual_contract(tmp_path: Path) -> None:
    fuser = _fuser()
    backbone_path, student_path = _write_pair(tmp_path)
    out = tmp_path / "contextual.onnx"

    onnx.save(fuser.fuse(onnx.load(backbone_path), onnx.load(student_path)), out)

    fused = onnx.load(out)
    shapes = {i.name: [d.dim_value for d in i.type.tensor_type.shape.dim] for i in fused.graph.input}
    assert shapes == {"obs": [1, PROPRIO], "depth_image": [1, 1, H, W]}
    assert [o.name for o in fused.graph.output] == ["actions"]
    metadata = {p.key: p.value for p in fused.metadata_props}
    assert metadata["joint_names"] == "a,b,c" and "fused_from" in metadata
    assert json.loads(metadata["action_scale"]) == [1.0, 1.0, 1.0]  # the kernel reads number lists as JSON
    assert fuser.check(backbone_path, student_path, out) == 0.0


def test_the_training_order_moves_the_last_action_first() -> None:
    order = _fuser().contextual_to_training_order(JOINTS)
    n = JOINTS
    assert order[:n].tolist() == list(range(6 + 2 * n, 6 + 3 * n))  # actions
    assert order[n : n + 3].tolist() == [3, 4, 5]  # angular velocity
    assert order[-15 - 3 : -15].tolist() == [0, 1, 2]  # projected gravity, before the command
    assert sorted(order.tolist()) == list(range(PROPRIO))


def test_a_student_whose_width_does_not_fit_is_rejected(tmp_path: Path) -> None:
    backbone_path, student_path = _write_pair(tmp_path, student_obs_dim=PROPRIO + LATENT + 1)

    with pytest.raises(ValueError, match="expected"):
        _fuser().fuse(onnx.load(backbone_path), onnx.load(student_path))


# The released PHP students record their terms in declaration order; current exports record them
# sorted. Both consume them sorted.
DECLARATION_ORDER = "robot_anchor_projected_gravity,base_ang_vel,dof_pos,dof_vel,actions,velocity_command,placeholder"
SORTED_ORDER = "actions,base_ang_vel,dof_pos,dof_vel,robot_anchor_projected_gravity,velocity_command,placeholder"


@pytest.mark.parametrize("names", [DECLARATION_ORDER, SORTED_ORDER])
def test_recorded_term_names_are_consumed_sorted(tmp_path: Path, names: str) -> None:
    fuser = _fuser()
    backbone_path, student_path = _write_pair(tmp_path, observation_names=names)

    fused = fuser.fuse(onnx.load(backbone_path), onnx.load(student_path))

    fused_from = json.loads({p.key: p.value for p in fused.metadata_props}["fused_from"])
    assert fused_from["student_obs_order"] == SORTED_ORDER.replace("placeholder", "depth_latent")
    assert fused_from["student_obs_order_source"] == "observation_names, sorted"
    terms = fuser.student_terms(onnx.load(student_path))[0]
    assert np.array_equal(
        fuser.contextual_to_training_order(JOINTS, terms=terms), fuser.contextual_to_training_order(JOINTS)
    )


def test_a_different_recorded_order_changes_the_gather(tmp_path: Path) -> None:
    fuser = _fuser()
    # Sorted, "a_projected_gravity" comes first: gravity, then actions, omega, q, dq, command.
    names = "a_projected_gravity,base_ang_vel,dof_pos,dof_vel,actions,velocity_command,placeholder"
    backbone_path, student_path = _write_pair(tmp_path, observation_names=names)
    out = tmp_path / "contextual.onnx"
    onnx.save(fuser.fuse(onnx.load(backbone_path), onnx.load(student_path)), out)

    n = JOINTS
    expected = [0, 1, 2, *range(6 + 2 * n, 6 + 3 * n), 3, 4, 5, *range(6, 6 + 2 * n), *range(6 + 3 * n, PROPRIO)]
    terms = fuser.student_terms(onnx.load(student_path))[0]
    assert fuser.contextual_to_training_order(n, terms=terms).tolist() == expected
    assert fuser.check(backbone_path, student_path, out) == 0.0


def test_without_term_names_the_usual_order_is_assumed_and_recorded(tmp_path: Path) -> None:
    fuser = _fuser()
    backbone_path, student_path = _write_pair(tmp_path)

    fused = fuser.fuse(onnx.load(backbone_path), onnx.load(student_path))

    fused_from = json.loads({p.key: p.value for p in fused.metadata_props}["fused_from"])
    assert fused_from["student_obs_order_source"].startswith("assumed")


@pytest.mark.parametrize(
    "names",
    [
        "base_ang_vel,dof_pos,dof_vel,actions,velocity_command,height_scan,placeholder",  # no contextual block
        "projected_gravity,base_ang_vel,dof_pos,dof_vel,velocity_command,placeholder",  # a block missing
    ],
)
def test_term_names_that_do_not_match_the_contextual_obs_are_rejected(tmp_path: Path, names: str) -> None:
    backbone_path, student_path = _write_pair(tmp_path, observation_names=names)

    with pytest.raises(ValueError, match="contextual"):
        _fuser().fuse(onnx.load(backbone_path), onnx.load(student_path))
