from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from holosoma.agents.distillation.distillation import Distillation
from holosoma.agents.distillation_ppo.distillation_ppo import DistillationMinibatch, DistillationPPO
from holosoma.agents.modules.module_utils import setup_depth_student_teacher_module, setup_student_teacher_module
from holosoma.agents.modules.student_teacher_modules import StudentTeacher
from holosoma.agents.ppo.ppo import EmpiricalNormalization
from holosoma.config_types.algo import DistillationPPOConfig, StudentTeacherModuleConfig
from holosoma.train_agent import infer_num_teachers_from_checkpoint
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim


@pytest.mark.parametrize("algo_cls", [Distillation, DistillationPPO])
@pytest.mark.parametrize(("start_iteration", "expected"), [(1, True), (5, False)])
def test_warmup_uses_absolute_iteration(
    algo_cls: type[Distillation | DistillationPPO],
    start_iteration: int,
    expected: bool,
) -> None:
    algo = cast("Any", object.__new__(algo_cls))
    algo.current_learning_iteration = start_iteration
    algo.config = SimpleNamespace(
        num_learning_iterations=1,
        distillation_warmup_steps=3,
        save_interval=100,
    )
    algo._train_mode = lambda: None
    algo.policy = SimpleNamespace(require_loaded_teachers=lambda: None)
    algo.env = SimpleNamespace(reset_all=dict)
    algo.logging_helper = SimpleNamespace(
        record_collection_time=nullcontext,
        record_learn_time=nullcontext,
    )
    observed: list[bool] = []

    def record_warmup(obs: dict[str, Any]) -> dict[str, Any]:
        observed.append(algo._warmup_active)
        return obs

    algo._rollout_step = record_warmup
    algo._training_step = dict
    algo.is_main_process = False
    algo.is_multi_gpu = False
    algo.log_dir = None
    if algo_cls is DistillationPPO:
        algo.adjust_ppo_dagger_coeff = lambda _iteration: None
        algo._set_std_lr_from_ppo_coef = lambda: None

    algo_cls.learn(algo, num_learning_iterations=1)

    assert observed == [expected]


def _stub_learn_loop(algo_cls: type[Distillation | DistillationPPO], events: list[tuple[Any, ...]]) -> Any:
    algo = cast("Any", object.__new__(algo_cls))
    algo.current_learning_iteration = 0
    algo.config = SimpleNamespace(num_learning_iterations=2, distillation_warmup_steps=0, save_interval=1)
    algo._train_mode = lambda: None
    algo.policy = SimpleNamespace(require_loaded_teachers=lambda: None)
    algo.logging_helper = SimpleNamespace(record_collection_time=nullcontext, record_learn_time=nullcontext)

    def rollout(obs: dict[str, Any]) -> dict[str, Any]:
        events.append(("rollout",))
        return obs

    algo._rollout_step = rollout
    algo._training_step = dict
    algo.log_dir = "/tmp/run"
    if algo_cls is DistillationPPO:
        algo.adjust_ppo_dagger_coeff = lambda _iteration: None
        algo._set_std_lr_from_ppo_coef = lambda: None
    return algo


@pytest.mark.parametrize("algo_cls", [Distillation, DistillationPPO])
def test_learn_synchronizes_curriculum_every_iteration_on_multi_gpu(
    algo_cls: type[Distillation | DistillationPPO],
) -> None:
    events: list[tuple[Any, ...]] = []
    algo = _stub_learn_loop(algo_cls, events)

    def synchronize_curriculum_state(*, device: str, world_size: int) -> None:
        del device
        events.append(("sync", world_size))

    algo.env = SimpleNamespace(
        reset_all=dict,
        use_reward_penalty_curriculum=True,
        synchronize_curriculum_state=synchronize_curriculum_state,
    )
    algo.device = "cpu"
    algo.is_multi_gpu = True
    algo.gpu_world_size = 2
    algo.is_main_process = False

    algo_cls.learn(algo)

    assert events == [("sync", 2), ("rollout",), ("sync", 2), ("rollout",)]


@pytest.mark.parametrize("algo_cls", [Distillation, DistillationPPO])
def test_learn_leaves_export_ownership_to_application_save_hook(algo_cls: type[Distillation | DistillationPPO]) -> None:
    events: list[tuple[Any, ...]] = []
    algo = _stub_learn_loop(algo_cls, events)
    algo.env = SimpleNamespace(reset_all=dict)
    algo.is_multi_gpu = False
    algo.is_main_process = True
    algo._post_epoch_logging = lambda _it, _losses: None
    algo.save = lambda path: events.append(("save", path))
    algo.export = lambda onnx_file_path: events.append(("export", onnx_file_path))

    algo_cls.learn(algo)

    saved = [event[1] for event in events if event[0] == "save"]
    exported = [event[1] for event in events if event[0] == "export"]
    assert saved == ["/tmp/run/model_00000.pt", "/tmp/run/model_00001.pt", "/tmp/run/model_00001.pt"]
    assert exported == []


class _TinyPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.student = torch.nn.Linear(2, 1, bias=False)
        self.critic = torch.nn.Linear(2, 1, bias=False)
        torch.nn.init.zeros_(self.student.weight)
        torch.nn.init.zeros_(self.critic.weight)

    def _student_input(self, actor_obs: torch.Tensor, _depth_obs: torch.Tensor) -> torch.Tensor:
        return actor_obs

    def project_action_std(self) -> None:
        pass


def test_warmup_update_is_dagger_only() -> None:
    algo = object.__new__(DistillationPPO)
    algo.policy = _TinyPolicy()
    algo.config = cast("DistillationPPOConfig", SimpleNamespace(dagger_loss_coef=2.0, max_grad_norm=10.0))
    algo.distill_loss_fn = torch.nn.functional.mse_loss
    algo.optimizer = torch.optim.SGD(algo.policy.parameters(), lr=0.1)
    algo.is_multi_gpu = False
    algo._warmup_active = True
    critic_before = algo.policy.critic.weight.detach().clone()

    losses = algo._update_step(
        cast(
            "DistillationMinibatch",
            {
                "actor_obs": torch.tensor([[1.0, 1.0]]),
                "depth_obs": torch.zeros(1, 1),
                "critic_obs": torch.tensor([[1.0, 1.0]]),
                "actions": torch.zeros(1, 1),
                "teacher_actions": torch.ones(1, 1),
            },
        )
    )

    assert not torch.equal(algo.policy.student.weight, torch.zeros_like(algo.policy.student.weight))
    torch.testing.assert_close(algo.policy.critic.weight, critic_before)
    assert losses["value_loss"].item() == 0.0
    assert losses["surrogate_loss"].item() == 0.0


def test_global_advantage_normalization_handles_uneven_shards(monkeypatch: pytest.MonkeyPatch) -> None:
    algo = object.__new__(DistillationPPO)
    local = torch.tensor([1.0, 2.0, 3.0])
    remote = torch.tensor([10.0, 20.0], dtype=torch.float64)

    def fake_all_reduce(stats: torch.Tensor, op: object) -> None:
        del op
        stats += torch.stack((remote.sum(), remote.square().sum(), remote.new_tensor(remote.numel())))

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    normalized = algo._normalize_advantages_multi_gpu(local)
    combined = torch.cat((local, remote.to(local.dtype)))
    expected = (local - combined.mean()) / (combined.std() + 1e-8)
    torch.testing.assert_close(normalized, expected)


def test_teacher_count_is_inferred_from_legacy_state_dict() -> None:
    checkpoint = {
        "model_state_dict": {
            "teachers.0.0.weight": torch.zeros(1),
            "teachers.1.0.weight": torch.zeros(1),
        }
    }

    assert infer_num_teachers_from_checkpoint(checkpoint) == 2


def test_teacher_count_is_inferred_from_single_teacher_state_dict() -> None:
    checkpoint = {"model_state_dict": {"teacher.0.weight": torch.zeros(1)}}

    assert infer_num_teachers_from_checkpoint(checkpoint) == 1


def test_mixed_teacher_key_formats_are_rejected() -> None:
    checkpoint = {
        "model_state_dict": {
            "teacher.0.weight": torch.zeros(1),
            "teachers.0.0.weight": torch.zeros(1),
        }
    }

    with pytest.raises(ValueError, match="mixes legacy"):
        infer_num_teachers_from_checkpoint(checkpoint)


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "2"])
def test_invalid_explicit_teacher_count_is_rejected(value: object) -> None:
    with pytest.raises(ValueError, match="Invalid num_teachers"):
        infer_num_teachers_from_checkpoint({"num_teachers": value})


def test_teacher_count_metadata_must_match_state_dict() -> None:
    checkpoint = {
        "num_teachers": 1,
        "model_state_dict": {
            "teachers.0.0.weight": torch.zeros(1),
            "teachers.1.0.weight": torch.zeros(1),
        },
    }

    with pytest.raises(ValueError, match="contains 2 teacher slots"):
        infer_num_teachers_from_checkpoint(checkpoint)


def test_non_contiguous_teacher_slots_are_rejected() -> None:
    checkpoint = {
        "model_state_dict": {
            "teachers.0.0.weight": torch.zeros(1),
            "teachers.2.0.weight": torch.zeros(1),
        }
    }

    with pytest.raises(ValueError, match="non-contiguous"):
        infer_num_teachers_from_checkpoint(checkpoint)


@pytest.mark.parametrize("algo_cls", [Distillation, DistillationPPO])
def test_legacy_checkpoint_does_not_certify_teacher_weights(
    algo_cls: type[Distillation | DistillationPPO],
    tmp_path: Path,
) -> None:
    policy = StudentTeacher(
        num_actor_obs=2,
        num_teacher_obs=3,
        num_actions=1,
        student_hidden_dims=[2],
        teacher_hidden_dims=[2],
    )
    checkpoint = tmp_path / "legacy.pt"
    torch.save({"model_state_dict": policy.state_dict()}, checkpoint)

    algo = cast("Any", object.__new__(algo_cls))
    algo.device = "cpu"
    algo.policy = policy
    algo._restore_env_state = lambda _state: None
    algo_cls.load(algo, str(checkpoint))

    assert policy.loaded_teachers == (False,)
    with pytest.raises(RuntimeError, match="not loaded"):
        policy.require_loaded_teachers()


def _write_ppo_teacher_checkpoint(
    path: Path, *, normalize: bool
) -> tuple[torch.nn.Module, EmpiricalNormalization | None]:
    """Write a holosoma PPO checkpoint for a 3-obs, 2-action teacher."""
    torch.manual_seed(0)
    teacher = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ELU(), torch.nn.Linear(4, 2))
    actor_state = {f"actor_module.module.{key}": value for key, value in teacher.state_dict().items()}
    actor_state["std"] = torch.ones(2)
    normalizer = None
    if normalize:
        normalizer = EmpiricalNormalization(shape=3, device="cpu")
        normalizer(torch.randn(64, 3) * torch.tensor([0.5, 2.0, 4.0]) + torch.tensor([1.0, -3.0, 10.0]))
        normalizer.eval()
    torch.save(
        {
            "actor_model_state_dict": actor_state,
            "actor_obs_normalizer_state_dict": normalizer.state_dict() if normalizer is not None else None,
            "iter": 100,
        },
        path,
    )
    return teacher, normalizer


def _checkpointable_algo(algo_cls: type[Distillation | DistillationPPO], log_dir: Path) -> Any:
    algo = cast("Any", object.__new__(algo_cls))
    algo.device = "cpu"
    algo.policy = StudentTeacher(
        num_actor_obs=2,
        num_teacher_obs=3,
        num_actions=2,
        student_hidden_dims=[4],
        teacher_hidden_dims=[4],
    )
    algo.optimizer = torch.optim.Adam(algo.policy.student.parameters())
    algo.current_learning_iteration = 0
    algo.learning_rate = 1e-3
    algo.ppo_coef = 0.0
    algo.log_dir = str(log_dir)
    algo._checkpoint_metadata = lambda **_kwargs: {}
    algo._collect_env_state = dict
    algo._restore_env_state = lambda _state: None
    algo.logging_helper = SimpleNamespace(save_checkpoint_artifact=lambda state, path: torch.save(state, path))
    return algo


@pytest.mark.parametrize("algo_cls", [Distillation, DistillationPPO])
@pytest.mark.parametrize("normalize", [True, False])
def test_teacher_runs_on_its_ppo_normalized_observations(
    algo_cls: type[Distillation | DistillationPPO],
    normalize: bool,
    tmp_path: Path,
) -> None:
    teacher, normalizer = _write_ppo_teacher_checkpoint(tmp_path / "teacher.pt", normalize=normalize)
    algo = _checkpointable_algo(algo_cls, tmp_path)

    algo_cls.load_teacher(algo, str(tmp_path / "teacher.pt"))

    obs = torch.randn(5, 3) * 3.0
    teacher_input = normalizer(obs, update=False) if normalizer is not None else obs
    torch.testing.assert_close(algo.policy.teacher_act(obs), teacher(teacher_input), rtol=0.0, atol=0.0)


@pytest.mark.parametrize("algo_cls", [Distillation, DistillationPPO])
def test_teacher_normalizer_round_trips_through_full_checkpoint(
    algo_cls: type[Distillation | DistillationPPO],
    tmp_path: Path,
) -> None:
    teacher, normalizer = _write_ppo_teacher_checkpoint(tmp_path / "teacher.pt", normalize=True)
    assert normalizer is not None
    source = _checkpointable_algo(algo_cls, tmp_path)
    algo_cls.load_teacher(source, str(tmp_path / "teacher.pt"))
    algo_cls.save(source, str(tmp_path / "model_00000.pt"))

    resumed = _checkpointable_algo(algo_cls, tmp_path)
    algo_cls.load(resumed, str(tmp_path / "model_00000.pt"))

    obs = torch.randn(5, 3) * 3.0
    assert resumed.policy.loaded_teachers == (True,)
    torch.testing.assert_close(
        resumed.policy.teacher_act(obs), teacher(normalizer(obs, update=False)), rtol=0.0, atol=0.0
    )


@pytest.mark.parametrize("algo_cls", [Distillation, DistillationPPO])
def test_export_writes_the_onnx_pair_the_depth_policy_loads(
    algo_cls: type[Distillation | DistillationPPO],
    tmp_path: Path,
) -> None:
    np = pytest.importorskip("numpy")
    depth_distillation = pytest.importorskip("holosoma_inference.policies.depth_distillation")

    dof_names = ["left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint"]
    module_config = StudentTeacherModuleConfig(
        student_hidden_dims=[8], teacher_hidden_dims=[8], critic_hidden_dims=[8], depth_output_dim=4
    )
    algo = cast("Any", object.__new__(algo_cls))
    if algo_cls is DistillationPPO:
        algo.policy = setup_student_teacher_module(
            num_actor_obs=5,
            num_teacher_obs=3,
            num_critic_obs=3,
            num_actions=3,
            module_config=module_config,
            device="cpu",
            init_noise_std=0.1,
        )
    else:
        algo.policy = setup_depth_student_teacher_module(
            num_actor_obs=5,
            num_teacher_obs=3,
            num_actions=3,
            module_config=module_config,
            device="cpu",
            init_noise_std=0.1,
        )
    algo.policy.train()
    urdf = tmp_path / "robot.urdf"
    urdf.write_text("<robot name='g1'/>")
    algo.env = SimpleNamespace(
        command_manager=None,
        robot_config=SimpleNamespace(
            dof_names=dof_names,
            control=SimpleNamespace(
                stiffness={"hip_pitch": 100.0, "waist_yaw": 200.0},
                damping={"hip_pitch": 2.0, "waist_yaw": 5.0},
                action_scale=0.25,
            ),
            asset=SimpleNamespace(urdf_file=urdf.name, asset_root=str(tmp_path)),
        ),
    )
    algo.device = "cpu"
    algo.depth_shape = (58, 87)
    algo.actor_obs_dim = 5
    algo.current_learning_iteration = 7
    algo._experiment_config = SimpleNamespace(to_serializable_dict=dict)
    algo._wandb_run_path = None
    uploaded: list[str] = []
    algo.logging_helper = SimpleNamespace(save_to_wandb=uploaded.append)

    algo_cls.export(algo, onnx_file_path=str(tmp_path / "exported" / "model_00007.onnx"))

    export_dir = tmp_path / "exported" / "model_00007"
    assert uploaded == [str(export_dir / "depth_backbone.onnx"), str(export_dir / "student.onnx")]
    assert algo.policy.training
    assert not any(teacher.training for teacher in algo.policy.teachers)

    # Load the pair through the inference policy's own loaders.
    deployed = object.__new__(depth_distillation.DepthDistillationPolicy)
    deployed.config = SimpleNamespace(
        camera=SimpleNamespace(props=SimpleNamespace(resized_height=58, resized_width=87))
    )
    deployed.num_dofs = 3
    deployed.robot_config = SimpleNamespace(dof_names=dof_names)
    deployed.policy_action_scale = 1.0
    deployed._load_depth_backbone(str(export_dir / "depth_backbone.onnx"))
    deployed._load_student_model(str(export_dir / "student.onnx"))

    assert (deployed.depth_buffer_len, deployed.depth_image_shape, deployed.depth_latent_dim) == (1, (58, 87), 4)
    assert deployed.onnx_input_names == ["obs"]
    assert deployed._model_joint_names == dof_names
    assert deployed.policy_action_scale == 0.25
    np.testing.assert_allclose(deployed.onnx_kp, [100.0, 100.0, 200.0])
    np.testing.assert_allclose(deployed.onnx_kd, [2.0, 2.0, 5.0])

    # The composed pair reproduces the student's deterministic action.
    depth = torch.rand(1, 58, 87) - 0.5
    actor_obs = torch.randn(1, 5)
    latent = deployed.depth_backbone_session.run(
        [deployed.depth_backbone_output_name], {deployed.depth_backbone_input_name: depth.numpy()}
    )[0]
    actions = deployed.policy({"obs": np.concatenate([actor_obs.numpy(), latent], axis=1)})[0]
    with torch.no_grad():
        expected = algo.policy.act_inference({"actor_obs": actor_obs, "depth_obs": depth})
    np.testing.assert_allclose(actions, expected.numpy(), rtol=1e-5, atol=1e-5)
