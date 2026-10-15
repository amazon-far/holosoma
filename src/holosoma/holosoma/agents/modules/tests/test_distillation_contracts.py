from __future__ import annotations

import copy
import json
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from holosoma.agents.distillation.base import DistillationBase
from holosoma.agents.distillation.distillation import Distillation
from holosoma.agents.distillation_ppo.distillation_ppo import DistillationMinibatch, DistillationPPO
from holosoma.agents.modules.module_utils import setup_depth_student_teacher_module, setup_student_teacher_module
from holosoma.agents.modules.student_teacher_modules import StudentTeacher
from holosoma.agents.modules.teacher_policies import MLPTeacher, PPOTeacher
from holosoma.config_types.algo import DistillationConfig, DistillationPPOConfig, StudentTeacherModuleConfig
from holosoma.train_agent import infer_num_teachers_from_checkpoint
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim

DEPTH_BACKBONE = "holosoma.agents.modules.depth_backbone.DepthOnlyFCBackbone58x87Small"


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
                "teacher_valid": torch.ones(1, 1, dtype=torch.bool),
            },
        )
    )

    assert not torch.equal(algo.policy.student.weight, torch.zeros_like(algo.policy.student.weight))
    torch.testing.assert_close(algo.policy.critic.weight, critic_before)
    assert losses["value_loss"].item() == 0.0
    assert losses["surrogate_loss"].item() == 0.0


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
    policy = StudentTeacher(num_actor_obs=2, num_teacher_obs=3, num_actions=1, student_hidden_dims=[2])
    legacy_teacher = MLPTeacher(num_obs=3, hidden_dims=[2], num_actions=1, activation="ELU")
    legacy_state = {
        **policy.state_dict(),
        **{f"teachers.0.{key}": value for key, value in legacy_teacher.mlp.state_dict().items()},
    }
    checkpoint = tmp_path / "legacy.pt"
    torch.save({"model_state_dict": legacy_state}, checkpoint)

    algo = cast("Any", object.__new__(algo_cls))
    algo.device = "cpu"
    algo.config = SimpleNamespace(module=SimpleNamespace(teacher_hidden_dims=[2], activation="ELU"))
    algo.policy = policy
    algo._restore_env_state = lambda _state: None
    algo_cls.load(algo, str(checkpoint))

    assert policy.loaded_teachers == (False,)
    with pytest.raises(RuntimeError, match="not loaded"):
        policy.require_loaded_teachers()


def _checkpointable_algo(algo_cls: type[Distillation | DistillationPPO], log_dir: Path) -> Any:
    algo = cast("Any", object.__new__(algo_cls))
    algo.device = "cpu"
    algo.policy = StudentTeacher(
        num_actor_obs=2,
        num_teacher_obs=3,
        num_actions=2,
        student_hidden_dims=[4],
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
    make_ppo_teacher_checkpoint: Callable[..., Any],
    make_fitted_normalizer: Callable[..., Any],
) -> None:
    normalizer = make_fitted_normalizer(3) if normalize else None
    checkpoint, actor = make_ppo_teacher_checkpoint(3, 2, [4], normalizer=normalizer)
    torch.save(checkpoint, tmp_path / "teacher.pt")
    algo = _checkpointable_algo(algo_cls, tmp_path)

    algo_cls.load_teacher(algo, str(tmp_path / "teacher.pt"))

    obs = torch.randn(5, 3) * 3.0
    teacher_input = normalizer(obs, update=False) if normalizer is not None else obs
    expected = actor.act_inference({"actor_obs": teacher_input})
    torch.testing.assert_close(algo.policy.teacher_act(obs).actions, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("algo_cls", [Distillation, DistillationPPO])
def test_teacher_round_trips_through_full_checkpoint(
    algo_cls: type[Distillation | DistillationPPO],
    tmp_path: Path,
    make_ppo_teacher_checkpoint: Callable[..., Any],
    make_fitted_normalizer: Callable[..., Any],
) -> None:
    normalizer = make_fitted_normalizer(3)
    checkpoint, actor = make_ppo_teacher_checkpoint(3, 2, [4], normalizer=normalizer)
    torch.save(checkpoint, tmp_path / "teacher.pt")
    source = _checkpointable_algo(algo_cls, tmp_path)
    algo_cls.load_teacher(source, str(tmp_path / "teacher.pt"))
    algo_cls.save(source, str(tmp_path / "model_00000.pt"))

    resumed = _checkpointable_algo(algo_cls, tmp_path)
    algo_cls.load(resumed, str(tmp_path / "model_00000.pt"))

    # The resumed policy rebuilt the PPO teacher from the saved construction config alone.
    obs = torch.randn(5, 3) * 3.0
    assert resumed.policy.loaded_teachers == (True,)
    assert isinstance(resumed.policy.teachers[0], PPOTeacher)
    expected = actor.act_inference({"actor_obs": normalizer(obs, update=False)})
    torch.testing.assert_close(resumed.policy.teacher_act(obs).actions, expected, rtol=0.0, atol=0.0)


def _export_ready_algo(
    algo_cls: type[Distillation | DistillationPPO], tmp_path: Path, depth_backbone: str | None
) -> Any:
    module_config = StudentTeacherModuleConfig(
        student_hidden_dims=[8],
        teacher_hidden_dims=[8],
        critic_hidden_dims=[8],
        depth_backbone=depth_backbone,
        depth_output_dim=4,
    )
    algo = cast("Any", object.__new__(algo_cls))
    algo.config = SimpleNamespace(module=module_config)
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
            dof_names=["left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint"],
            control=SimpleNamespace(
                stiffness={"hip_pitch": 100.0, "waist_yaw": 200.0},
                damping={"hip_pitch": 2.0, "waist_yaw": 5.0},
                action_scale=0.25,
            ),
            asset=SimpleNamespace(urdf_file=urdf.name, asset_root=str(tmp_path)),
        ),
    )
    algo.device = "cpu"
    algo.depth_shape = (58, 87) if depth_backbone is not None else None
    algo.actor_obs_dim = 5
    algo.current_learning_iteration = 7
    algo._experiment_config = SimpleNamespace(to_serializable_dict=dict)
    algo._wandb_run_path = None
    algo.uploaded = []
    algo.logging_helper = SimpleNamespace(save_to_wandb=algo.uploaded.append)
    return algo


@pytest.mark.parametrize("algo_cls", [Distillation, DistillationPPO])
def test_export_writes_a_depth_backbone_and_student_pair(
    algo_cls: type[Distillation | DistillationPPO],
    tmp_path: Path,
) -> None:
    ort = pytest.importorskip("onnxruntime")
    algo = _export_ready_algo(algo_cls, tmp_path, depth_backbone=DEPTH_BACKBONE)

    algo_cls.export(algo, onnx_file_path=str(tmp_path / "exported" / "model_00007.onnx"))

    export_dir = tmp_path / "exported" / "model_00007"
    assert algo.uploaded == [str(export_dir / "depth_backbone.onnx"), str(export_dir / "student.onnx")]
    assert algo.policy.training
    assert not any(teacher.training for teacher in algo.policy.teachers)

    backbone = ort.InferenceSession(str(export_dir / "depth_backbone.onnx"))
    student = ort.InferenceSession(str(export_dir / "student.onnx"))
    assert [i.name for i in backbone.get_inputs()] == ["depth_image"]
    assert [i.name for i in student.get_inputs()] == ["obs"]
    metadata = student.get_modelmeta().custom_metadata_map
    assert json.loads(metadata["joint_names"]) == algo.env.robot_config.dof_names
    assert json.loads(metadata["kp"]) == [100.0, 100.0, 200.0]
    assert json.loads(metadata["action_scale"]) == 0.25

    # The composed pair reproduces the student's deterministic action.
    depth = torch.rand(1, 58, 87) - 0.5
    actor_obs = torch.randn(1, 5)
    (latent,) = backbone.run(None, {"depth_image": depth.numpy()})
    (actions,) = student.run(None, {"obs": torch.cat([actor_obs, torch.from_numpy(latent)], dim=1).numpy()})
    with torch.no_grad():
        expected = algo.policy.act_inference({"actor_obs": actor_obs, "depth_obs": depth})
    torch.testing.assert_close(torch.from_numpy(actions), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("algo_cls", [Distillation, DistillationPPO])
def test_export_writes_a_single_policy_file_without_depth(
    algo_cls: type[Distillation | DistillationPPO],
    tmp_path: Path,
) -> None:
    ort = pytest.importorskip("onnxruntime")
    algo = _export_ready_algo(algo_cls, tmp_path, depth_backbone=None)
    onnx_path = tmp_path / "exported" / "model_00007.onnx"

    algo_cls.export(algo, onnx_file_path=str(onnx_path))

    assert algo.uploaded == [str(onnx_path)]
    session = ort.InferenceSession(str(onnx_path))
    # Same input/output names as PPO.export.
    assert [i.name for i in session.get_inputs()] == ["actor_obs"]
    assert [o.name for o in session.get_outputs()] == ["action"]
    actor_obs = torch.randn(1, 5)
    (actions,) = session.run(None, {"actor_obs": actor_obs.numpy()})
    with torch.no_grad():
        expected = algo.policy.act_inference({"actor_obs": actor_obs})
    torch.testing.assert_close(torch.from_numpy(actions), expected, rtol=1e-5, atol=1e-5)


class _FakeObservationManager:
    def __init__(self, groups: dict[str, Any]) -> None:
        self.groups = groups

    def get_obs_dims(self) -> dict[str, Any]:
        return {
            name: {term: value.shape[1:].numel() for term, value in group.items()}
            if isinstance(group, dict)
            else group.shape[1]
            for name, group in self.groups.items()
        }

    def compute_group(self, name: str, *, modify_history: bool = True) -> Any:
        assert not modify_history
        return self.groups[name]


class _FakeEnv:
    """Random-walk env exposing whatever observation groups it is given."""

    num_envs = 4

    def __init__(self, depth_group: tuple[str, str] | None = None) -> None:
        self.depth_group = depth_group
        self.robot_config = SimpleNamespace(actions_dim=2)
        self.command_manager = SimpleNamespace(get_state=lambda _name: None)
        self.observation_manager = _FakeObservationManager(self._obs())

    def _obs(self) -> dict[str, Any]:
        obs: dict[str, Any] = {
            "actor_obs": torch.randn(self.num_envs, 3),
            "teacher_obs": torch.randn(self.num_envs, 3),
            "critic_obs": torch.randn(self.num_envs, 3),
        }
        if self.depth_group is not None:
            group, term = self.depth_group
            obs[group] = {term: torch.rand(self.num_envs, 58, 87)}
        return obs

    def reset_all(self) -> dict[str, Any]:
        return self._obs()

    def step(self, _actor_state: dict[str, Any]) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, dict[str, Any]]:
        obs = self._obs()
        return obs, torch.randn(self.num_envs), torch.zeros(self.num_envs, dtype=torch.bool), {}

    def get_checkpoint_state(self) -> dict[str, Any]:
        return {}


def _build_algo(
    algo_cls: type[Distillation | DistillationPPO],
    env: _FakeEnv,
    tmp_path: Path,
    before_setup: Callable[[Any], None] | None = None,
    **module_overrides: Any,
) -> Any:
    module_overrides.setdefault("depth_backbone", None)
    module = StudentTeacherModuleConfig(
        student_hidden_dims=[8], teacher_hidden_dims=[4], critic_hidden_dims=[8], **module_overrides
    )
    config_cls: Any = DistillationPPOConfig if algo_cls is DistillationPPO else DistillationConfig
    config = config_cls(module=module, num_steps_per_env=3, num_mini_batches=2, num_learning_iterations=2)
    algo = cast("Any", algo_cls(env=cast("Any", env), config=config, log_dir=str(tmp_path), device="cpu"))
    algo.logging_helper = SimpleNamespace(
        record_collection_time=nullcontext,
        record_learn_time=nullcontext,
        update_episode_stats=lambda *_args: None,
        post_epoch_logging=lambda **_kwargs: None,
        save_checkpoint_artifact=lambda state, path: torch.save(state, path),
    )
    algo._checkpoint_metadata = lambda **_kwargs: {}
    if before_setup is not None:
        before_setup(algo)
    algo.setup()
    return algo


def _load_random_teacher(algo: Any, tmp_path: Path, make_checkpoint: Callable[..., Any]) -> None:
    checkpoint, _ = make_checkpoint(3, 2, [4])
    path = tmp_path / "teacher.pt"
    torch.save(checkpoint, path)
    algo.load_teacher(str(path))


@pytest.mark.parametrize("algo_cls", [Distillation, DistillationPPO])
def test_student_without_depth_backbone_trains_on_proprioception_only(
    algo_cls: type[Distillation | DistillationPPO], tmp_path: Path, make_ppo_teacher_checkpoint: Callable[..., Any]
) -> None:
    algo = _build_algo(algo_cls, _FakeEnv(depth_group=None), tmp_path)
    _load_random_teacher(algo, tmp_path, make_ppo_teacher_checkpoint)
    before = copy.deepcopy(algo.policy.student.state_dict())

    algo.learn()

    assert not hasattr(algo.policy, "depth_backbone")
    assert algo.depth_shape is None
    assert "depth_obs" not in algo.storage._buffers
    assert any(not torch.equal(before[k], v) for k, v in algo.policy.student.state_dict().items())
    assert (tmp_path / "model_00001.pt").exists()


@pytest.mark.parametrize("algo_cls", [Distillation, DistillationPPO])
def test_depth_group_is_required_only_with_a_depth_backbone(
    algo_cls: type[Distillation | DistillationPPO], tmp_path: Path
) -> None:
    with pytest.raises(KeyError, match="depth_camera"):
        _build_algo(algo_cls, _FakeEnv(depth_group=None), tmp_path, depth_backbone=DEPTH_BACKBONE)


@pytest.mark.parametrize("algo_cls", [Distillation, DistillationPPO])
def test_depth_observation_group_and_term_are_configurable(
    algo_cls: type[Distillation | DistillationPPO], tmp_path: Path, make_ppo_teacher_checkpoint: Callable[..., Any]
) -> None:
    algo = _build_algo(
        algo_cls,
        _FakeEnv(depth_group=("head_camera", "depth")),
        tmp_path,
        depth_backbone=DEPTH_BACKBONE,
        depth_obs_group="head_camera",
        depth_obs_term="depth",
    )
    _load_random_teacher(algo, tmp_path, make_ppo_teacher_checkpoint)

    algo.learn()

    assert algo.depth_shape == (58, 87)
    assert algo.storage["depth_obs"].shape[2:] == (58, 87)


# ------------------------------------------------------- learning-iteration counter


@pytest.mark.parametrize("algo_cls", [Distillation, DistillationPPO])
def test_learning_in_chunks_continues_after_the_completed_iteration(
    algo_cls: type[Distillation | DistillationPPO],
) -> None:
    algo = _stub_learn_loop(algo_cls, [])
    algo.config.distillation_warmup_steps = 1
    algo.env = SimpleNamespace(reset_all=dict)
    algo.is_multi_gpu = False
    algo.is_main_process = True
    algo._post_epoch_logging = lambda _it, _losses: None
    saved: list[str] = []
    algo.save = saved.append
    warmup: list[bool] = []

    def rollout(obs: dict[str, Any]) -> dict[str, Any]:
        warmup.append(algo._warmup_active)
        return obs

    algo._rollout_step = rollout

    algo_cls.learn(algo, num_learning_iterations=1)
    algo_cls.learn(algo, num_learning_iterations=1)

    # The second call runs iteration 1, which is past the one-iteration warmup.
    assert warmup == [True, False]
    assert algo.current_learning_iteration == 2
    assert saved == [f"/tmp/run/model_0000{i}.pt" for i in (0, 0, 1, 1)]


@pytest.mark.parametrize("algo_cls", [Distillation, DistillationPPO])
def test_checkpoint_records_the_last_completed_iteration_and_resumes_after_it(
    algo_cls: type[Distillation | DistillationPPO], tmp_path: Path
) -> None:
    source = _checkpointable_algo(algo_cls, tmp_path)
    source.current_learning_iteration = 5  # iterations 0-4 are complete
    algo_cls.save(source, str(tmp_path / "model_00004.pt"))

    resumed = _checkpointable_algo(algo_cls, tmp_path)
    algo_cls.load(resumed, str(tmp_path / "model_00004.pt"))

    assert torch.load(tmp_path / "model_00004.pt")["iter"] == 4
    assert resumed.current_learning_iteration == 5


# ------------------------------------------------------------ gated adaptive KL


def _diagonal_gaussian_kl(old_mu: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    # Unit sigmas, as in the test inputs below.
    return torch.sum(torch.log(torch.tensor(1.0 + 1.0e-5)) + (1.0 + (old_mu - mu) ** 2) / 2.0 - 0.5, dim=-1)


def test_gated_kl_is_the_weighted_mean_over_all_ranks(monkeypatch: pytest.MonkeyPatch) -> None:
    algo = cast("Any", object.__new__(DistillationPPO))
    algo.is_multi_gpu = True
    algo.gpu_world_size = 2
    local_old_mu, local_mu, local_w = torch.zeros(2, 1), torch.tensor([[0.1], [0.3]]), torch.tensor([0.5, 0.5])
    remote_old_mu, remote_mu, remote_w = torch.zeros(2, 1), torch.tensor([[0.2], [0.0]]), torch.tensor([3.0, 0.0])
    remote_kl = _diagonal_gaussian_kl(remote_old_mu, remote_mu)

    def fake_all_reduce(stats: torch.Tensor, op: object) -> None:
        del op
        stats += torch.stack(((remote_kl * remote_w).sum(), remote_w.sum()))

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    ones = torch.ones(2, 1)

    kl = algo._compute_kl_div(local_old_mu, ones, local_mu, ones, weights=local_w)

    kl_all = torch.cat((_diagonal_gaussian_kl(local_old_mu, local_mu), remote_kl))
    weights_all = torch.cat((local_w, remote_w))
    torch.testing.assert_close(kl, (kl_all * weights_all).sum() / weights_all.sum())


# ------------------------------------------------------------ schedule constants


def test_schedule_constants_default_to_the_reference_values_and_are_configurable() -> None:
    module = StudentTeacherModuleConfig(student_hidden_dims=[4], teacher_hidden_dims=[4], depth_backbone=None)
    defaults = DistillationPPOConfig(module=module)
    assert (defaults.max_ppo_coef, defaults.ppo_active_threshold) == (0.9, 0.1)
    assert (defaults.min_learning_rate, defaults.max_learning_rate) == (1e-5, 1e-2)

    algo = cast("Any", object.__new__(DistillationPPO))
    algo.config = DistillationPPOConfig(
        module=module, ppo_start_epoch=0, dagger_end_epoch=10, max_ppo_coef=1.0, min_learning_rate=9e-4
    )
    algo.adjust_ppo_dagger_coeff(20)
    assert algo.ppo_coef == 1.0

    algo.is_multi_gpu = False
    algo.learning_rate = 1e-3
    algo.optimizer = SimpleNamespace(param_groups=[{"lr": 1e-3} for _ in range(4)])
    algo._STD_GROUP_IDX = 3
    algo._update_learning_rate(torch.tensor(1.0))  # far above desired_kl: decay, floored
    assert algo.learning_rate == 9e-4


# ------------------------------------------------------------- evaluation inputs


@pytest.mark.parametrize("algo_cls", [Distillation, DistillationPPO])
def test_student_evaluation_needs_only_the_student_inputs(
    algo_cls: type[Distillation | DistillationPPO], tmp_path: Path, make_ppo_teacher_checkpoint: Callable[..., Any]
) -> None:
    algo = _build_algo(algo_cls, _FakeEnv(depth_group=None), tmp_path)
    _load_random_teacher(algo, tmp_path, make_ppo_teacher_checkpoint)
    student_obs = {"actor_obs": torch.randn(_FakeEnv.num_envs, 3)}

    state = algo._pre_eval_env_step({"obs": student_obs}, algo.get_inference_policy())

    assert set(algo._extract_student_obs(student_obs)) == {"actor_obs"}
    assert state["actions"].shape == (_FakeEnv.num_envs, 2)


# ------------------------------------------------- multi-teacher checkpoint loading


@pytest.mark.parametrize("algo_cls", [Distillation, DistillationPPO])
def test_evaluation_order_restores_a_multi_teacher_checkpoint(
    algo_cls: type[Distillation | DistillationPPO], tmp_path: Path
) -> None:
    def two_teachers(algo: Any) -> None:
        algo.num_teachers = 2

    source = _build_algo(algo_cls, _FakeEnv(depth_group=None), tmp_path, before_setup=two_teachers)
    source.save(str(tmp_path / "model.pt"))
    checkpoint = torch.load(tmp_path / "model.pt")

    def restore(algo: Any) -> None:
        algo.restore_structure_from_checkpoint(checkpoint)

    # Eval entrypoint order: construct, restore the saved structure, setup(), load().
    evaluated = _build_algo(algo_cls, _FakeEnv(depth_group=None), tmp_path, before_setup=restore)
    evaluated.load(str(tmp_path / "model.pt"))

    assert evaluated.policy.num_teachers == 2


# ------------------------------------------------------------ export input prep


def _with_policy_override(algo: Any, name: str, method: Callable[..., torch.Tensor]) -> None:
    algo.policy.__class__ = type("CustomPolicy", (type(algo.policy),), {name: method})


@pytest.mark.parametrize("algo_cls", [Distillation, DistillationPPO])
def test_export_includes_the_policy_student_input_preparation(
    algo_cls: type[Distillation | DistillationPPO], tmp_path: Path
) -> None:
    ort = pytest.importorskip("onnxruntime")
    algo = _export_ready_algo(algo_cls, tmp_path, depth_backbone=DEPTH_BACKBONE)
    base = type(algo.policy)._combine_student_input

    def normalized(self: Any, actor_obs: torch.Tensor, depth_latent: torch.Tensor) -> torch.Tensor:
        return base(self, (actor_obs - 1.0) / 2.0, depth_latent)

    _with_policy_override(algo, "_combine_student_input", normalized)

    algo_cls.export(algo, onnx_file_path=str(tmp_path / "model_00007.onnx"))

    backbone = ort.InferenceSession(str(tmp_path / "model_00007" / "depth_backbone.onnx"))
    student = ort.InferenceSession(str(tmp_path / "model_00007" / "student.onnx"))
    depth, actor_obs = torch.rand(1, 58, 87) - 0.5, torch.randn(1, 5)
    (latent,) = backbone.run(None, {"depth_image": depth.numpy()})
    (actions,) = student.run(None, {"obs": torch.cat([actor_obs, torch.from_numpy(latent)], dim=1).numpy()})
    with torch.no_grad():
        expected = algo.policy.act_inference({"actor_obs": actor_obs, "depth_obs": depth})
    torch.testing.assert_close(torch.from_numpy(actions), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("algo_cls", [Distillation, DistillationPPO])
def test_export_rejects_input_preparation_it_cannot_reproduce(
    algo_cls: type[Distillation | DistillationPPO], tmp_path: Path
) -> None:
    algo = _export_ready_algo(algo_cls, tmp_path, depth_backbone=DEPTH_BACKBONE)
    base = type(algo.policy)._student_input

    def normalized(self: Any, actor_obs: torch.Tensor, depth_obs: torch.Tensor | None = None) -> torch.Tensor:
        return base(self, (actor_obs - 1.0) / 2.0, depth_obs)

    _with_policy_override(algo, "_student_input", normalized)

    with pytest.raises(RuntimeError, match="export_modules"):
        algo_cls.export(algo, onnx_file_path=str(tmp_path / "model_00007.onnx"))


# ------------------------------------------------------ teacher validity mask


def test_dagger_loss_ignores_rows_without_a_teacher_but_trains_on_valid_zero_actions() -> None:
    algo = cast("Any", object.__new__(DistillationPPO))
    algo.distill_loss_fn = torch.nn.functional.mse_loss
    student_mean = torch.ones(2, 1, requires_grad=True)
    # Both teacher actions are zero; only the first row actually has a teacher.
    teacher_actions = torch.zeros(2, 1)
    teacher_valid = torch.tensor([[True], [False]])

    per_sample = algo._dagger_loss_per_sample(student_mean, teacher_actions, teacher_valid)
    per_sample.sum().backward()

    assert per_sample.tolist() == [1.0, 0.0]
    assert student_mean.grad is not None
    assert student_mean.grad[:, 0].tolist() == [2.0, 0.0]


def test_dagger_target_is_unchanged_when_every_row_has_a_teacher() -> None:
    student_mean = torch.randn(3, 2)
    teacher_actions = torch.randn(3, 2)

    target = DistillationBase._dagger_target(student_mean, teacher_actions, torch.ones(3, 1, dtype=torch.bool))

    assert torch.equal(target, teacher_actions)


def test_warmup_steps_envs_without_a_teacher_with_the_student_action() -> None:
    actions = torch.tensor([[5.0], [6.0]])
    teacher_actions = torch.tensor([[1.0], [0.0]])
    teacher_valid = torch.tensor([True, False])

    warm = DistillationBase._step_actions(actions, teacher_actions, teacher_valid, warmup=True)
    cold = DistillationBase._step_actions(actions, teacher_actions, teacher_valid, warmup=False)

    assert warm.tolist() == [[1.0], [6.0]]
    assert torch.equal(cold, actions)


def test_pure_dagger_loss_masks_rows_without_a_teacher_over_the_full_batch() -> None:
    algo = cast("Any", object.__new__(Distillation))
    algo.distill_loss_fn = torch.nn.functional.mse_loss
    student_mean = torch.tensor([[1.0], [3.0]])

    loss = algo._dagger_loss(student_mean, torch.zeros(2, 1), torch.tensor([[True], [False]]))

    # Only the first row has a teacher: (1 - 0)^2 + 0, averaged over both rows.
    assert loss.item() == 0.5
