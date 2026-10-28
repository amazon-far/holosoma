from __future__ import annotations

import math
from typing import cast

import pytest

from holosoma.agents.distillation.checkpoint import convert_legacy_teacher_state, legacy_mlp_teacher_configs
from holosoma.agents.modules.student_teacher_modules import MIN_ACTION_STD, StudentTeacher
from holosoma.agents.modules.teacher_policies import MLPTeacher
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim


def _module(*, num_teachers: int = 1, init_noise_std: float = 0.1) -> StudentTeacher:
    return StudentTeacher(
        num_actor_obs=2,
        num_teacher_obs=3,
        num_actions=2,
        student_hidden_dims=[4],
        init_noise_std=init_noise_std,
        num_teachers=num_teachers,
    )


def _mlp_teacher() -> MLPTeacher:
    teacher = MLPTeacher(num_obs=3, hidden_dims=[4], num_actions=2, activation="ELU")
    with torch.no_grad():
        for parameter in teacher.mlp.parameters():
            parameter.zero_()
    return teacher


def _load_constant_teachers(module: StudentTeacher) -> None:
    """Teacher ``i`` outputs ``i + 1`` for every action."""
    for index in range(module.num_teachers):
        teacher = _mlp_teacher()
        with torch.no_grad():
            teacher.mlp[-1].bias.fill_(index + 1)
        module.set_teacher(index, teacher)


def test_teacher_action_requires_loaded_weights() -> None:
    module = _module()

    with pytest.raises(RuntimeError, match="not loaded"):
        module.teacher_act(torch.zeros(2, 3))


def test_rebuilt_teachers_are_not_ready_until_their_readiness_is_restored() -> None:
    module = _module()

    module.build_teachers([_mlp_teacher().construction_config()])

    assert module.loaded_teachers == (False,)
    with pytest.raises(RuntimeError, match="not loaded"):
        module.require_loaded_teachers()
    module.set_loaded_teachers([True])
    module.require_loaded_teachers()


def test_a_teacher_must_map_the_policy_dims() -> None:
    module = _module()

    with pytest.raises(ValueError, match="maps 4 observations"):
        module.set_teacher(0, MLPTeacher(num_obs=4, hidden_dims=[4], num_actions=2, activation="ELU"))


def test_teachers_stay_frozen_in_train_mode() -> None:
    module = _module()
    _load_constant_teachers(module)

    module.train()

    assert module.student.training
    assert not module.teachers[0].training
    assert not any(parameter.requires_grad for parameter in module.teachers.parameters())


def test_multi_teacher_routing_marks_rows_without_a_teacher_invalid() -> None:
    module = _module(num_teachers=2)
    _load_constant_teachers(module)

    actions, valid = module.teacher_act(
        torch.zeros(3, 3),
        torch.tensor([0, 1, -1]),
    )

    torch.testing.assert_close(actions[:2], torch.tensor([[1.0, 1.0], [2.0, 2.0]]))
    assert valid.tolist() == [True, True, False]


def test_a_single_teacher_labels_every_row() -> None:
    module = _module()
    _load_constant_teachers(module)

    _, valid = module.teacher_act(torch.zeros(3, 3))

    assert valid.tolist() == [True, True, True]


def test_multiple_teachers_require_explicit_routing() -> None:
    module = _module(num_teachers=2)
    _load_constant_teachers(module)

    with pytest.raises(ValueError, match="teacher_idx is required"):
        module.teacher_act(torch.zeros(2, 3))


@pytest.mark.parametrize(
    ("teacher_idx", "error", "message"),
    [
        (torch.tensor([[0], [1]]), ValueError, "must have shape"),
        (torch.tensor([True, False]), TypeError, "integer-valued"),
        (torch.tensor([0.0, 0.5]), ValueError, "finite integer-valued"),
        (torch.tensor([0, 2]), IndexError, "invalid values"),
        (torch.tensor([-2, 0]), IndexError, "invalid values"),
    ],
)
def test_invalid_teacher_routing_fails_loudly(
    teacher_idx: torch.Tensor,
    error: type[Exception],
    message: str,
) -> None:
    module = _module(num_teachers=2)
    _load_constant_teachers(module)

    with pytest.raises(error, match=message):
        module.teacher_act(torch.zeros(2, 3), teacher_idx)


def test_teacher_observation_shape_is_exact() -> None:
    module = _module()
    _load_constant_teachers(module)

    with pytest.raises(ValueError, match=r"shape \[N, 3\]"):
        module.teacher_act(torch.zeros(2, 4))


def test_action_std_is_projected_positive() -> None:
    module = _module(init_noise_std=-1.0)

    torch.testing.assert_close(module.std, torch.full((2,), MIN_ACTION_STD))
    module.update_distribution(torch.zeros(3, 2))
    assert torch.all(module.action_std > 0)


def test_non_finite_action_std_is_rejected() -> None:
    with pytest.raises(FloatingPointError, match="non-finite"):
        _module(init_noise_std=math.nan)

    module = _module()
    with torch.no_grad():
        module.std.fill_(math.inf)
    with pytest.raises(FloatingPointError, match="non-finite"):
        module.update_distribution(torch.zeros(1, 2))


def _legacy_state(*, single: bool = False, normalizers: bool = True) -> dict[str, torch.Tensor]:
    """A student state dict as saved before teachers recorded their construction."""
    teacher = _mlp_teacher()
    with torch.no_grad():
        teacher.mlp[-1].bias.fill_(7.0)
    prefix = "teacher." if single else "teachers.0."
    state = {f"{prefix}{key}": value for key, value in teacher.mlp.state_dict().items()}
    if normalizers:
        state.update(
            {f"teacher_obs_normalizers.0.{key}": value for key, value in teacher.obs_normalizer.state_dict().items()}
        )
    state.update({f"student.{key}": value for key, value in _module().student.state_dict().items()})
    return state


@pytest.mark.parametrize("single", [False, True])
def test_legacy_mlp_teachers_are_converted_and_rebuilt(single: bool) -> None:
    converted = convert_legacy_teacher_state(_legacy_state(single=single), num_teachers=1)
    module = _module()
    module.build_teachers(legacy_mlp_teacher_configs(converted, 1, hidden_dims=[4], activation="ELU", num_actions=2))

    module.load_state_dict({**module.state_dict(), **converted}, strict=True)
    module.set_loaded_teachers([True])

    assert isinstance(module.teachers[0], MLPTeacher)
    torch.testing.assert_close(module.teacher_act(torch.zeros(2, 3)).actions, torch.full((2, 2), 7.0))


def test_legacy_teachers_without_normalizers_keep_reading_raw_teacher_obs() -> None:
    converted = convert_legacy_teacher_state(_legacy_state(normalizers=False), num_teachers=1)
    module = _module()
    module.build_teachers(legacy_mlp_teacher_configs(converted, 1, hidden_dims=[4], activation="ELU", num_actions=2))

    module.load_state_dict({**module.state_dict(), **converted}, strict=True)

    obs = torch.randn(4, 3)
    torch.testing.assert_close(module.teachers[0].obs_normalizer(obs), obs, rtol=0.0, atol=0.0)


def test_legacy_teacher_layout_must_match_the_module_config() -> None:
    converted = convert_legacy_teacher_state(_legacy_state(), num_teachers=1)
    module = _module()
    module.build_teachers(legacy_mlp_teacher_configs(converted, 1, hidden_dims=[8], activation="ELU", num_actions=2))

    with pytest.raises(RuntimeError, match="size mismatch"):
        module.load_state_dict({**module.state_dict(), **converted}, strict=True)


def test_mixed_teacher_state_dict_keys_are_rejected() -> None:
    state = _legacy_state()
    state["teacher.0.weight"] = state["teachers.0.0.weight"]

    with pytest.raises(ValueError, match="mixes legacy"):
        convert_legacy_teacher_state(state, num_teachers=1)


def test_teacher_readiness_metadata_is_strict() -> None:
    module = _module(num_teachers=2)

    with pytest.raises(ValueError, match="Expected readiness state"):
        module.set_loaded_teachers([True])
    with pytest.raises(TypeError, match="only booleans"):
        module.set_loaded_teachers(cast("list[bool]", [1, 0]))


def _ppo_normalizer_state(mean: list[float], std: list[float]) -> dict[str, torch.Tensor]:
    return {
        "_mean": torch.tensor([mean]),
        "_var": torch.tensor([std]).square(),
        "_std": torch.tensor([std]),
        "count": torch.tensor(10),
    }


def test_each_teacher_uses_its_own_obs_normalizer() -> None:
    module = _module(num_teachers=2)
    for index in range(2):
        teacher = _mlp_teacher()
        with torch.no_grad():
            teacher.mlp[0].weight[0, 0] = 1.0
            teacher.mlp[-1].weight[0, 0] = 1.0
        if index == 0:
            teacher.obs_normalizer.load_ppo_state_dict(_ppo_normalizer_state([1.0, 0.0, 0.0], [0.99, 1.0, 1.0]))
        module.set_teacher(index, teacher)

    actions = module.teacher_act(torch.full((2, 3), 3.0), torch.tensor([0, 1])).actions

    # Teacher 0 sees (3 - 1) / (0.99 + 0.01) = 2; teacher 1 sees the raw 3.
    torch.testing.assert_close(actions[:, 0], torch.tensor([2.0, 3.0]))


def test_teacher_obs_normalizer_shape_must_match_the_teacher_input() -> None:
    teacher = _mlp_teacher()

    with pytest.raises(ValueError, match="normalizer has shape"):
        teacher.obs_normalizer.load_ppo_state_dict(_ppo_normalizer_state([0.0] * 4, [1.0] * 4))
