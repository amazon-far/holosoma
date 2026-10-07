from __future__ import annotations

import math
from typing import cast

import pytest

from holosoma.agents.modules.student_teacher_modules import MIN_ACTION_STD, StudentTeacher
from holosoma.utils.safe_torch_import import torch

pytestmark = pytest.mark.no_sim


def _module(*, num_teachers: int = 1, init_noise_std: float = 0.1) -> StudentTeacher:
    return StudentTeacher(
        num_actor_obs=2,
        num_teacher_obs=3,
        num_actions=2,
        student_hidden_dims=[4],
        teacher_hidden_dims=[4],
        init_noise_std=init_noise_std,
        num_teachers=num_teachers,
    )


def _load_constant_teachers(module: StudentTeacher) -> None:
    for index, teacher in enumerate(module.teachers):
        state = {key: torch.zeros_like(value) for key, value in teacher.state_dict().items()}
        state[f"{len(teacher) - 1}.bias"].fill_(index + 1)
        module.load_teacher_state_dict(state, teacher_index=index)


def test_teacher_action_requires_loaded_weights() -> None:
    module = _module()

    with pytest.raises(RuntimeError, match="not loaded"):
        module.teacher_act(torch.zeros(2, 3))


def test_partial_non_strict_load_does_not_mark_teacher_ready() -> None:
    module = _module()

    module.load_teacher_state_dict({}, strict=False)

    assert module.loaded_teachers == (False,)
    with pytest.raises(RuntimeError, match="not loaded"):
        module.require_loaded_teachers()


def test_multi_teacher_routing_and_no_teacher_sentinel() -> None:
    module = _module(num_teachers=2)
    _load_constant_teachers(module)

    actions = module.teacher_act(
        torch.zeros(3, 3),
        torch.tensor([0, 1, -1]),
    )

    torch.testing.assert_close(
        actions,
        torch.tensor([[1.0, 1.0], [2.0, 2.0], [0.0, 0.0]]),
    )


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


def test_legacy_single_teacher_state_dict_is_upgraded() -> None:
    source = _module()
    legacy_state = {
        f"teacher.{key[len('teachers.0.') :]}" if key.startswith("teachers.0.") else key: value
        for key, value in source.state_dict().items()
    }
    target = _module()

    target.load_training_state_dict(legacy_state)

    for key, value in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[key], value)


def test_mixed_teacher_state_dict_keys_are_rejected() -> None:
    module = _module()
    state = dict(module.state_dict())
    state["teacher.0.weight"] = state["teachers.0.0.weight"]

    with pytest.raises(ValueError, match="mixes legacy"):
        module.load_training_state_dict(state)


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
    for index, teacher in enumerate(module.teachers):
        state = {key: torch.zeros_like(value) for key, value in teacher.state_dict().items()}
        state["0.weight"][0, 0] = 1.0
        state[f"{len(teacher) - 1}.weight"][0, 0] = 1.0
        normalizer = _ppo_normalizer_state([1.0, 0.0, 0.0], [0.99, 1.0, 1.0]) if index == 0 else None
        module.load_teacher_state_dict(
            {"model_state_dict": state, "actor_obs_normalizer_state_dict": normalizer}, teacher_index=index
        )

    actions = module.teacher_act(torch.full((2, 3), 3.0), torch.tensor([0, 1]))

    # Teacher 0 sees (3 - 1) / (0.99 + 0.01) = 2; teacher 1 sees the raw 3.
    torch.testing.assert_close(actions[:, 0], torch.tensor([2.0, 3.0]))


def test_teacher_obs_normalizer_shape_must_match_teacher_obs() -> None:
    module = _module()
    state = {key: torch.zeros_like(value) for key, value in module.teachers[0].state_dict().items()}

    with pytest.raises(ValueError, match="normalizer has shape"):
        module.load_teacher_state_dict(
            {"model_state_dict": state, "actor_obs_normalizer_state_dict": _ppo_normalizer_state([0.0] * 4, [1.0] * 4)}
        )


def test_checkpoint_without_teacher_normalizers_keeps_raw_teacher_obs() -> None:
    source = _module()
    _load_constant_teachers(source)
    pre_normalizer_state = {
        key: value for key, value in source.state_dict().items() if not key.startswith("teacher_obs_normalizers.")
    }
    target = _module()
    teacher_state = {key: torch.zeros_like(value) for key, value in target.teachers[0].state_dict().items()}
    target.load_teacher_state_dict(
        {
            "model_state_dict": teacher_state,
            "actor_obs_normalizer_state_dict": _ppo_normalizer_state([5.0] * 3, [2.0] * 3),
        }
    )

    target.load_training_state_dict(pre_normalizer_state)

    obs = torch.randn(4, 3)
    torch.testing.assert_close(target.teacher_obs_normalizers[0](obs), obs, rtol=0.0, atol=0.0)
