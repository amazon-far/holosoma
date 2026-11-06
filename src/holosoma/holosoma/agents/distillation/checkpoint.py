"""Read the policy structure of saved distillation checkpoints, and convert legacy ones."""

from __future__ import annotations

from typing import Any


def infer_num_teachers_from_checkpoint(checkpoint: dict[str, Any]) -> int | None:
    """Read the teacher-slot count from a full distillation checkpoint."""
    explicit_count = checkpoint.get("num_teachers")
    if explicit_count is not None:
        if type(explicit_count) is not int or explicit_count < 1:
            raise ValueError(f"Invalid num_teachers checkpoint metadata: {explicit_count!r}")

    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        return explicit_count

    legacy_single_teacher = any(key.startswith("teacher.") for key in state_dict)
    indices: set[int] = set()
    for key in state_dict:
        parts = key.split(".", 2)
        if len(parts) >= 2 and parts[0] == "teachers" and parts[1].isdigit():
            indices.add(int(parts[1]))

    if legacy_single_teacher and indices:
        raise ValueError("Checkpoint mixes legacy 'teacher.*' and current 'teachers.*' keys.")
    if legacy_single_teacher:
        inferred_count = 1
    elif indices:
        expected = set(range(max(indices) + 1))
        if indices != expected:
            raise ValueError(f"Checkpoint has non-contiguous teacher slots: {sorted(indices)}.")
        inferred_count = len(indices)
    else:
        return explicit_count
    if explicit_count is not None and explicit_count != inferred_count:
        raise ValueError(
            f"Checkpoint num_teachers metadata says {explicit_count}, but its state dict "
            f"contains {inferred_count} teacher slots."
        )
    return inferred_count


def convert_legacy_teacher_state(state_dict: dict[str, Any], num_teachers: int) -> dict[str, Any]:
    """Move the teachers of a student checkpoint saved before teachers recorded their construction.

    Those policies built every teacher as a plain MLP stored under ``teachers.<i>.<layer>``
    (``teacher.<layer>`` for the single teacher of the oldest checkpoints), with its observation
    normalizer under ``teacher_obs_normalizers.<i>.*``. Checkpoints older than teacher
    normalization have no normalizer; their teachers read raw observations. Returns the state
    dict with those entries under :class:`~holosoma.agents.modules.teacher_policies.MLPTeacher`'s
    layout, ``teachers.<i>.mlp.*`` and ``teachers.<i>.obs_normalizer.*``, adding identity
    normalizers where none were saved.
    """
    legacy_single = any(key.startswith("teacher.") for key in state_dict)
    if legacy_single and any(key.startswith("teachers.") for key in state_dict):
        raise ValueError("Policy state dict mixes legacy 'teacher.*' and current 'teachers.*' keys.")
    converted: dict[str, Any] = {}
    for key, value in state_dict.items():
        if key.startswith("teacher."):
            converted[f"teachers.0.mlp.{key[len('teacher.') :]}"] = value
        elif key.startswith("teachers."):
            slot, rest = key[len("teachers.") :].split(".", 1)
            converted[f"teachers.{slot}.mlp.{rest}"] = value
        elif key.startswith("teacher_obs_normalizers."):
            slot, rest = key[len("teacher_obs_normalizers.") :].split(".", 1)
            converted[f"teachers.{slot}.obs_normalizer.{rest}"] = value
        else:
            converted[key] = value
    for index in range(num_teachers):
        first_layer = converted.get(f"teachers.{index}.mlp.0.weight")
        if first_layer is None or f"teachers.{index}.obs_normalizer.mean" in converted:
            continue
        num_obs = first_layer.shape[1]
        converted[f"teachers.{index}.obs_normalizer.mean"] = first_layer.new_zeros(1, num_obs)
        converted[f"teachers.{index}.obs_normalizer.std"] = first_layer.new_ones(1, num_obs)
        converted[f"teachers.{index}.obs_normalizer.eps"] = first_layer.new_zeros(())
    return converted


def legacy_mlp_teacher_configs(
    state_dict: dict[str, Any],
    num_teachers: int,
    hidden_dims: list[int],
    activation: str,
    num_actions: int,
) -> list[dict[str, Any] | None]:
    """Construction configs of the MLP teachers in a :func:`convert_legacy_teacher_state` result.

    ``hidden_dims`` and ``activation`` are the student module config's, which those policies built
    their teachers with. Loading the state dict strictly then checks them against the weights.
    """
    configs: list[dict[str, Any] | None] = []
    for index in range(num_teachers):
        first_layer = state_dict.get(f"teachers.{index}.mlp.0.weight")
        if first_layer is None:
            configs.append(None)
            continue
        configs.append(
            {
                "type": "mlp",
                "num_obs": int(first_layer.shape[1]),
                "hidden_dims": list(hidden_dims),
                "num_actions": num_actions,
                "activation": activation,
            }
        )
    return configs
