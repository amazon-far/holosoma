"""Consumer-side soft contraction of the simulator's hard DOF position limits.

Every backend's ``get_dof_limits_properties`` returns the hard limits its physics enforces (the
``RobotConfig`` lists where given, the asset's values otherwise). Consumers that want a contracted
("soft") position range apply :func:`soft_limits` themselves with their own factor (the
``limits_dof_pos`` reward terms, the WBT initial-pose clamp).
"""

from __future__ import annotations

from holosoma.utils.safe_torch_import import torch


def soft_limits(hard_limits: torch.Tensor, soft_factor: float) -> torch.Tensor:
    """Contract ``[num_dof, 2]`` hard limits toward their midpoints by ``soft_factor``.

    With ``soft_factor == 1`` this is the hard range; smaller factors shrink each fully finite row
    symmetrically. A row with either unbounded side is preserved unchanged: midpoint arithmetic on
    ``[-inf, inf]`` would produce NaNs, and a one-sided range has no finite midpoint to contract around.
    """
    result = hard_limits.clone()
    finite_rows = torch.isfinite(hard_limits).all(dim=1)
    finite = hard_limits[finite_rows]
    if finite.numel() > 0:
        midpoint = (finite[:, 0] + finite[:, 1]) / 2
        width = finite[:, 1] - finite[:, 0]
        result[finite_rows] = torch.stack(
            (midpoint - 0.5 * width * soft_factor, midpoint + 0.5 * width * soft_factor), dim=1
        )
    return result
