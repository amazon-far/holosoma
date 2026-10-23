"""IsaacSim drawing adapter - moved from simulator class."""

from __future__ import annotations

from typing import Any, Protocol

import numpy.typing as npt

from holosoma.utils.adapters.draw_utils import convert_to_list, convert_to_tuple
from holosoma.utils.safe_torch_import import torch


class IsaacSim(Protocol):
    """What these drawing helpers actually need from an Isaac Sim backend.

    Previously this module imported ``holosoma.simulator.isaacsim.isaacsim.IsaacSim`` for the
    annotation alone -- every function below is duck-typed, touching only ``.draw`` and
    ``.isaacsim``. That import was a problem once a second Isaac generation existed: it ran at
    *import* time (``utils/draw.py`` imports this module when ``SimulatorType.ISAACSIM`` is
    configured, and that enum value covers both generations), so a process running the 6.1
    backend would eagerly import the 5.1 package -- loading the wrong generation's module graph
    purely to satisfy a type hint.

    A protocol states the real requirement instead, so either generation's simulator satisfies it
    structurally and neither package is imported here.
    """

    draw: Any
    isaacsim: Any


def clear_lines(simulator: IsaacSim) -> None:
    """Clear all debug lines from the viewer."""
    if hasattr(simulator, "draw") and simulator.draw:
        simulator.draw.clear_lines()
        simulator.draw.clear_points()


def draw_sphere(
    simulator: IsaacSim,
    pos: list[float] | tuple[float, float, float] | npt.NDArray[Any] | torch.Tensor,
    radius: float,
    color: list[float] | tuple[float, float, float] | npt.NDArray[Any] | torch.Tensor,
    env_id: int,
    pos_id: int | None = None,
) -> None:
    """Draw a sphere using points with unified type support."""
    if not hasattr(simulator, "draw") or not simulator.draw:
        return

    point_list = [convert_to_tuple(pos)]
    color_list = [convert_to_list(color) + [1.0]]
    sizes = [20]
    simulator.draw.draw_points(point_list, color_list, sizes)


def draw_line(
    simulator: IsaacSim,
    start_point: list[float] | tuple[float, float, float] | npt.NDArray[Any] | torch.Tensor,
    end_point: list[float] | tuple[float, float, float] | npt.NDArray[Any] | torch.Tensor,
    color: list[float] | tuple[float, float, float] | npt.NDArray[Any] | torch.Tensor,
    env_id: int,
) -> None:
    """Draw a line with unified type support."""
    if not hasattr(simulator, "draw") or not simulator.draw:
        return

    start_point_list = [convert_to_tuple(start_point)]
    end_point_list = [convert_to_tuple(end_point)]
    color_list = [convert_to_list(color) + [1.0]]
    sizes = [1]
    simulator.draw.draw_lines(start_point_list, end_point_list, color_list, sizes)


# Set the rest to no-op since we only need these 3
def draw_points(*args: Any, **kwargs: Any) -> None:
    """No-op implementation for draw_points."""


def draw_height_points(*args: Any, **kwargs: Any) -> None:
    """No-op implementation for draw_height_points."""


def draw_foot_height_points(*args: Any, **kwargs: Any) -> None:
    """No-op implementation for draw_foot_height_points."""
